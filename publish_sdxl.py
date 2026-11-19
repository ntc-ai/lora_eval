#!/usr/bin/env python3
"""Validate or publish reviewed files to an existing SDXL slider model."""

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from tempfile import TemporaryDirectory

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.utils import get_session, hf_raise_for_status


MODEL_ID = r"ntc-ai/SDXL-LoRA-slider\.[A-Za-z0-9][A-Za-z0-9_.-]{0,90}"
GIT_SHA = r"[0-9a-f]{40}"
HUB = "https://huggingface.co"
SOURCE = "ntc-ai/lora_eval"


@dataclass(frozen=True)
class Payload:
    target_path: str
    data: bytes
    sha256: str

    @property
    def git_blob(self):
        header = f"blob {len(self.data)}\0".encode()
        return hashlib.sha1(header + self.data, usedforsecurity=False).hexdigest()


def relative_path(value):
    if not isinstance(value, str):
        raise ValueError("Release paths must be strings")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts or str(path) != value or "\\" in value:
        raise ValueError(f"Use a canonical relative path for {value!r}")
    return path


def load_manifest(path):
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict) or manifest.keys() != {"target", "parent", "files"}:
        raise ValueError(f"Expected target, parent and files in {path}")
    target, parent, files = manifest["target"], manifest["parent"], manifest["files"]
    if (not isinstance(target, str) or not re.fullmatch(MODEL_ID, target)
            or not isinstance(parent, str) or not re.fullmatch(GIT_SHA, parent)):
        raise ValueError(f"Invalid SDXL model or Hub parent in {path}")
    if not isinstance(files, list):
        raise ValueError(f"Expected a release file list in {path}")

    root = path.parent.resolve()
    payload = {}
    for item in files:
        if not isinstance(item, dict) or item.keys() != {"source", "path", "sha256"}:
            raise ValueError("Each file needs source, path and sha256")
        target_path = relative_path(item["path"])
        model_card = target_path == PurePosixPath("README.md")
        weight = len(target_path.parts) == 1 and target_path.suffix == ".safetensors"
        image = (len(target_path.parts) == 2 and target_path.parts[0] == "images"
                 and target_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
        if not (model_card or weight or image) or str(target_path) in payload:
            raise ValueError(f"Unsupported or duplicate destination {target_path}")
        source = (root / relative_path(item["source"])).resolve(strict=True)
        if not source.is_relative_to(root) or not source.is_file():
            raise ValueError(f"Release input must be a file inside {root.name}")
        data = source.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if sha != item["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {item['source']}")
        payload[str(target_path)] = Payload(str(target_path), data, sha)
    return target, parent, payload


def matches(remote_file, local):
    if remote_file is None or remote_file.size != len(local.data):
        return False
    if remote_file.lfs:
        return remote_file.lfs.sha256 == local.sha256
    return remote_file.blob_id == local.git_blob


def identity(file):
    return file.size, file.lfs.sha256 if file.lfs else file.blob_id


def release(target, parent, payload, api, *, writer=None, source_sha=None):
    current = api.model_info(target, revision="main", files_metadata=True)
    if current.id != target or current.private or current.disabled:
        raise ValueError(f"Expected an existing public model at {target}")
    remote = {f.rfilename: f for f in current.siblings}
    changes = [file for path, file in payload.items() if not matches(remote.get(path), file)]
    if payload and not changes:
        return {"target": target, "revision": current.sha, "status": "unchanged"}
    if current.sha != parent:
        raise ValueError(f"{target} moved from the reviewed parent {parent} to {current.sha}")
    if not payload:
        return {"target": target, "revision": current.sha, "status": "baseline"}
    if writer is None:
        return {"target": target, "revision": parent, "status": "planned",
                "changed": [file.target_path for file in changes]}

    operations = [CommitOperationAdd(path_in_repo=file.target_path, path_or_fileobj=file.data) for file in changes]
    commit = writer.create_commit(target, operations, revision="main", parent_commit=parent,
                                  commit_message=f"Publish reviewed SDXL files from Git {source_sha}")
    published = api.model_info(target, revision="main", files_metadata=True)
    files = {f.rfilename: f for f in published.siblings}
    if published.sha != commit.oid or any(not matches(files.get(path), file) for path, file in payload.items()):
        raise RuntimeError(f"Published files did not match the reviewed manifest at {commit.oid}")
    if not remote.keys() <= files.keys():
        raise RuntimeError(f"The published revision {commit.oid} is missing existing files")
    updated_paths = {file.target_path for file in changes}
    if (
        files.keys() - remote.keys() - updated_paths - {".gitattributes"}
        or any(identity(files[path]) != identity(remote[path])
               for path in remote.keys() - updated_paths - {".gitattributes"})
    ):
        raise RuntimeError(f"The published revision {commit.oid} contains unreviewed changes")
    return {"target": target, "revision": commit.oid, "status": "published", "source": source_sha}


def git_changes(before):
    if not re.fullmatch(GIT_SHA, before):
        raise ValueError("A full previous Git revision is required")
    raw = subprocess.check_output(
        ["git", "diff", "--no-renames", "--name-only", "-z", before, "HEAD", "--", "releases/"]
    )
    directories = set()
    for name in raw.decode().split("\0"):
        if name:
            parts = relative_path(name).parts
            if len(parts) < 3:
                raise ValueError("Put each artifact release in releases/<release>/")
            directories.add(Path(*parts[:2]))
    return sorted(directories)


def github_source_revision():
    if (
        os.getenv("GITHUB_ACTIONS") != "true" or os.getenv("GITHUB_REPOSITORY") != SOURCE
        or os.getenv("GITHUB_EVENT_NAME") != "push" or os.getenv("GITHUB_REF") != "refs/heads/master"
    ):
        raise ValueError("Publishing requires the canonical GitHub workflow on master")
    sha = os.environ.get("GITHUB_SHA", "")
    checkout = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if not re.fullmatch(GIT_SHA, sha) or checkout != sha:
        raise ValueError("The checked-out commit must match the GitHub release event")
    return sha


def publisher(target):
    if not re.fullmatch(MODEL_ID, target):
        raise ValueError("Expected a single SDXL model to publish")
    # Obtain a fresh credential for this target only. A clean HF home avoids
    # accidentally reusing a local or shared login.
    with TemporaryDirectory() as cache:
        env = {key: value for key, value in os.environ.items()
               if key not in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH", "HF_OIDC_ID_TOKEN"}}
        env.update(HF_HOME=cache, HF_OIDC_RESOURCE=target, HF_ENDPOINT=HUB)
        exchanged = subprocess.run(["hf", "auth", "token"], env=env, capture_output=True, text=True, timeout=30)
    token = exchanged.stdout.strip()
    if exchanged.returncode or not re.fullmatch(r"hf_jwt_[A-Za-z0-9_.-]+", token):
        raise RuntimeError(f"Failed to obtain a short-lived publisher credential for {target}")
    print(f"::add-mask::{token}", flush=True)
    response = get_session().post(f"{HUB}/api/models/{target}/preupload/main", timeout=20,
                                  headers={"Authorization": f"Bearer {token}"}, json={"files": []})
    hf_raise_for_status(response)
    preflight_sha = response.json()["commitOid"]
    if not re.fullmatch(GIT_SHA, preflight_sha):
        raise RuntimeError(f"Unexpected upload preflight revision for {target}")
    print(f"Verified upload preflight for {target} at {preflight_sha}", flush=True)
    return HfApi(endpoint=HUB, token=token)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--changed-from", help="Select directories changed since this Git revision")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    sha = github_source_revision() if args.apply else None
    api = HfApi(endpoint=HUB, token=False)
    if args.manifest:
        manifests = [args.manifest]
    else:
        directories = git_changes(args.changed_from)
        if args.apply and directories:
            subprocess.run(["git", "fetch", "--no-tags", "origin", "master"], check=True)
            unchanged = subprocess.run(["git", "diff", "--quiet", "FETCH_HEAD", "HEAD", "--",
                                        *(str(d) for d in directories)]).returncode == 0
            if not unchanged:
                raise ValueError("Selected release inputs changed in current master")
        manifests = [directory / "manifest.json" for directory in directories if directory.exists()]
    releases = [load_manifest(manifest) for manifest in manifests]
    if len(releases) != len({target for target, _, _ in releases}):
        raise ValueError("Select one manifest per model in each release")
    plans = [release(target, parent, payload, api) for target, parent, payload in releases]
    for (target, parent, payload), plan in zip(releases, plans, strict=True):
        result = plan
        if args.apply:
            writer = publisher(target)
            result = release(target, parent, payload, api, writer=writer, source_sha=sha)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
