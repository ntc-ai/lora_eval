import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from publish_sdxl import Payload, github_source_revision, git_changes, load_manifest, publisher, release


TARGET = "ntc-ai/SDXL-LoRA-slider.Example"
PARENT = "0" * 40
REVISION = "1" * 40


def metadata(path, data, *, lfs=False):
    sha256 = hashlib.sha256(data).hexdigest()
    payload = Payload(path, data, sha256)
    return SimpleNamespace(
        rfilename=path, size=len(data), blob_id=payload.git_blob,
        lfs=SimpleNamespace(sha256=sha256) if lfs else None,
    )


def model_info(revision, *files):
    return SimpleNamespace(id=TARGET, private=False, disabled=False, sha=revision, siblings=files)


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "release"
        self.root.mkdir()
        self.manifest = self.root / "manifest.json"
        self.weight = b"sample weight bytes"
        self.old_card = metadata("README.md", b"Model card")
        (self.root / "weight.safetensors").write_bytes(self.weight)
        self.spec = {
            "target": TARGET,
            "parent": PARENT,
            "files": [{
                "source": "weight.safetensors",
                "path": "slider.safetensors",
                "sha256": hashlib.sha256(self.weight).hexdigest(),
            }],
        }
        self.api = Mock()
        self.writer = Mock()
        self.api.model_info.return_value = model_info(PARENT, self.old_card)
        self.writer.create_commit.return_value = SimpleNamespace(oid=REVISION)

    def read_manifest(self):
        self.manifest.write_text(json.dumps(self.spec))
        return load_manifest(self.manifest)

    def run_release(self, **kwargs):
        return release(*self.read_manifest(), self.api, **kwargs)

    def test_plan_reads_public_metadata_without_publishing(self):
        result = self.run_release()
        self.assertEqual(result["status"], "planned")
        self.assertEqual(result["changed"], ["slider.safetensors"])
        self.api.model_info.assert_called_once_with(TARGET, revision="main", files_metadata=True)
        self.api.create_commit.assert_not_called()

    def test_publish_uses_exact_payload_and_parent(self):
        released_weight = metadata("slider.safetensors", self.weight, lfs=True)
        self.api.model_info.side_effect = [
            model_info(PARENT, self.old_card),
            model_info(REVISION, self.old_card, released_weight),
        ]
        result = self.run_release(writer=self.writer, source_sha="2" * 40)
        self.assertEqual(result["revision"], REVISION)
        self.assertEqual(result["status"], "published")
        self.writer.create_commit.assert_called_once()
        args, kwargs = self.writer.create_commit.call_args
        target, operations = args
        self.assertEqual(target, TARGET)
        self.assertEqual(kwargs["parent_commit"], PARENT)
        self.assertEqual(kwargs["revision"], "main")
        self.assertEqual(len(operations), 1)
        self.assertEqual(operations[0].path_in_repo, "slider.safetensors")
        self.assertEqual(operations[0].path_or_fileobj, self.weight)
        self.api.model_info.assert_called_with(TARGET, revision="main", files_metadata=True)
        self.api.create_commit.assert_not_called()

    def test_identical_payload_is_a_noop_with_either_storage_type(self):
        for lfs in (True, False):
            with self.subTest(lfs=lfs):
                self.api.model_info.return_value = model_info(
                    REVISION, self.old_card, metadata("slider.safetensors", self.weight, lfs=lfs)
                )
                self.assertEqual(self.run_release(writer=self.writer)["status"], "unchanged")
        self.writer.create_commit.assert_not_called()

    def test_changed_payload_requires_current_reviewed_parent(self):
        self.api.model_info.return_value = model_info(REVISION, self.old_card)
        with self.assertRaisesRegex(ValueError, "moved from the reviewed parent"):
            self.run_release(writer=self.writer)
        self.writer.create_commit.assert_not_called()

    def test_verification_detects_a_lost_file_or_unexpected_bytes(self):
        for files, error in (
            ((metadata("slider.safetensors", self.weight),), "missing existing files"),
            ((self.old_card, metadata("slider.safetensors", b"wrong data")), "did not match"),
            ((metadata("README.md", b"other card"), metadata("slider.safetensors", self.weight)), "unreviewed changes"),
        ):
            with self.subTest(error=error):
                self.api.model_info.side_effect = [
                    model_info(PARENT, self.old_card), model_info(REVISION, *files)
                ]
                with self.assertRaisesRegex(RuntimeError, error):
                    self.run_release(writer=self.writer)

    def test_private_and_outside_namespace_models_are_rejected(self):
        private = model_info(PARENT)
        private.private = True
        self.api.model_info.return_value = private
        with self.assertRaisesRegex(ValueError, "existing public model"):
            self.run_release(writer=self.writer)
        self.spec["target"] = "other/SDXL-LoRA-slider.Example"
        with self.assertRaisesRegex(ValueError, "Invalid SDXL model"):
            self.run_release(writer=self.writer)
        self.writer.create_commit.assert_not_called()

    def test_baseline_verifies_exact_parent_without_writing(self):
        self.spec["files"] = []
        self.assertEqual(self.run_release(writer=self.writer),
                         {"target": TARGET, "revision": PARENT, "status": "baseline"})
        self.api.model_info.return_value = model_info(REVISION, self.old_card)
        with self.assertRaisesRegex(ValueError, "moved from the reviewed parent"):
            self.run_release(writer=self.writer)
        self.writer.create_commit.assert_not_called()

    def test_write_context_requires_the_canonical_repository_and_checkout(self):
        context = {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "ntc-ai/lora_eval",
                   "GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/master", "GITHUB_SHA": REVISION}
        with patch.dict("publish_sdxl.os.environ", context, clear=True), \
                patch("publish_sdxl.subprocess.check_output", return_value=REVISION + "\n"):
            self.assertEqual(github_source_revision(), REVISION)
            with patch.dict("publish_sdxl.os.environ", {"GITHUB_REPOSITORY": "fork/lora_eval"}):
                with self.assertRaisesRegex(ValueError, "canonical GitHub workflow"):
                    github_source_revision()
            with patch.dict("publish_sdxl.os.environ", {"GITHUB_SHA": PARENT}):
                with self.assertRaisesRegex(ValueError, "checked-out commit"):
                    github_source_revision()

    def test_publisher_overrides_other_scopes_and_uses_the_upload_preflight(self):
        env = {"HF_TOKEN": "local-test-token", "HF_OIDC_RESOURCE": "other/model"}
        with patch.dict("publish_sdxl.os.environ", env, clear=True), \
                patch("publish_sdxl.subprocess.run", return_value=SimpleNamespace(stdout="hf_jwt_test\n", returncode=0)) as run, \
                patch("publish_sdxl.get_session") as session, patch("publish_sdxl.hf_raise_for_status"), \
                patch("builtins.print"):
            session.return_value.post.return_value.json.return_value = {"commitOid": REVISION}
            self.assertEqual(publisher(TARGET).token, "hf_jwt_test")
            exchanged = run.call_args.kwargs["env"]
            self.assertEqual(exchanged["HF_OIDC_RESOURCE"], TARGET)
            self.assertNotIn("HF_TOKEN", exchanged)
            session.return_value.post.assert_called_once_with(
                f"https://huggingface.co/api/models/{TARGET}/preupload/main", timeout=20,
                headers={"Authorization": "Bearer hf_jwt_test"}, json={"files": []},
            )

    def test_hash_mismatch_is_rejected(self):
        self.spec["files"][0]["sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.run_release()
        self.api.model_info.assert_not_called()

    def test_directory_traversal_and_outside_symlink_are_rejected(self):
        outside = self.root.parent / "outside.safetensors"
        outside.write_bytes(self.weight)
        (self.root / "symlink.safetensors").symlink_to(outside)
        for path, error in (("../outside.safetensors", "canonical relative"),
                            ("symlink.safetensors", "inside release")):
            with self.subTest(path=path):
                self.spec["files"][0]["source"] = path
                with self.assertRaisesRegex(ValueError, error):
                    self.run_release()
        self.api.model_info.assert_not_called()

    def test_asset_only_change_selects_the_release_directory(self):
        with TemporaryDirectory() as directory:
            def git(*args):
                return subprocess.check_output(["git", "-C", directory, *args], text=True).strip()

            git("init", "-q")
            git("config", "user.name", "Release Test")
            git("config", "user.email", "release-test@example.invalid")
            git("config", "commit.gpgsign", "false")
            git("config", "core.hooksPath", "/dev/null")
            asset = Path(directory) / "releases" / "test release" / "weight.safetensors"
            asset.parent.mkdir(parents=True)
            asset.write_bytes(b"first")
            git("add", ".")
            git("commit", "-qm", "Initial artifact")
            before = git("rev-parse", "HEAD")
            asset.write_bytes(b"second")
            git("add", ".")
            git("commit", "-qm", "Changed artifact")

            with patch("publish_sdxl.subprocess.check_output", wraps=subprocess.check_output) as run:
                run.side_effect = lambda args: subprocess.run(
                    args, cwd=directory, check=True, stdout=subprocess.PIPE
                ).stdout
                self.assertEqual(git_changes(before), [Path("releases/test release")])


if __name__ == "__main__":
    unittest.main()
