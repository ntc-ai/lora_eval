# Publishing SDXL slider updates

Use a pull request to submit a reviewed model card, preview image, or weight for an
existing public `ntc-ai/SDXL-LoRA-slider.*` model. Training and evaluation happen
before the release. GitHub exchanges a short-lived token for one exact Hub model
when its manifest is merged to `master`.

## Enable the first model

1. In `ntc-ai/lora_eval`, create the `sdxl-model-publish` GitHub environment and
   allow deployment only from the exact `master` branch.
2. On the public Chiaroscuro model's Hugging Face settings page, add a Trusted
   Publisher with `repo.content.write` and the issuer
   `https://token.actions.githubusercontent.com`. Match each claim exactly.

   | Claim | Value |
   | --- | --- |
   | `repository` | `ntc-ai/lora_eval` |
   | `repository_id` | `727418694` |
   | `ref` | `refs/heads/master` |
   | `workflow_ref` | `ntc-ai/lora_eval/.github/workflows/sdxl-release.yml@refs/heads/master` |
   | `environment` | `sdxl-model-publish` |
   | `event_name` | `push` |
   | `runner_environment` | `github-hosted` |

3. Merge the checked PR. Its first release points the Chiaroscuro card thumbnail
   to an image already in the published model. Verify the exact new card and
   the retained weight digest. An empty `files` list is available for checking
   a model's parent and upload permission before preparing a changed release.
4. Add the same GitHub claim policy on each additional public model when its
   reviewed release is ready. Each policy belongs to the exact model named in
   the manifest. Private repositories use their separate release process.

## Prepare a release

Put `manifest.json` and the exact files for the release under `releases/<release>/`. Include one manifest per target in each pull request. The example uses placeholders for the current Hub commit and the payload digest. Obtain the commit from the model's Files and versions view and compute the file digest with `sha256sum`.

```json
{
  "target": "ntc-ai/SDXL-LoRA-slider.Example",
  "parent": "0000000000000000000000000000000000000000",
  "files": [
    {
      "source": "weight.safetensors",
      "path": "slider.safetensors",
      "sha256": "1111111111111111111111111111111111111111111111111111111111111111"
    }
  ]
}
```

The pull request checks the payloads and shows which files would change using public Hub access. For a local preview, install `huggingface_hub==1.19.0` and run `python publish_sdxl.py --manifest releases/<release>/manifest.json`.

After the merge to `master`, the publish job sends only changed files and
requires the reviewed Hub parent commit. If the Hub revision has moved, review
the current files and update the manifest before merging again. Identical files
are skipped even on a retry. The job verifies the resulting Hub revision, the
reviewed file digests, and the identity of the other released files. It records
both the GitHub and Hub commits for uploads. Consumers can pin the previous Hub
parent during recovery.
