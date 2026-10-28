# Release notes

This folder is the claim package the organisers accepted on 28 October 2026 (version 7, submitted 23 October). Every file is byte-identical to the submitted one except those listed here. `SHA256SUMS` matches this copy.

| change | files |
|---|---|
| cut down to indexes; the method now lives in the repository's `docs/METHOD.md` | `README.md`, `method/METHOD.md` |
| rewritten for readers outside the team | `method/model/exclusion/README.md`, `docker/README.md`, `supporting/README.md` |
| header comments: removed "PRIVATE, owner only" and similar notes; `code/training/SOURCES.sha256` updated to match | 42 scripts in `code/` and `docker/` |
| comments rewritten; `VC3D_URL` points at our copy of the AppImage, since its release is no longer on GitHub; the apt snapshot now also applies to the install step (it covered only `apt-get update`) and is the first build's date; PyTorch's CUDA packages, triton and torchvision are pinned to the versions the first build resolved (`constraints-torch.txt` is new) | `docker/Dockerfile`, `docker/constraints.txt`, `docker/constraints-torch.txt`, `docker/Dockerfile.from_v1`, `docker/build.sh` |
| corrected (below) | `code/models/v8Q/MODEL_CARD.md`, `method/checks/letter_dimensions.md` |
| removed: a copy of a private email to the organisers, and our previous version kept for the reviewers' comparison | `update_email_2026-10-23.txt`, `_REFERENCE_LAST_SUBMISSION/` |
| added: the licences of everything inside the Docker image (the image is unchanged) | `docker/THIRD_PARTY_NOTICES.md`, `docker/GPL-3.0.txt` |
| added | `RELEASE_NOTES.md` |

Corrections to the submitted text (sections refer to `docs/METHOD.md`):
- The papyrologist's tracings were never training labels, but they were used as checks (section 7).
- No model trained on PHerc. 0841, but TRACE v3B's authors chose their checkpoint on it.
- TRACE v3B is one of A-hn's teachers, so the two models are not independent.
- The surface touch-up was adopted after our pre-registered rule rejected it (section 7).
- The decoy draws blurrier strokes, not noisier ones.
- Letter boxes are 1.2 to 3.8 mm tall.
- Our files are CC BY-NC 4.0; the submitted copy said CC BY-NC-SA 4.0.
- The v8Q model card said the tracings chose nothing, but they gated v8Q. Its section on use described version 5.
- The letter table now says why three faint letters are missing from it.

Apart from those comments, the scripts are the ones in the shipped Docker image.
