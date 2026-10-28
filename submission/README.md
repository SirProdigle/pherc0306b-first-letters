# Claim package: PHerc. 0306B

The First Letters claim for PHerc. 0306B, accepted by the organisers on 28 October 2026. We submitted it on 23 October as our version 7. [`RELEASE_NOTES.md`](RELEASE_NOTES.md) lists how this copy differs from the submitted one, and `SHA256SUMS` covers every file.

The reading and the checks are in the [repository README](../README.md), and the method is in [`docs/METHOD.md`](../docs/METHOD.md).

## Start here

| file | what it is |
|---|---|
| `PHerc0306B_20250521133212_w074own_touchup_V5Rc.png` | the claim image; its `.json` records every input checksum and setting |
| `papyrologist_reading_2026-10-23.png` | the claim box with our papyrologist's reading typed above and below it |
| `evidence_grid_line1_at_papyrologist_letters.jpg` | each letter of line 1, one column per letter, shown in 26 ink maps |
| `ink_over_CT.jpg` | the claim ink over the CT of the papyrus, with line 1 enlarged |
| `alternate_denoisings.jpg` | the comparison models A, B and C beside the claim image |

## How the rules are met

| rule | how | where |
|---|---|---|
| one static image of the text region | one PNG, ink on a plain page | the claim image |
| at least 10 legible letters within 4 cm² | a 3.60 cm² box | the claim image, `papyrologist_reading_2026-10-23.png` |
| made by a program from the CT and the submitted mesh, without manual annotation | code draws the image; no letter is drawn or used as a training label | `code/`, `docker/` |
| image named after the mesh | `PHerc0306B_20250521133212_w074own_touchup_V5Rc.png` | `mesh/` |
| mesh in tifxyz, with a low-distortion flattening | median stretch in the box 0.995 along the line, 1.000 across | `mesh/distortion_in_box.json` |
| 1 cm scale bar | bottom left | the claim image |
| pixel and letter sizes | 8.64 µm per pixel; letter boxes 1.2 to 3.8 mm tall | `method/checks/letter_dimensions.md` |
| rows marked without covering the text | ticks in the margins | the claim image |
| eligible volume only | all CT from volume `20250521133212` | the claim image's `.json` |
| no overlap between training and prediction | exclusion zone; every training tile checked in 3-D | `method/model/exclusion_audit.json` |
| false-positive mitigation | depth shuffle, blank control, decoy | `method/checks/` |
| held-out validation | PHerc. 0841 known text | `method/checks/known_text/` |
| reproducible method | code as it ran; a Docker image, in the Hugging Face dataset repo | `code/`, `docker/` |

## Folders

| folder | contents |
|---|---|
| `mesh/` | the tifxyz mesh, its region copy, the box area and the distortion figures |
| `method/` | the records behind every step (training, exclusion zone, checks, letter sizes, Docker test logs), indexed in [`METHOD.md`](method/METHOD.md) |
| `code/` | the image, control, surface and training code, as it ran |
| `docker/` | the Docker recipe and the licences of everything inside the image |
| `supporting/` | per-model views, controls, letter grids and the papyrologist's tracings |
