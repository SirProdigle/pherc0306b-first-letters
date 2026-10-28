# Supporting scripts

These made the files the Docker image does not rebuild. They run in our working tree and import its geometry and map helpers.

| script | what it made |
|---|---|
| `letter_table_v7.py` | `method/checks/letter_dimensions.md` and `.json`: each letter's box is the claim image's ink, at half tone or darker, inside our papyrologist's traced letter (23 October tracing), widened 0.15 mm |
| `grids_final.py`, with `evidence_grid_line1.py` and `evidence_grid_dn.py` | the letter grid in the root, the two grids in `supporting/letter_grids/`, and their layouts in `method/checks/letter_grids/` |
| `banner_models.py` | the reading-aid banner in the header of each page in `supporting/models/`, and the matching sha256 in each page's `.json` |
