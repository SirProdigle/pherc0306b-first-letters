# Credits and licences

## Our work

Copyright 2026 Liam Warner and Nicola Bodill.

| what | licence |
|---|---|
| code | MIT ([`LICENSE`](../LICENSE)) |
| model weights, training records, machine labels, maps, renders, meshes and images | CC BY-NC 4.0 ([`LICENSE-DATA`](../LICENSE-DATA)) |

All of it is derived from the Vesuvius Challenge open data, which is CC BY-NC 4.0. Please credit the Vesuvius Challenge as well as us.

## What we used

| item | used for | source | licence | by |
|---|---|---|---|---|
| PHerc. 0306B scan, volume `20250521133212` | all CT | Vesuvius Challenge open data | CC BY-NC 4.0 | Vesuvius Challenge, ESRF |
| m7 surface prediction, VC3D, villa tools | segmentation and rendering | Vesuvius Challenge; GitHub `ScrollPrize/villa` | data CC BY-NC 4.0; villa MIT; VC3D GPLv3 | Vesuvius Challenge |
| PHerc. 1447 surfaces w058 and w060, team labels | human half of training | HF `YoussefMoNader/ink-8um-pherc1447-surfaces` | CC BY-NC 4.0 | Vesuvius Challenge; Youssef Nader et al. |
| PHerc. 0841 and PHerc. 0009B organiser labels | held-out scoring | Vesuvius Challenge open data | CC BY-NC 4.0 | Vesuvius Challenge |
| v8-in | starting weights; teacher | HF `YoussefMoNader/ink-8um-v8in` | MIT | Youssef Nader |
| TRACE v3B | half of the claim rule; teacher | GitHub `Jashann/vesuvius-scrolling`, release v1.0 | code MIT; weights CC BY-NC-SA 4.0 | Jashann |
| Reader V3 | teacher | HF `domenicor046/reader-v3` | MIT | domenicor046 |
| Reader v2 | teacher; screening | HF `domenicor046/reader-v2` | MIT | domenicor046 |
| Hecate 9.6 µm | screening only | HF `scrollprize/hecate` | MIT | Vesuvius Challenge |
| PHerc. 0343 stroke tracings | comparison models B and C only | GitHub `jeff-j-chen` | MIT | jeff-j-chen |
| PHerc. 0139 organiser labels | comparison models B and C only | Vesuvius Challenge open data | CC BY-NC 4.0 | Vesuvius Challenge |

Our screening reused Erwin Nieuwlaar's open PHerc. 343 First Letters recipe (spiral-fit snapping and ink scripts). We also used it to check our pipeline on known letters.

The Docker image contains more third-party software. [`submission/docker/THIRD_PARTY_NOTICES.md`](../submission/docker/THIRD_PARTY_NOTICES.md) lists it.
