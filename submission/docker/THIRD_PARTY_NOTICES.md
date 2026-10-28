# Third-party notices for the Docker image `fl-claim-a7:final`

The image `fl-claim-a7_final.tar.zst` (sha256 `1832832330ab67b134795eaedd43d6d5c3dcf004596e1fd287d2e71b3af73536`) has no
licence files of its own apart from villa's MIT licence, so this file lists the licence of each component:

| component | where in the image | licence | source |
|---|---|---|---|
| VC3D AppImage (`vc_render_tifxyz`), build `a4979b9`, 13 Oct 2026 | `/opt/vc3d` | GPLv3 (`GPL-3.0.txt`) | binary: first released at https://github.com/ScrollPrize/villa/releases, now in the HF dataset `Prodigle/pherc0306b-first-letters` (`docker/build/VC3D-a4979b9-2026-10-13-linux-x86_64.AppImage`, sha256 `e897cdc9…794c`); corresponding source: https://github.com/ScrollPrize/villa at commit `a4979b9` (`volume-cartographer/`) |
| villa ink-detection sources (`vesuvius/src`) | `/opt/villa` | MIT (Vesuvius Challenge) | https://github.com/ScrollPrize/villa at commit `a329895ea5e9bde78752e60dbc0a4878dd2ab909` |
| v8-in weights and loader | `/opt/fl/third_party/v8in/` | MIT (Youssef Nader) | HF `YoussefMoNader/ink-8um-v8in` |
| Reader v2, Reader V3 weights | `/opt/fl/models/community/` | MIT (domenicor046) | HF `domenicor046/reader-v2`, `domenicor046/reader-v3` |
| TRACE v3B weights | `/opt/fl/models/community/trace/` | CC BY-NC-SA 4.0 (Jashann) | GitHub `Jashann/vesuvius-scrolling`, release v1.0 |
| Hecate 9.6 µm weights and code (kept from our first version's image; the claim does not use it) | `/opt/fl/models/ink9_ens/` | MIT (Vesuvius Challenge) | HF `scrollprize/hecate` |
| A-hn members, v8Q (our fine-tunes) | `/opt/fl/third_party/v8in/finalist/`, `/opt/fl/third_party/v8in/v8Q_final.safetensors` | CC BY-NC 4.0 (ours) | HF `Prodigle/pherc0306b-ink-ahn` |
| our code | `/opt/fl` (`claim/`, `final/`, `train/`, `x_*/`) | MIT (ours) | GitHub `SirProdigle/pherc0306b-first-letters` |
| claim maps, CT layer, mesh | `/opt/fl/data`, `/opt/fl/mesh` | CC BY-NC 4.0 | Vesuvius Challenge open data (PHerc. 0306B, CC BY-NC 4.0) and ours |
| zlib-ng-compat 2.3.3 `libz.so.1` | `/opt/zlib-ng` | zlib licence | https://github.com/zlib-ng/zlib-ng |
| Python 3.14, PyTorch and the pip packages in `constraints.txt` | `/opt/python`, `/opt/venv` | their own licences (BSD, MIT, Apache 2.0 and others) | PyPI and download.pytorch.org at the pinned versions |
| Ubuntu 24.04 base and its packages | `/` | their own licences (`/usr/share/doc/*/copyright` inside the image) | Ubuntu archive snapshot 20261012 |
