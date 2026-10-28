# fl-claim-a7: reproduces the PHerc. 0306B First Letters claim image (version 7, TRACE v3B x A-hn).
#
# Modes (run.sh): final (default) redraws the claim image and 19 other views from the shipped maps and checks each PNG's sha256;
# smoke re-runs the three A-hn seeds on a CT crop; exclusion re-checks every training tile; pipeline and train need a GPU.
#
# Build context: docker/build.sh in the GitHub repo assembles it from the repo and the Hugging Face repos (every file sha256-checked,
# listed in docker/context.tsv). Everything else is pinned here: the base image digest, the Ubuntu archive snapshot, uv, Python,
# every pip package (constraints.txt), the villa commit, the VC3D AppImage and each public checkpoint (revision or release, and sha256).
ARG BASE=ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55
FROM ghcr.io/astral-sh/uv:0.12.9@sha256:8b940d3a9d65bed080436972241af2e21c84b5e8c9193f7014ed71479ee795ff AS uv
FROM ${BASE}
ARG APT_SNAPSHOT=20261014T000000Z
ARG PYTHON_VERSION=3.14.7
ARG TORCH_VERSION=2.14.1
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
ARG VILLA_SHA=a329895ea5e9bde78752e60dbc0a4878dd2ab909
ARG VC3D_URL=https://huggingface.co/datasets/Prodigle/pherc0306b-first-letters/resolve/f435998b55269762942a979052044309b29650ac/docker/build/VC3D-a4979b9-2026-10-13-linux-x86_64.AppImage
ARG VC3D_SHA256=e897cdc95b6f3e30335367d09385f8c23c88f06ba66fcbb34c7632d5e992794c
ENV DEBIAN_FRONTEND=noninteractive FL_ROOT=/opt/fl VILLA_SRC=/opt/villa/vesuvius/src \
    UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy UV_NO_CACHE=1 \
    PATH=/opt/venv/bin:/opt/vc3d/usr/bin:$PATH OPENBLAS_NUM_THREADS=1 HOME=/tmp
COPY --from=uv /uv /usr/local/bin/uv

# OS packages (the AppImage's runtime libraries) from the Ubuntu archive snapshot taken just before our first build (14 Oct 2026,
# 00:54 UTC). ca-certificates comes first from the live archive, since the snapshot is served over https, and then goes back to
# the snapshot's version.
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends ca-certificates >/dev/null \
 && mkdir -p /opt/fl/env && (apt-get update --snapshot ${APT_SNAPSHOT} -qq -o APT::Update::Error-Mode=any \
       && echo "APT::Snapshot \"${APT_SNAPSHOT}\";" > /etc/apt/apt.conf.d/50snapshot && echo "apt snapshot ${APT_SNAPSHOT}" > /opt/fl/env/apt_source \
     || { echo "WARNING: apt snapshot unavailable, current archive used" | tee /opt/fl/env/apt_source; apt-get update -qq; }) \
 && CA=$(apt-cache madison ca-certificates | awk -F'|' 'NR==1{gsub(/ /,"",$2); print $2}') \
 && apt-get install -y -qq --no-install-recommends --allow-downgrades "ca-certificates=$CA" curl git file procps libgl1 libglib2.0-0t64 \
      libfreetype6 libfontconfig1 libxkbcommon0 libdbus-1-3 libegl1 libopengl0 libx11-6 libxcb1 libgomp1 >/dev/null \
 && rm -rf /var/lib/apt/lists/*

# Python environment
COPY constraints.txt constraints-torch.txt requirements.txt /opt/fl/env/
RUN uv venv -q -p ${PYTHON_VERSION} /opt/venv \
 && uv pip install -q -p /opt/venv/bin/python torch==${TORCH_VERSION} --index-url ${TORCH_INDEX} -c /opt/fl/env/constraints-torch.txt \
 && uv pip install -q -p /opt/venv/bin/python -r /opt/fl/env/requirements.txt -c /opt/fl/env/constraints.txt \
 && /opt/venv/bin/python -c "import torch, zarr, cv2, numpy; print('torch', torch.__version__, 'zarr', zarr.__version__, 'cv2', cv2.__version__, 'numpy', numpy.__version__)"

# villa ink-detection sources (Reader v2 / TRACE pre- and post-processing) at a pinned commit
RUN git init -q /opt/villa && cd /opt/villa && git remote add origin https://github.com/ScrollPrize/villa.git \
 && git sparse-checkout init --cone && git sparse-checkout set vesuvius/src \
 && git fetch -q --depth 1 --filter=blob:none origin ${VILLA_SHA} && git checkout -q FETCH_HEAD && rm -rf .git

# VC3D (vc_render_tifxyz): the AppImage in the context if there is one, else VC3D_URL; sha256-checked either way.
# Build a4979b9 is no longer in villa's GitHub releases, so VC3D_URL points at our copy.
COPY vendor/ /tmp/vendor/
RUN set -e; f=/tmp/vendor/VC3D.AppImage; [ -s $f ] || curl -fsSL --retry 3 -o $f "${VC3D_URL}"; \
    echo "${VC3D_SHA256}  $f" | sha256sum -c --quiet; chmod +x $f; cd /opt && $f --appimage-extract >/dev/null && mv squashfs-root vc3d && chmod -R a+rX /opt/vc3d && rm -rf /tmp/vendor; \
    /opt/vc3d/AppRun vc_render_tifxyz --help 2>&1 | head -1

# public model checkpoints, pinned by revision and sha256 (MODELS_DIR=/models at run time replaces them, same checks)
COPY checkpoints.tsv /opt/fl/env/
RUN set -e; while read -r rel url sha; do case "$rel" in ''|\#*) continue;; esac; \
      mkdir -p /opt/fl/$(dirname $rel); curl -fsSL --retry 3 -o /opt/fl/$rel "$url"; echo "$sha  /opt/fl/$rel" | sha256sum -c --quiet; \
    done < /opt/fl/env/checkpoints.tsv; ls -la /opt/fl/models/*/*

# PNG encoder library. The opencv wheel links the system libz.so.1, and the submitted PNGs were written on a machine whose
# libz is zlib-ng-compat 2.3.3 (Fedora 44 RPM zlib-ng-compat-2.3.3-5, needs glibc >= 2.14 only); Ubuntu's zlib 1.3 compresses
# the same pixels to different bytes. run.sh puts this copy on LD_LIBRARY_PATH only for drawing the images (PNG_ZLIB=system
# turns that off), so the PNG bytes, not only the pixels, match the submitted files.
ARG ZLIBNG_SHA256=cc2d0ee053bda906e3d9de9118fe7eb62d4c193ee1df3c323e60dae88e60780e
COPY zlib-ng/ /opt/zlib-ng/
RUN echo "${ZLIBNG_SHA256}  /opt/zlib-ng/libz.so.1" | sha256sum -c --quiet && echo "zlib-ng-compat 2.3.3 OK"

# v8-in: our ink8um loader (modeling.py, inference.py, config.json) and the released base weights (Hugging Face
# YoussefMoNader/ink-8um-v8in, model.safetensors), from the context and sha256-checked.
# v8Q, used only in reading-aid views: v8-in fine-tuned on PHerc1447 w058 + PHerc0343 tracings (no PHerc0306B data)
ARG V8_SHA256=3b94548d7f9ba9f99bc4b0ad8cafcec6d42585c06304e1bd11e47b5d0f88c4cd
ARG V8Q_SHA256=4cb8e9217e5fba795cc0d7fc53afd998ad82c6c5d1250cdb9e63008f36a8f305
COPY v8in/ /opt/fl/third_party/v8in/
RUN echo "${V8_SHA256}  /opt/fl/third_party/v8in/model.safetensors" | sha256sum -c --quiet && echo "v8-in weights OK" \
 && echo "${V8Q_SHA256}  /opt/fl/third_party/v8in/v8Q_final.safetensors" | sha256sum -c --quiet && echo "v8Q weights OK"

# Reader V3 (Hugging Face domenicor046/reader-v3 at revision 39f626e5, MIT), also in the context: Dockerfile.from_v1 builds on our
# first version's image, whose checkpoint layer predates V3. This build already has it from checkpoints.tsv (same file, same check).
ARG READER_V3_SHA256=ea405417450abd8ace1b8ff79ebce3adb646c5e48ab95da46844d77efe984d8a
COPY reader-v3/reader-v3.pth /opt/fl/models/community/reader-v3/reader-v3.pth
COPY checkpoints.tsv /opt/fl/env/checkpoints.tsv
RUN echo "${READER_V3_SHA256}  /opt/fl/models/community/reader-v3/reader-v3.pth" | sha256sum -c --quiet && echo "Reader V3 weights OK"

# A-hn, the three seeds of our v8-in fine-tune, sha256-checked. finalist.json names them; mode 'smoke' re-runs them on data/smoke/crop.npz
COPY finalist/ /opt/fl/third_party/v8in/finalist/
RUN sha256sum -c --quiet /opt/fl/third_party/v8in/finalist/SHA256SUMS && echo "version-7 model weights OK"
# mode 'train': the training code at the paths it ran from, its registries, the exclusion zone and the pool meshes; the CT renders
# and labels are mounted at run time (train/INPUTS.tsv); links.tsv points the code at weights already here
COPY train/ /opt/fl/train/
RUN mkdir -p /home/liam/projects/vesuvius-work && cp -r /opt/fl/train/W/. /home/liam/projects/vesuvius-work/ \
 && TAB="$(printf '\t')" && while IFS="$TAB" read -r p t; do case "$p" in ''|\#*) continue;; esac; \
      mkdir -p "$(dirname /home/liam/projects/vesuvius-work/$p)"; ln -sfn "$t" "/home/liam/projects/vesuvius-work/$p"; done < /opt/fl/train/links.tsv \
 && chmod -R a+rwX /home/liam/projects && echo "training tree OK"

# the inputs of the images: the shipped per-model maps (data/maps/fwd_*.npy on the render raster, 4900 x 7780 float16), the A-hn
# members (data/members/), the CT surface layer (ct20.png), and every view with its sha256 (data/final/)
COPY data/ /opt/fl/data/
# code (claim image, ink steps, compare, entrypoint, ink libraries) and the claim mesh
COPY code/ /opt/fl/
COPY mesh/ /opt/fl/mesh/
RUN cd /opt/fl && sha256sum -c --quiet data/inputs.sha256 && echo "inputs sha256 OK"
WORKDIR /out
ENTRYPOINT ["bash", "/opt/fl/claim/run.sh"]
CMD ["final"]
