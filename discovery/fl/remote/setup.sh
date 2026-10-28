#!/bin/bash
# Build a First Letters screening worker on a fresh Vast box (pytorch base image, run as root).
set -euo pipefail
W=/home/liam/projects/vesuvius-work; V=/home/liam/projects/villa
SHA=a329895ea5e9bde78752e60dbc0a4878dd2ab909
APPIMG=https://github.com/ScrollPrize/villa/releases/download/latest/VC3D-4eecd08-2026-10-12-linux-x86_64.AppImage   # tested 2026-10-13; the 'latest' tag moves
export DEBIAN_FRONTEND=noninteractive
(apt-get update -qq && apt-get install -y -qq curl git rsync file libgl1 libglib2.0-0 procps libfreetype6 libfontconfig1 libxkbcommon0 libdbus-1-3 libegl1 libopengl0 libx11-6 libxcb1 build-essential >/dev/null) || true
mkdir -p $W/envs $W/models/ink9_ens $W/vc_build_main/bin $W/fl/vc3dcfg $W/fl/vol $W/fl/tmp $W/fl/run $W/logs
command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null)
export PATH=$HOME/.local/bin:$PATH
[ -x $W/envs/vcpy/bin/python ] || uv venv -q -p 3.14 $W/envs/vcpy
uv pip install -q -p $W/envs/vcpy/bin/python torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130 || uv pip install -q -p $W/envs/vcpy/bin/python torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install -q -p $W/envs/vcpy/bin/python zarr==3.4.1 numpy tifffile timm einops fsspec aiohttp huggingface_hub numcodecs opencv-python-headless scipy safetensors awscli requests pynrrd imagecodecs
if [ ! -d $V/vesuvius/src ]; then
  git init -q $V && cd $V && git remote add origin https://github.com/ScrollPrize/villa.git
  git sparse-checkout init --cone && git sparse-checkout set vesuvius/src
  git fetch -q --depth 1 --filter=blob:none origin $SHA && git checkout -q FETCH_HEAD
fi
if [ ! -x /opt/squashfs-root/AppRun ]; then
  # the pinned asset disappears whenever villa re-cuts its 'latest' release (2026-10-12: fb8c2c4 -> 404, every lane rendered 0 windings):
  # fall back to whatever x86_64 AppImage the 'latest' release carries now
  cd /opt && { curl -fsSL -o VC3D.AppImage $APPIMG || curl -fsSL -o VC3D.AppImage "$(curl -fsSL https://api.github.com/repos/ScrollPrize/villa/releases/tags/latest \
      | python3 -c "import json,sys;print(next(a['browser_download_url'] for a in json.load(sys.stdin)['assets'] if a['name'].endswith('linux-x86_64.AppImage')))")"; } \
    && chmod +x VC3D.AppImage && ./VC3D.AppImage --appimage-extract >/dev/null
fi
[ -x /opt/squashfs-root/AppRun ] || { echo "SETUP FAILED: no VC3D AppImage (renderer)"; exit 1; }   # fail here, not after a 4 h fit
printf '#!/bin/bash\nexport OPENBLAS_NUM_THREADS=1\nexec /opt/squashfs-root/AppRun vc_grow_seg_from_seed "$@"\n' > $W/vc_build_main/bin/vc_grow_seg_from_seed
cp $W/fl/remote/vc_render_wrapper.sh $W/vc_build_main/bin/vc_render_tifxyz   # folds --remote-url into --volume (VC3D >= 2026-10-12)
chmod +x $W/vc_build_main/bin/vc_grow_seg_from_seed $W/vc_build_main/bin/vc_render_tifxyz
printf '[viewer]\nremote_cache_dir=%s/fl/vcache\n' $W > $W/fl/vc3dcfg/VC3D.ini
cd $W/models/ink9_ens
for u in hybrid_3d2d-seed42/step-075000.pth:hybrid_3d2d-seed42_step-075000.pth hybrid_3d2d-seed43/step-075000.pth:hybrid_3d2d-seed43_step-075000.pth; do
  [ -s ${u#*:} ] || curl -fsSL -o ${u#*:} https://huggingface.co/scrollprize/ink_9um/resolve/main/${u%%:*}
done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$W/envs/vcpy/bin/python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
/opt/squashfs-root/AppRun vc_render_tifxyz --help | head -2
echo SETUP_OK
