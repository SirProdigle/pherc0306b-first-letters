#!/bin/bash
# erwin_setup.sh : on a box AFTER fl/remote/setup.sh. Old spiral env (villa 0fb38c45 + Erwin's shell-optional patch,
# pushed to x_recipe_erwin/spiral_old), lasagna (villa a329895) for flattening, v8-in + Hecate from Hugging Face.
set -uo pipefail
W=/home/liam/projects/vesuvius-work; V=/home/liam/projects/villa; X=$W/x_recipe_erwin
export PATH=$HOME/.local/bin:$PATH DEBIAN_FRONTEND=noninteractive
(apt-get install -y -qq build-essential cmake ninja-build >/dev/null) || true
cd $V && git sparse-checkout add lasagna && git checkout -q || exit 1
uv pip install -q -p $W/envs/vcpy/bin/python pillow scikit-image tensorstore imagecodecs requests pynrrd numcodecs
[ -x $W/envs/spiral_old/bin/python ] || uv venv -q -p 3.14 $W/envs/spiral_old
uv pip install -q -p $W/envs/spiral_old/bin/python 'torch>=2.11,<2.13' --index-url https://download.pytorch.org/whl/cu128 || exit 1
cd $X/spiral_old && uv pip install -q -p $W/envs/spiral_old/bin/python -e . || exit 1
cd $X/spiral_old && $W/envs/spiral_old/bin/python -c "import torch,fit_spiral;print('spiral_old ok torch',torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0))" || exit 1
cd $V/lasagna && PYTHONPATH=$V/vesuvius/src:$V/lasagna $W/envs/vcpy/bin/python fit.py --help >/dev/null && echo "lasagna ok"
mkdir -p $W/third_party/v8in $W/models/ink9_ens
$W/envs/vcpy/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('YoussefMoNader/ink-8um-v8in', local_dir='$W/third_party/v8in')" > /dev/null 2>&1 && echo "v8in ok"
cd $W/models/ink9_ens
[ -s hecate_9.6um.pth ] || curl -fsSL -o hecate_9.6um.pth https://huggingface.co/scrollprize/hecate/resolve/main/hecate_9.6um.pth
[ -s hecate.py ] || curl -fsSL -o hecate.py https://huggingface.co/scrollprize/hecate/resolve/main/hecate.py
ls -la $W/models/ink9_ens $W/third_party/v8in
echo ERWIN_SETUP_OK
