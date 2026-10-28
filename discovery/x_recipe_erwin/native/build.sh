#!/bin/bash
# build.sh : on the box -> x_recipe_erwin/native_pkg/vc/{__init__.py, track_crossings*.so, track_store*.so, spiral_sampling*.so}
set -euo pipefail
W=/home/liam/projects/vesuvius-work; X=$W/x_recipe_erwin; PY=${SPIRAL_PY:-$W/envs/spiral_old/bin/python}
export PATH=$HOME/.local/bin:$PATH
uv pip install -q -p $PY nanobind
mkdir -p $X/native_pkg/vc && echo '"""old volume-cartographer native spiral modules (standalone build)"""' > $X/native_pkg/vc/__init__.py
cmake -S $X/native -B $X/native/build -G Ninja -DPython_EXECUTABLE=$PY -DPKG_OUT=$X/native_pkg > $X/native/cmake.log 2>&1
cmake --build $X/native/build -j 16 > $X/native/build.log 2>&1 || { tail -30 $X/native/build.log; exit 1; }
cd $X && PYTHONPATH=$X/native_pkg $PY -c "import vc.track_crossings, vc.track_store, vc.spiral_sampling; print('native ok')"
