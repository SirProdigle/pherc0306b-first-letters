#!/bin/bash
# vc_render_tifxyz wrapper for the VC3D AppImage (2026-10-13): VC3D builds from 2026-10-12 on reject --volume <local dir> together with
# --remote-url <url> ("--remote-url [DEPRECATED] Same as passing the remote OME-Zarr URL as --volume").  lane_box.sh / older callers
# pass both, so fold --remote-url into --volume here; all other arguments pass through unchanged.
export OPENBLAS_NUM_THREADS=1
args=(); url=""
while [ $# -gt 0 ]; do
  case "$1" in
    --remote-url) url="$2"; shift 2 ;;
    *) args+=("$1"); shift ;;
  esac
done
if [ -n "$url" ]; then
  out=(); i=0; n=${#args[@]}; seen=0
  while [ $i -lt $n ]; do
    a="${args[$i]}"
    if [ "$a" = "--volume" ] || [ "$a" = "-v" ]; then out+=("--volume" "$url"); i=$((i + 2)); seen=1
    else out+=("$a"); i=$((i + 1)); fi
  done
  [ $seen = 1 ] || out+=("--volume" "$url")
  args=("${out[@]}")
fi
exec /opt/squashfs-root/AppRun vc_render_tifxyz "${args[@]}"
