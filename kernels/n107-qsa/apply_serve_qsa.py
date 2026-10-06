#!/usr/bin/env python3
"""N107 QSA: add the env-guarded QSA patch to a serve script (n104_serve.sh / n106_serve_prof.sh). Idempotent.

  python3 kernels/xpu_bmg/n107-qsa/apply_serve_qsa.py runs/N104-nvtier/n104_serve.sh [runs/N106-ingest/code-path/n106_serve_prof.sh]

What it does (backup <script>.pre_n107qsa):
  * extracts the image's own exl3xpu/qsa_xpu.py to n107-qsa/qsa_xpu_base.py (falls back to runs/N104-nvtier/src/
    exl3xpu/qsa_xpu.py, which the N106 analysis found byte-identical) - the overlay imports it as exl3xpu.qsa_xpu_base
  * inserts, before the `--entrypoint python3 24c872759256` line, the mounts + env knobs:
      QSATRITON=0|1 (default 0 = unchanged server)  QSAVAR=row|tile|compact|auto  QSATILECTX=<auto threshold>
      QSAFP8=bits|cast  QSACFG=<BLOCK_N,warps[,stages]>  QSATILECFG=<BM,BN,warps[,stages]>
      QSACHECK=<n calls compared vs union>  QSANSCHECK=0|1 (host-bound asserts)  QSADUMP=1 (dump top-k indices)
      QSANOSYNC=1|0  QSAEXPAND=1|0 (unset: patch default, 1 with QSATRITON=1, 0 with QSAIMPL=sycl_row)
      QSACACHE=1 (persistent Triton cache in n107-qsa/results; changes caching of ALL kernels)
      N108: QSAIMPL=sycl_row (SYCL per-query kernel, build/qsa_row_sycl.so)  QSASYCLCFG=<kernel,cpw,sg[,grf[,fp8]]>
      QSASYCLMIN=<min rows>  QSASELECT=sycl (SYCL indexer selection)  QSASELCHECK=<n calls compared with torch>
With QSATRITON=0 and QSAIMPL unset the overlay module behaves exactly like the original qsa_xpu.py.
"""
import os
import shutil
import subprocess
import sys

MARK = "# n107-qsa"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))           # ~/freetoken-exl3
P = "/opt/trellis-serve/xpu/exl3xpu"
IMG = "24c872759256"

LINES = r'''  -v {Q}:/n107qsa:ro -v {Q}/results:/n107out -v {Q}/qsa_xpu_base.py:{P}/qsa_xpu_base.py:ro -v {Q}/qsa_xpu_overlay.py:{P}/qsa_xpu.py:ro
  -e EXL3_QSA_TRITON=${{QSATRITON:-0}} -e EXL3_QSA_TRITON_VARIANT=${{QSAVAR:-row}} -e EXL3_QSA_TILE_MAX_CTX=${{QSATILECTX:-0}} -e EXL3_QSA_TRITON_FP8=${{QSAFP8:-bits}}
  ${{QSACFG:+-e EXL3_QSA_TRITON_CFG=$QSACFG}} ${{QSATILECFG:+-e EXL3_QSA_TILE_CFG=$QSATILECFG}} -e EXL3_QSA_TRITON_CHECK=${{QSACHECK:-0}} -e EXL3_QSA_NOSYNC_CHECK=${{QSANSCHECK:-0}}
  ${{QSAIMPL:+-e EXL3_QSA_IMPL=$QSAIMPL}} ${{QSASYCLCFG:+-e EXL3_QSA_SYCL_CFG=$QSASYCLCFG}} ${{QSASYCLMIN:+-e EXL3_QSA_SYCL_MIN_ROWS=$QSASYCLMIN}} ${{QSASELECT:+-e EXL3_QSA_SELECT=$QSASELECT}} ${{QSASELCHECK:+-e EXL3_QSA_SELECT_CHECK=$QSASELCHECK}}
  ${{QSANOSYNC:+-e EXL3_QSA_NOSYNC=$QSANOSYNC}} ${{QSAEXPAND:+-e EXL3_QSA_TRITON_EXPAND=$QSAEXPAND}} ${{QSADUMP:+-e EXL3_QSA_DUMP_DIR=/n107out/dump}} ${{QSACACHE:+-e TRITON_CACHE_DIR=/n107out/triton-cache-serve}}
'''


def extract_base():
    dst = os.path.join(HERE, "qsa_xpu_base.py")
    try:
        r = subprocess.run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "cat", IMG,
                            f"{P}/qsa_xpu.py"], capture_output=True, timeout=120)
        if r.returncode == 0 and b"def qsa_sparse_attention_union" in r.stdout:
            open(dst, "wb").write(r.stdout)
            print(f"extracted image {IMG}:{P}/qsa_xpu.py -> {dst}")
            return
        print(f"docker extract failed (rc={r.returncode}): {r.stderr[:300]!r}")
    except Exception as e:
        print(f"docker extract failed: {e!r}")
    src = os.path.join(ROOT, "runs", "N104-nvtier", "src", "exl3xpu", "qsa_xpu.py")
    if os.path.exists(src):
        shutil.copyfile(src, dst)
        print(f"copied {src} -> {dst} (N106: byte-identical to the image's copy)")
    elif os.path.exists(dst):
        print(f"kept existing {dst}")
    else:
        sys.exit(f"no source for {dst}: neither the image nor {src} is readable")


def patch(script):
    s = open(script).read()
    if MARK in s:
        print(f"{script}: already patched")
        return
    key = f"--entrypoint python3 {IMG}"
    i = s.find(key)
    if i < 0:
        sys.exit(f"{script}: '{key}' not found")
    line_start = s.rfind("\n", 0, i) + 1
    shutil.copyfile(script, script + ".pre_n107qsa")
    # every inserted line ends with ' \' so the docker run command continues
    lines = [l.rstrip() + " \\\n" for l in LINES.format(Q=HERE, P=P).splitlines()]
    s = s[:line_start] + "".join(lines) + s[line_start:]
    s = s.replace("#!/bin/bash\n", f"#!/bin/bash\n{MARK}: QSATRITON=1 enables kernels/xpu_bmg/n107-qsa (see its README)\n", 1)
    open(script, "w").write(s)
    os.makedirs(os.path.join(HERE, "results", "dump"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "results", "triton-cache-serve"), exist_ok=True)
    print(f"{script}: patched (backup {script}.pre_n107qsa)")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    extract_base()
    for sc in sys.argv[1:]:
        patch(sc)


if __name__ == "__main__":
    main()
