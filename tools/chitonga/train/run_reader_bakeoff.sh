#!/bin/bash
# ON THE POD (24 Oct 2026, the Jesuit): the reader bake-off, step A. Started by launch_reader_bakeoff.sh with nohup;
# log /root/run.log; markers: MACHINE CAP armed|NOT ARMED, SETUP DONE, SMOKE OK <tag>, TRAIN / TRAINING DONE per arm,
# F20_DONE / F20_FAILED <why>.
# Backstop first (lessons of stage 18 and the translator pod): if runpodctl and RUNPOD_POD_ID exist, the pod removes
# itself at CAP_MIN; otherwise a KEY-FREE backstop (no key of ours on the pod) ends the container at CAP_MIN
# (kill -TERM 1, then poweroff), behind the Beelink's own hard cap and the overwatch.
# llama.cpp: the release asset of the SAME build the Beelink serves with (b11574, commit 8bb3cf0d), CUDA 12.8 x64,
# sha256 pinned in models.json; no compile.
set -u
cd /root
fail() { echo "F20_FAILED $1"; exit 1; }
set -a; [ -f /root/.env ] && . /root/.env; set +a
CAP_MIN=${CAP_MIN:-340}; SOFT_MIN=${SOFT_MIN:-240}
if command -v runpodctl > /dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
  ( sleep $(( CAP_MIN * 60 )); echo "MACHINE CAP $CAP_MIN min reached $(date)"; runpodctl remove pod "$RUNPOD_POD_ID" ) > machine_cap.log 2>&1 &
  echo "MACHINE CAP armed: $CAP_MIN min, pod $RUNPOD_POD_ID (self-remove)"
else
  ( sleep $(( CAP_MIN * 60 )); echo "KEY-FREE BACKSTOP $CAP_MIN min $(date)"; kill -TERM 1; sleep 20; poweroff -f ) > backstop.log 2>&1 &
  echo "MACHINE CAP NOT ARMED (runpodctl=$(command -v runpodctl || echo none) RUNPOD_POD_ID=${RUNPOD_POD_ID:-unset}): key-free backstop armed at $CAP_MIN min"
fi
mkdir -p /root/pack /root/out /root/m /root/llama
tar -xzf /root/rb_pack.tgz -C /root || fail "untar pack"
cd /root/pack && sha256sum -c /root/pack_sha.txt > /root/pack_check.txt 2>&1 || { cat /root/pack_check.txt; fail "pack sha mismatch"; }
cd /root
echo "PACK OK $(wc -l < /root/pack/bodies.jsonl) bodies"
ASSET=$(python3 -c "import json; print(json.load(open('/root/pack/models.json'))['llama_cpp']['pod_asset'])")
ASHA=$(python3 -c "import json; print(json.load(open('/root/pack/models.json'))['llama_cpp']['pod_asset_sha256'])")
curl -sL --retry 3 -o /root/llama.tgz "https://github.com/ggml-org/llama.cpp/releases/download/b11574/$ASSET" || fail "llama.cpp download"
echo "$ASHA  /root/llama.tgz" | sha256sum -c - > /dev/null || fail "llama.cpp asset sha mismatch"
tar -xzf /root/llama.tgz -C /root/llama || fail "untar llama.cpp"
LLAMA=$(find /root/llama -name llama-server -type f | head -1)
[ -n "$LLAMA" ] || fail "no llama-server in the asset"
chmod +x "$LLAMA"; export LD_LIBRARY_PATH="$(dirname "$LLAMA"):/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
# the asset carries llama.cpp's own libraries but not the CUDA runtime / cuBLAS (checked 24 Oct: 62 files, no cudart);
# the CUDA 12.8 image normally has them; if not, llama.cpp's matching cudart asset (sha pinned here) is unpacked beside it
if ldd "$LLAMA" 2>&1 | grep -q "libcudart.*not found\|libcublas.*not found"; then
  echo "CUDA runtime missing: fetching the cudart asset"
  curl -sL --retry 3 -o /root/cudart.tgz "https://github.com/ggml-org/llama.cpp/releases/download/b11574/cudart-llama-b11574-bin-ubuntu-cuda-12.8-x64.tar.gz" || fail "cudart download"
  echo "6ff88f6e9e03295172eab780268f63aa0f098762e0c925a5c9bc86a32b9ff698  /root/cudart.tgz" | sha256sum -c - > /dev/null || fail "cudart sha mismatch"
  mkdir -p /root/cudart && tar -xzf /root/cudart.tgz -C /root/cudart || fail "untar cudart"
  export LD_LIBRARY_PATH="$(dirname "$(find /root/cudart -name 'libcudart.so*' | head -1)"):$LD_LIBRARY_PATH"
fi
# 24 Oct, first launch: the release binaries need GLIBC 2.38 (built on Ubuntu 24.04); the RunPod image is Ubuntu 22.04
# (GLIBC 2.35). Then llama-server is BUILT here from the same tag, commit asserted, CUDA for this card only (the devel
# image has nvcc 12.8): the same code as the Beelink's b11574, compiled for this GPU.
if ! "$LLAMA" --version > /dev/null 2>&1; then
  echo "release binary does not run here ($(ldd "$LLAMA" 2>&1 | grep -o "GLIBC_[0-9.]* not found" | sort -u | tr '\n' ' ')): building b11574 from source"
  command -v cmake > /dev/null || python3 -m pip install -q cmake ninja > pip_cmake.log 2>&1 || fail "cmake install"
  git clone -q --depth 1 --branch b11574 https://github.com/ggml-org/llama.cpp /root/llsrc || fail "git clone b11574"
  [ "$(git -C /root/llsrc rev-parse HEAD)" = "8bb3cf0d216a5ae48360179f2931daf284be4d01" ] || fail "b11574 is not commit 8bb3cf0d"
  CC_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
  NVCC=$(command -v nvcc || ls /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc 2>/dev/null | head -1)
  [ -n "$NVCC" ] && [ -x "$NVCC" ] || fail "no nvcc in the image (try 2, 24 Oct: nvcc was not on PATH)"
  export PATH="$(dirname "$NVCC"):$PATH" CUDACXX="$NVCC"
  echo "BUILD start $(date +%H:%M:%S) CUDA arch $CC_ARCH, $(nproc) cores, nvcc $("$NVCC" --version | grep -o 'release [0-9.]*')"
  cmake -S /root/llsrc -B /root/llsrc/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CC_ARCH" -DLLAMA_CURL=OFF \
        -DCMAKE_CUDA_COMPILER="$NVCC" \
        -DCMAKE_BUILD_TYPE=Release > /root/build.log 2>&1 || { tail -5 /root/build.log; fail "cmake configure"; }
  cmake --build /root/llsrc/build --target llama-server -j "$(nproc)" >> /root/build.log 2>&1 || { tail -8 /root/build.log; fail "build llama-server"; }
  LLAMA=/root/llsrc/build/bin/llama-server
  export LD_LIBRARY_PATH="/root/llsrc/build/bin:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
  echo "BUILD done $(date +%H:%M:%S)"
fi
"$LLAMA" --version > /dev/null 2>&1 || { ldd "$LLAMA" 2>&1 | grep "not found"; fail "llama-server does not run"; }
"$LLAMA" --version 2>&1 | tail -2
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3 -m pip install -q "huggingface_hub>=0.30" hf_transfer psutil > pip.log 2>&1 || fail "pip: $(tail -2 pip.log)"
export HF_HUB_ENABLE_HF_TRANSFER=1
echo "SETUP DONE $(date +%H:%M:%S) llama $LLAMA"
python3 /root/pack/pod_reader_bakeoff.py --pack /root/pack --out /root/out --mdir /root/m --llama "$LLAMA" --soft-min "$SOFT_MIN" ${ONLY:+--only "$ONLY"} \
  || { grep -q "^F20_FAILED" /root/run.log || echo "F20_FAILED pod_reader_bakeoff.py exited non-zero"; exit 1; }
