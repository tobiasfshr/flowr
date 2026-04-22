#!/usr/bin/env bash
set -eo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create --name flowr -y python=3.10
conda activate flowr
conda install -y -c "nvidia/label/cuda-12.2.2" cuda-toolkit
conda install -y -c conda-forge \
  "gcc_linux-64=12" \
  "gxx_linux-64=12" \
  cmake \
  ninja \
  boost \
  ccache \
  "eigen=3.4.0" \
  flann \
  freeimage \
  lz4-c \
  openimageio \
  curl \
  metis \
  glog \
  gtest \
  ceres-solver \
  suitesparse \
  qt \
  glew \
  sqlite \
  cgal-cpp \
  mesa-libgl-devel-cos7-x86_64

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
cat >"$CONDA_PREFIX/etc/conda/activate.d/flowr-toolchain.sh" <<EOF
export _FLOWR_OLD_CC="\${CC-}"
export _FLOWR_OLD_CXX="\${CXX-}"
export _FLOWR_OLD_CUDAHOSTCXX="\${CUDAHOSTCXX-}"
export _FLOWR_OLD_CMAKE_PREFIX_PATH="\${CMAKE_PREFIX_PATH-}"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CMAKE_PREFIX_PATH="${CONDA_PREFIX}:\${CMAKE_PREFIX_PATH-}"
EOF
cat >"$CONDA_PREFIX/etc/conda/deactivate.d/flowr-toolchain.sh" <<'EOF'
export CC="${_FLOWR_OLD_CC-}"
export CXX="${_FLOWR_OLD_CXX-}"
export CUDAHOSTCXX="${_FLOWR_OLD_CUDAHOSTCXX-}"
export CMAKE_PREFIX_PATH="${_FLOWR_OLD_CMAKE_PREFIX_PATH-}"
unset _FLOWR_OLD_CC _FLOWR_OLD_CXX _FLOWR_OLD_CUDAHOSTCXX _FLOWR_OLD_CMAKE_PREFIX_PATH
EOF
source "$CONDA_PREFIX/etc/conda/activate.d/flowr-toolchain.sh"

python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --extra-index-url https://download.pytorch.org/whl/cu122
python -m pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt251/download.html
python -m pip install "pycolmap==3.11.1"

git submodule update --init --recursive
cd extern/colmap
cmake -S . -B build -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DCMAKE_CUDA_ARCHITECTURES="all-major" \
  -DCUDA_ENABLED=OFF \
  -DCGAL_ENABLED=OFF \
  -DGUI_ENABLED=OFF \
  -DOPENGL_ENABLED=OFF
cmake --build build --target install
cd ../..
python -m pip install -e .
