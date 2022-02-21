#!/bin/bash
set -e
BUILD_MODE=Release
CLUSTER=""
for i in "$@"
do
case $i in
    --debug)
    echo "Building in Debug mode..."
    BUILD_MODE=Debug
    ;;
    --cluster)
    echo "Building in Cluster..."
    CLUSTER=true
    LIBRARY_DIR="/share/earza/library"
    DOWNLOAD_DIR="/share/earza/Downloads"
    SUDO_COMMAND=""
    ;;
    --local)
    echo "Building locally..."
    CLUSTER=false
    LIBRARY_DIR="/usr/local"
    DOWNLOAD_DIR="~/Downloads"
    SUDO_COMMAND="sudo"
    ;;
    *)
            # unknown option
    ;;
esac
done


if [ -z "$CLUSTER" ]; then 
    echo "ERROR: Use of either --cluster or --local parameter is required. Exit...";
    exit 1 
fi


cd $DOWNLOAD_DIR
git clone https://github.com/ci-group/MultiNEAT.git
cd MultiNEAT
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$LIBRARY_DIR  ..
make 
$($SUDO_COMMAND make install)

cd $DOWNLOAD_DIR
git clone https://github.com/portaloffreedom/polyvox.git
cd polyvox
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$LIBRARY_DIR  ..
make 
$($SUDO_COMMAND make install)

cd $DOWNLOAD_DIR
git clone https://github.com/m-renaud/libdlibxx.git
cd libdlibxx
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$LIBRARY_DIR  ..
make 
$($SUDO_COMMAND make install) 

cd $DOWNLOAD_DIR
git clone https://github.com/beniz/libcmaes.git
cd libcmaes
mkdir -p build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$LIBRARY_DIR  ..
make
$($SUDO_COMMAND make install)

cd $DOWNLOAD_DIR
git clone https://github.com/oneapi-src/oneTBB.git
cd oneTBB
mkdir -p build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$LIBRARY_DIR -DTBB_TEST=OFF ..
make
$($SUDO_COMMAND make install)


cd $DOWNLOAD_DIR
git clone https://github.com/m-renaud/libdlibxx.git
cd libdlibxx
mkdir -p build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$LIBRARY_DIR  ..
make
$($SUDO_COMMAND make install)


cd $DOWNLOAD_DIR
git clone https://github.com/opencv/opencv.git
cd opencv
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$LIBRARY_DIR  ..
make 
$($SUDO_COMMAND make install)



cd $DOWNLOAD_DIR
git clone https://github.com/LeniLeGoff/nn2.git
cd nn2
git checkout are_v2_1
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$LIBRARY_DIR  ..
make 
$($SUDO_COMMAND make install)


