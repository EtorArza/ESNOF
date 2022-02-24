#!/bin/bash

WORKING_DIR=`pwd`

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
    evolutionary_robotics_framework="$WORKING_DIR/../evolutionary_robotics_framework"
    ;;
    --local)
    echo "Building locally..."
    CLUSTER=false
    evolutionary_robotics_framework="$WORKING_DIR/evolutionary_robotics_framework"
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


rsync -r -v --exclude=*.csv "experiments/" "$evolutionary_robotics_framework/experiments/"

# BUILD_MODE=Debug
# BUILD_MODE=Release

cd $evolutionary_robotics_framework 
mkdir -p build
cd build


# Cluster.
if [[ "$CLUSTER" == true ]]; then
    export LD_LIBRARY_PATH=/share/earza/library/lib/
    /share/earza/cmake/bin/cmake -DONLY_SIMULATION=1 -DCMAKE_INSTALL_PREFIX=/share/earza/library/ -DCOPPELIASIM_FOLDER=/share/earza/CoppeliaSim_Edu_V4_3_0_Ubuntu18_04  -DLIMBO_FOLDER=$evolutionary_robotics_framework/modules/are_limbo -DWITH_NN2=1 -DCMAKE_BUILD_TYPE=$BUILD_MODE ..
    make
    make install
# Local laptop.
else
    # # When using V-REP
    # cmake -DONLY_SIMULATION=1 -DCOPPELIASIM_FOLDER= -DVREP_FOLDER=$evolutionary_robotics_framework/V-REP_PRO_EDU_V3_6_2_Ubuntu18_04 -DCMAKE_INSTALL_PREFIX=/usr/local/  -DLIMBO_FOLDER=$evolutionary_robotics_framework/modules/limbo -DWITH_NN2=1 -DCMAKE_BUILD_TYPE=$BUILD_MODE ..

    # When using Coppelia
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local/ -DCOPPELIASIM_FOLDER=CoppeliaSim_Edu_V4_3_0_Ubuntu18_04 -DLIMBO_FOLDER=$evolutionary_robotics_framework/modules/limbo -DWITH_NN2=1 ..
    make -j
    sudo make install
fi

cd $WORKING_DIR
