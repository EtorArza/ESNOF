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
    ;;
    --local)
    echo "Building in Cluster..."
    CLUSTER=false
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


evolutionary_robotics_framework="`pwd`/evolutionary_robotics_framework"
rsync -r -v --exclude=*.csv "experiments/" "evolutionary_robotics_framework/experiments/"

# BUILD_MODE=Debug
# BUILD_MODE=Release

cd $evolutionary_robotics_framework 
mkdir -p build
cd build



# EXECUTE COPPELIA local
if [[ "$CLUSTER" == true ]]; then
    cmake -DONLY_SIMULATION=1 -DCOPPELIASIM_FOLDER= -DVREP_FOLDER=$evolutionary_robotics_framework/V-REP_PRO_EDU_V3_6_2_Ubuntu18_04 -DCMAKE_INSTALL_PREFIX=/share/earza/library/  -DLIMBO_FOLDER=$evolutionary_robotics_framework/modules/are_limbo -WITH_NN2=1 -DCMAKE_BUILD_TYPE=$BUILD_MODE ..

else
    # # When using V-REP
    # cmake -DONLY_SIMULATION=1 -DCOPPELIASIM_FOLDER= -DVREP_FOLDER=$evolutionary_robotics_framework/V-REP_PRO_EDU_V3_6_2_Ubuntu18_04 -DCMAKE_INSTALL_PREFIX=/usr/local/  -DLIMBO_FOLDER=$evolutionary_robotics_framework/modules/limbo -WITH_NN2=1 -DCMAKE_BUILD_TYPE=$BUILD_MODE ..

    # When using Coppelia
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local/ -DCOPPELIASIM_FOLDER=$evolutionary_robotics_framework/CoppeliaSim_Edu_V4_3_0_Ubuntu18_04 -DLIMBO_FOLDER=$evolutionary_robotics_framework/modules/limbo -WITH_NN2=1 ..
    make
    sudo make install
    cd ../..
fi
