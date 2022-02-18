from utils.UpdateParameter import *
import matplotlib.pyplot as plt
import subprocess
import numpy as np
from tqdm import tqdm as tqdm
from matplotlib.ticker import StrMethodFormatter
from joblib import Parallel, delayed
import time
import re
from os.path import exists

parameter_file = "experiments/nipes/parameters.csv"

parameter_text = """
#experimentName,string,nipes
#subexperimentName,string,measure_ranks
#preTextInResultFile,string,ftest5
#resultFile,string,result5.txt


#expPluginName,string,/usr/local/lib/libNIPES.so
#scenePath,string,/home/paran/Dropbox/BCAM/07_estancia_1/code/evolutionary_robotics_framework/experiments/sim/obstacles_2.ttt
#robotPath,string,/home/paran/Dropbox/BCAM/07_estancia_1/code/evolutionary_robotics_framework/experiments/sim/model0.ttm
#modelPath,string,/home/paran/Dropbox/BCAM/07_estancia_1/code/evolutionary_robotics_framework/experiments/sim/

#repository,string,/home/paran/Dropbox/BCAM/07_estancia_1/code/logs
#fitnessFile,string,fitnesses.csv
#evalTimeFile,string,eval_durations.csv
#behavDescFile,string,final_pos.csv
#stopCritFile,string,stop_crit.csv
#noveltyFile,string,novelty.csv
#archiveFile,string,archive.csv
#energyCostFile,string,energyCost.csv
#simTimeFile,string,simTime.csv

#isScreenshotEnable,bool,0
#isVideoRecordingEnable,bool,0

#jointControllerType,int,0
#verbose,bool,1
#instanceType,int,0
#killWhenNotConnected,bool,0
#shouldReopenConnections,bool,0
#seed,int,7

#populationSize,int,80
#maxEvalTime,float,30.0
#maxNbrEval,int,10000
#timeStep,float,0.1

#modifyMaxEvalTime,bool,0
#constantmodifyMaxEvalTime,float,0.0
#minEvalTime,float,2.0

#noiseLevel,double,0.
#maxVelocity,double,10.

#envType,int,1
#arenaSize,double,2.
#target_x,double,0.75
#target_y,double,0.75
#target_z,double,0.05
#init_x,float,0
#init_y,float,0
#init_z,float,0.05
#MaxWeight,float,1.0
#energyBudget,double,100
#energyReduction,bool,0
#NNType,int,2
#NbrInputNeurones,int,8
#NbrOutputNeurones,int,2
#NbrHiddenNeurones,int,8
#UseInternalBias,bool,1

#reloadController,bool,1
#CMAESStep,double,1.
#FTarget,double,0.05
#elitistRestart,bool,0
#withRestart,bool,1
#incrPop,bool,0
#lengthOfStagnation,int,20
#kValue,int,15
#noveltyThreshold,double,0.9
#archiveAddingProb,double,0.4
#noveltyRatio,double,1.
#noveltyDecrement,double,0.05
#populationStagnationThreshold,float,0.01

#nbrWaypoints,int,50
#withBeacon,bool,1
#flatFloor,bool,1
#use_sim_sensor_data,bool,0
#withTiles,bool,1     

"""






mass_update_parameters(parameter_file, parameter_text)




def run_with_conf_index(seed):
    time.sleep(seed*4 % 20)
    update_parameter(parameter_file, "seed", str(seed))
    update_parameter(parameter_file, "resultFile", f"ranks_exp_result_{seed}.txt")
    update_parameter(parameter_file, "preTextInResultFile", f"seed_{seed}")
    exec_res=subprocess.run(f"bash launch.sh -e=nipes",shell=True, capture_output=True)  

    
seeds = list(range(2,101))

# Parallel(n_jobs=5, verbose=12)(delayed(run_with_conf_index)(i) for i in seeds)


def distance_between_orders(order1, order2):

    n = len(order1)
    if n % 2 != 0:
        print("ERROR, n should be even.")
        exit(1)

    top_half_indexes1 = np.argpartition(order1, -n//2)[-n//2:]
    top_half_indexes2 = np.argpartition(order2, -n//2)[-n//2:]


    res = 0
    for index in top_half_indexes2:
        if index not in top_half_indexes1:
            res += 1

    return res

    

    return int(i1 != i2)


distances = dict()
for i in seeds:
    if not exists(f"logs/ranks_exp_result_{i}.txt"):
        continue
    with open(f"logs/ranks_exp_result_{i}.txt") as f:
        lines = list(map(lambda x: x.strip("\n"), f.readlines()))
        if len(lines) != 7:
            print("Skipping line of len",len(lines))
            continue
        lines = lines[:-1]
        lines = [re.split("\(|\)",el) for el in lines]

        lines = [lines[i][0].strip(",").split(",") + lines[i][1:] for i in range(len(lines))]
        for line_idx in range(len(lines)):
            lines[line_idx][1] = float(lines[line_idx][1])
            lines[line_idx][3] = list(map(float, lines[line_idx][3].split(",")))
            lines[line_idx].remove(",")
            lines[line_idx][4] = list(map(float, lines[line_idx][4].split(",")))
        ref_order = lines[0][3]
        for line in (lines[1:]):
            if line[1] not in distances:
                distances[line[1]] = []
            distances[line[1]].append(distance_between_orders(ref_order, line[3]))



x_list = []
y_list = []
for x,y in sorted(distances.items()):
    x_list.append(x)
    y_list.append(np.average(y))


plt.plot(x_list,y_list)
plt.show()