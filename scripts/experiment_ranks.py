from argparse import ArgumentError
from utils.UpdateParameter import *
import subprocess
import time
import re
from os.path import exists
import sys

seeds=list(range(2,400))

if len(sys.argv) != 2:
    raise ArgumentError("this script requires only one argument --plot --launch_local or --launch_cluster")

if sys.argv[1] not in ("--plot", "--launch_local", "--launch_cluster"):
    raise ArgumentError("this script requires only one argument --plot --launch_local or --launch_cluster")


# update parameters
if sys.argv[1] in ("--launch_local", "--launch_cluster"):
    parameter_file = "experiments/nipes/parameters.csv"
    parameter_text = """
#experimentName,string,nipes
#subexperimentName,string,measure_ranks
#preTextInResultFile,string,seed_202
#resultFile,string,../results/data/ranks_results/ranks_exp_result_202.txt


#expPluginName,string,/usr/local/lib/libNIPES.so
#scenePath,string,/home/paran/Dropbox/BCAM/07_estancia_1/code/evolutionary_robotics_framework/simulation/models/scenes/shapes_exploration.ttt
#robotPath,string,/home/paran/Dropbox/BCAM/07_estancia_1/code/evolutionary_robotics_framework/simulation/models/robots/model0.ttm
#modelsPath,string,/home/paran/Dropbox/BCAM/07_estancia_1/code/evolutionary_robotics_framework/simulation/models

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
#seed,int,202

#populationSize,int,100
#maxEvalTime,float,30.0
#maxNbrEval,int,10000
#timeStep,float,0.1

#modifyMaxEvalTime,bool,1
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
#init_z,float,0.12
#MaxWeight,float,1.0
#energyBudget,double,100
#energyReduction,bool,0
#NNType,int,2
#NbrInputNeurones,int,2
#NbrOutputNeurones,int,4
#NbrHiddenNeurones,int,8
#UseInternalBias,bool,1

#reloadController,bool,1
#CMAESStep,double,1.
#FTarget,double,-0.05
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




#region local_launch

if sys.argv[1] == "--launch_local":
    from joblib import Parallel, delayed

    n_jobs = 5

    def run_with_seed(seed):
        time.sleep(int(1.25*seed) % n_jobs)
        update_parameter(parameter_file, "seed", str(seed))
        update_parameter(parameter_file, "resultFile", f"../results/data/ranks_results/ranks_exp_result_{seed}.txt")
        update_parameter(parameter_file, "preTextInResultFile", f"seed_{seed}")
        exec_res=subprocess.run(f"bash launch.sh --vrep -e=nipes",shell=True, capture_output=True)
        with open(f"logs_{seed}.txt", "w") as f:
            f.write("OUT: ------------------")
            f.write(exec_res.stdout.decode("utf-8"))
            f.write("ERR: ------------------")
            f.write(exec_res.stderr.decode("utf-8"))
        

    Parallel(n_jobs=n_jobs, verbose=12)(delayed(run_with_seed)(i) for i in seeds)



#endregion


#region launch_cluster

if sys.argv[1] == "--launch_cluster":


    def run_with_seed(seed, port):
        update_parameter(parameter_file, "seed", str(seed))
        update_parameter(parameter_file, "resultFile", f"../results/data/ranks_results/ranks_exp_result_{seed}.txt")
        update_parameter(parameter_file, "preTextInResultFile", f"seed_{seed}")
        # Parallel
        subprocess.run(f"bash launch.sh -e=nipes --coppelia --cluster --port={port}",shell=True)
        # # Sequential
        # subprocess.run(f"bash launch.sh -e=nipes --cluster --port={port} --sequential",shell=True)
        print(f"Launched experiment with seed {seed} in port {port}.")

    port = int(10e4)
    for i in seeds:
        time.sleep(1.0)
        run_with_seed(i, port)
        port += int(10e4)



#endregion






#region plot

if sys.argv[1] == "--plot":
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter
    import pandas as pd
    pd.set_option('display.max_rows', 100)

    import numpy as np
    from tqdm import tqdm as tqdm
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

        if res > 0:
            print(res)
        return res

    def extract_rank_from_line(line):
        lines = re.split("\(|\)",line)
        ranks = lines[1]
        res = ranks.split(",")
        res = np.array([float(el) for el in res])
        return res


    def extract_fitness_from_line(line):
        lines = re.split("\(|\)",line)
        ranks = lines[3]
        res = ranks.split(",")
        res = np.array([float(el) for el in res])
        return res

    dataframe_data = []
    for i in seeds:
        if not exists(f"results/data/ranks_results/ranks_exp_result_{i}.txt"):
            continue

        with open(f"results/data/ranks_results/ranks_exp_result_{i}.txt") as f:
            lines = list(map(lambda x: x.strip("\n"), f.readlines()))
            if len(lines) < 7:
                print("Skipping line of len",len(lines))
                continue
            n_evals=-1
            for line in lines:
                if "(" not in line:
                    continue
                seed = int(line.removeprefix("seed_").split("_")[0])
                runtime = float(line.split(",")[1])
                evals = int(line.split(",")[2])
                ranks = extract_rank_from_line(line)
                fitness = extract_fitness_from_line(line)
                dataframe_data.append([seed, runtime, evals, ranks, fitness])

    df = pd.DataFrame(dataframe_data, columns=["seed", "runtime", "evals", "ranks", "fitness"])


    rank_distance_column = []
    is_reference_column = []
    n_with_this_ref = np.empty(df.shape[0], dtype=np.int64)
    n_with_this_ref[:] =  -99999999
    count_n_with_this_ref = 0
    _current_seed = None
    _current_evals = None
    _reference_runtime = None
    ref_rank = None
    for i, row in tqdm(df.iterrows()):
        if row.loc["seed"] != _current_seed or abs(row.loc["evals"] - _current_evals) > 3:
            _current_seed = row.loc["seed"]
            _current_evals = row.loc["evals"]
            _reference_runtime = row.loc["runtime"]
            is_reference_column.append(True)
            ref_rank = row.loc["ranks"][:]
            count_n_with_this_ref = 0
             
        else:
            assert _reference_runtime >= row.loc["runtime"]
            is_reference_column.append(False)
            count_n_with_this_ref += 1

        print(ref_rank - row.loc["ranks"])
        n_with_this_ref[i-count_n_with_this_ref:i+1] = count_n_with_this_ref
        rank_distance_column.append(distance_between_orders(ref_rank, row.loc["ranks"]))



    df['is_reference'] = is_reference_column
    df['distance_to_ref'] = rank_distance_column
    df['n_with_this_ref'] = n_with_this_ref
    # print(df[df["seed"]==20])


    # Discard incomplete files: files with not enough lines (or too many), 
    # files with not enough values per ref. permu (or too many). 

    # Most frequent number of rows with the same ref permu
    most_frequent_n_with_this_ref = int(df.mode(axis='index', numeric_only=True).loc[0,"n_with_this_ref"])

    # Most frequent number of rows with the same seed
    usuall_number_of_rows = int(df["seed"].value_counts().mode()[0])

    # Seeds that have these number of rows.
    seeds_with_usual_number_of_rows = np.array(df["seed"].value_counts()[df["seed"].value_counts() == usuall_number_of_rows].index, dtype=np.int64)

    print(df)
    df = df[df["n_with_this_ref"] == most_frequent_n_with_this_ref]
    df = df[df["seed"].isin(seeds_with_usual_number_of_rows)]
    print(df)
    exit(0)

    exit(0)



    x_list = []
    y_list = []
    for x,y in sorted(distances.items()):
        x_list.append(x)
        y_list.append(np.average(y))


    plt.plot(x_list,y_list)
    plt.show()
#endregion
