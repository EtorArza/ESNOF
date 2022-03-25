from argparse import ArgumentError
from multiprocessing.sharedctypes import copy

from utils.UpdateParameter import *
import subprocess
import time
import re
from os.path import exists
import sys

savefig_paths = ["results/figures", "/home/paran/Dropbox/BCAM/07_estancia_1/paper/images"]
parameter_file="experiments/nipes/parameters.csv"

maxEvalTimes = [1.0, 3.0, 5.0, 10.0, 20.0, 30.0]
seeds = list(range(2,5))




n_tasks = 3
task_list = ["ExploreObstacles", "ExploreObstaclesDistanceBonus", "ExploreHardRace"]
scene_list = ["shapes_exploration.ttt", "shapes_exploration_bounus_4_distance.ttt", "hard_race.ttt"]

for index, task, scene in zip(range(n_tasks), task_list, scene_list):

    if index != 0:
        continue

    if len(sys.argv) != 2:
        raise ArgumentError("this script requires only one argument --plot --launch_local or --launch_cluster")

    if sys.argv[1] not in ("--plot", "--launch_local", "--launch_cluster"):
        raise ArgumentError("this script requires only one argument --plot --launch_local or --launch_cluster")


    # update parameters
    if sys.argv[1] in ("--launch_local", "--launch_cluster"):
        parameter_file = "experiments/nipes/parameters.csv"
        parameter_text = f"""
#experimentName,string,nipes
#subexperimentName,string,standard
#preTextInResultFile,string,seed_8_maxEvalTime_0.5
#resultFile,string,../results/data/runtimewrtmaxevaltime_results/runtimewrtmaxevaltime_exp_result_8_maxEvalTime_0.5.txt


#expPluginName,string,/usr/local/lib/libNIPES.so
#scenePath,string,/home/paran/Dropbox/BCAM/07_estancia_1/code/evolutionary_robotics_framework/simulation/models/scenes/{scene}
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
#seed,int,8

#populationSize,int,8
#maxEvalTime,float,30.0
#maxNbrEval,int,80
#timeStep,float,0.1

#modifyMaxEvalTime,bool,0
#constantmodifyMaxEvalTime,float,0.0
#minEvalTime,float,3.0

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
#lengthOfStagnation,int,200
#kValue,int,15
#noveltyThreshold,double,0.9
#archiveAddingProb,double,0.4
#noveltyRatio,double,1.
#noveltyDecrement,double,0.05
#populationStagnationThreshold,float,0.001

#nbrWaypoints,int,50
#withBeacon,bool,1
#flatFloor,bool,1
#use_sim_sensor_data,bool,0
#withTiles,bool,1     
#jointSubs,sequence_int,-1;-1;-1;0;1;2
"""

        mass_update_parameters(parameter_file, parameter_text)


    #region local_cluster

    if sys.argv[1] == "--launch_cluster":
        import itertools
        import time

        def run_with_seed_and_runtime(maxEvalTime, seed, port):

            time.sleep(0.5)
            update_parameter(parameter_file, "seed", str(seed))
            update_parameter(parameter_file, "maxEvalTime", str(maxEvalTime))
            update_parameter(parameter_file, "resultFile", f"../results/data/runtimewrtmaxevaltime_results/{task}_runtimewrtmaxevaltime_exp_result_{seed}_maxEvalTime_{maxEvalTime}.txt")
            update_parameter(parameter_file, "preTextInResultFile", f"seed_{seed}_maxEvalTime_{maxEvalTime}")

            subprocess.run(f"bash launch.sh -e=nipes --vrep --cluster --parallel --port={port}",shell=True)

            
        port = int(26100000)
        for maxEvalTime, seed in itertools.product(maxEvalTimes, seeds):
            time.sleep(0.25)
            run_with_seed_and_runtime(maxEvalTime, seed, port)
            port += int(10e4)


    #endregion
        





    #region local_launch

    if sys.argv[1] == "--launch_local":
        import itertools
        import time

        def run_with_seed_and_runtime(maxEvalTime, seed):

            time.sleep(0.5)
            update_parameter(parameter_file, "seed", str(seed))
            update_parameter(parameter_file, "maxEvalTime", str(maxEvalTime))
            update_parameter(parameter_file, "resultFile", f"../results/data/runtimewrtmaxevaltime_results/{task}_runtimewrtmaxevaltime_exp_result_{seed}_maxEvalTime_{maxEvalTime}.txt")
            update_parameter(parameter_file, "preTextInResultFile", f"seed_{seed}_maxEvalTime_{maxEvalTime}")

            exec_res=subprocess.run(f"bash launch.sh --coppelia -e=nipes --parallel",shell=True, capture_output=True)
            with open(f"{task}_logs_{seed}.txt", "w") as f:
                f.write("OUT: ------------------")
                f.write(exec_res.stdout.decode("utf-8"))
                f.write("ERR: ------------------")
                f.write(exec_res.stderr.decode("utf-8"))
            
        for maxEvalTime, seed in itertools.product(maxEvalTimes, seeds):
            run_with_seed_and_runtime(maxEvalTime, seed)


    #endregion




    #region plot

    if sys.argv[1] == "--plot":

        from matplotlib import pyplot as plt
        import numpy as np
        from cycler import cycler
        from statistics import mean,median,mode
        from pylab import polyfit
        import subprocess
        import sys


        savefig_paths = ["results/figures", "/home/paran/Dropbox/BCAM/07_estancia_1/paper/images"]

        def load_bw_theme(ax: plt.Axes):
            # taken from http://olsgaard.dk/monochrome-black-white-plots-in-matplotlib.html
            # Create cycler object. Use any styling from above you please
            monochrome = (cycler('color', ['k']) * cycler('marker', [' ', '.', 'x', '^']) * cycler('linestyle', ['-', '--', ':', '-.']))
            ax.set_prop_cycle(monochrome)
            #ax.grid()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)



        fig, ax = plt.subplots(1,1)
        load_bw_theme(ax)

        # ax.boxplot(max_eval_time_5,positions=[50],widths=[40])
        # ax.boxplot(max_eval_time_15,positions=[150],widths=[40])
        # ax.boxplot(max_eval_time_30,positions=[300],widths=[40])
        # ax.boxplot(max_eval_time_60,positions=[600],widths=[40])
        # ax.boxplot(max_eval_time_90,positions=[900],widths=[40])
        # ax.boxplot(max_eval_time_120,positions=[1200],widths=[40])


        def get_most_common_n_lines():
            res = []
            for maxEvalTime in maxEvalTimes:
                for seed in seeds:
                    res_filepath = f"results/data/runtimewrtmaxevaltime_results/{task}_runtimewrtmaxevaltime_exp_result_{seed}_maxEvalTime_{maxEvalTime}.txt"
                    if exists(res_filepath):
                        with open(res_filepath, "r") as f:
                            all_text = f.readlines()
                            res.append(len(all_text))
            return mode(res)




        average_runtimes = []
        x_lower = min(maxEvalTimes)
        x_upper = max(maxEvalTimes)

        mode_nlines = get_most_common_n_lines()

        for maxEvalTime in maxEvalTimes:
            runtimes = []
            for seed in seeds:
                res_filepath = f"results/data/runtimewrtmaxevaltime_results/{task}_runtimewrtmaxevaltime_exp_result_{seed}_maxEvalTime_{maxEvalTime}.txt"
                if exists(res_filepath):
                    with open(res_filepath, "r") as f:
                        all_text = f.readlines()
                        if len(all_text) != mode_nlines:
                            print(f"Skipping file {res_filepath} of only {len(all_text)} lines, when it should have {mode_nlines} lines.")
                            continue
                        split_line = all_text[-1].strip("\n").split(",")                    
                        if len(split_line) != 6:
                            print("Skipping line of length",len(split_line))
                            continue
                        fitness = float(split_line[1])
                        clock_time = float(split_line[2])
                        rw_time = float(split_line[3])
                        maxEvalTime = float(split_line[4])
                        evals = int(split_line[5])
                        runtimes.append(clock_time)
            average_runtimes.append(mean(runtimes))
            ax.boxplot(runtimes, positions=[maxEvalTime],widths=[(x_upper - x_lower) / len(maxEvalTimes) / 2])

        x = np.array(maxEvalTimes)
        y = np.array(average_runtimes) 

    

        m,b = polyfit(x, y, 1)
        ax.plot([x_lower,x_upper], [x_lower*m+b, x_upper*m+b], label=f"$f(x) = {m:2f} \cdot x  + {b:2f}$", linestyle="--")

        ax.scatter(maxEvalTimes, average_runtimes, marker="+", label="Average")
        # ax.scatter(x[0],median(maxEvalTimes[0]),marker="_", color="orange", label="Median")
        ax.legend()

        plt.xlabel("maxEvalTime for each controller")
        plt.ylabel("Runtime of the evolutionary algorithm")
        for path in savefig_paths:
            plt.savefig(path + f"/{task}_runtime_of_one_controller_evaluation_with_respect_to_max_eval_time.pdf")
        plt.close()
    #endregion