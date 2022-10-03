from argparse import ArgumentError
from statistics import median

from utils.UpdateParameter import *
import subprocess
import time
import re
from os.path import exists
import sys
from tqdm import tqdm as tqdm

seeds = list(range(2,25))
port = int(10e4)

# The execution stops when no movement only for M-nipes. This is likely because some morphologies do not even  move, while there 
# is almost always a small movement in NIPES if a morphology that can move is trained.
savefig_paths = ["results/figures/are_project", "/home/paran/Dropbox/BCAM/07_estancia_1/paper/images/are_project"]

# time[0] + maxEvalTime * time[1]
sim_time_coefs = [1.43, 0.06, "simulation"]
physical_time_coefs = [4.73, 1.0, "physical"] 

n_tasks = 7
task_list = ["ExploreObstacles", "ExploreObstaclesDistanceBonus", "ExploreHardRace", "MazeMultiMaze", "MazeMiddleWall", "MazeScapeRoom", "MazeEasyRace"]
scene_list = ["shapes_exploration.ttt", "shapes_exploration_bounus_4_distance.ttt", "hard_race.ttt", "MAZE_multi_maze.ttt","MAZE_middle_wall.ttt","MAZE_escape_room-1.ttt","MAZE_easy_race.ttt"]
max_eval_times = [30, 30, 30, 120, 120, 120, 120]
targets_MAZE =      [None,  None,   None,         [0.0,-0.5], [0.0,-0.5], [-0.8,-0.8], [-0.8,-0.8]]
initial_positions = [[0,0], [0,0], [-0.75,-0.85], [-0.3,0.0], [0.0,0.5],  [0.0,0.0],    [0.8,0.8]]


method_list = ["constant", "bestasref"]

for index, task, scene in zip(range(n_tasks), task_list, scene_list):

    print("Working on ", task, scene)

    if len(sys.argv) != 2:
        raise ArgumentError("this script requires only one argument --plot --launch_local or --launch_cluster")

    if sys.argv[1] not in ("--plot", "--launch_local", "--launch_cluster"):
        raise ArgumentError("this script requires only one argument --plot --launch_local or --launch_cluster")


    # update parameters
    if sys.argv[1] in ("--launch_local", "--launch_cluster"):
        parameter_file = "experiments/nipes/parameters.csv"
        parameter_text = f"""
#experimentName,string,nipes
#subexperimentName,string,halving
#preTextInResultFile,string,seed_2
#resultFile,string,../results/data/simulatedARE_results/runtimereduced_result_2.txt


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
#seed,int,2

#populationSize,int,10
#maxEvalTime,float,120.0
#maxNbrEval,int,10000
#timeStep,float,0.1
#maxComputationTime,double,86400.0

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
#initPosition,sequence_double,0;0;0.12
#evaluationOrder,int,1

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
#incrPop,bool,1
#lengthOfStagnation,int,20
#kValue,int,15
#noveltyThreshold,double,0.9
#archiveAddingProb,double,0.4
#noveltyRatio,double,1.
#noveltyDecrement,double,0.05
#populationStagnationThreshold,float,0.00001

#nbrWaypoints,int,50
#withBeacon,bool,1
#flatFloor,bool,1
#use_sim_sensor_data,bool,0
#withTiles,bool,1     
#jointSubs,sequence_int,-1;-1;-1;0;1;2
"""

        mass_update_parameters(parameter_file, parameter_text)




    #region local_launch

    if sys.argv[1] == "--launch_local":
        import itertools
        import time


        def run_with_seed_and_runtime(seed, subexperimentName):

            time.sleep(0.5)

            update_parameter(parameter_file, "init_x", str(initial_positions[index][0]))
            update_parameter(parameter_file, "init_y", str(initial_positions[index][1]))
            update_parameter(parameter_file, "initPosition", ";".join([str(el) for el in initial_positions[index] + [0.12]]))

            # Update parameters to the paper "Sample and time efficient policy learning with CMA-ES and Bayesian Optimisation"
            # envType 0 is get to objective, and envType 1 is exploration.
            if "MAZE" in scene:
                update_parameter(parameter_file, "maxEvalTime", str(120.0))
                update_parameter(parameter_file, "bestasrefGrace", str(24.0))
                update_parameter(parameter_file, "envType", str(0))
                update_parameter(parameter_file, "target_x", str(targets_MAZE[index][0]))
                update_parameter(parameter_file, "target_y", str(targets_MAZE[index][1]))
            else:
                update_parameter(parameter_file, "maxEvalTime", str(30.0))
                update_parameter(parameter_file, "bestasrefGrace", str(6.0))
                update_parameter(parameter_file, "envType", str(1))


            update_parameter(parameter_file, "subexperimentName", subexperimentName)
            update_parameter(parameter_file, "seed", str(seed))
            update_parameter(parameter_file, "resultFile", f"../results/data/simulatedARE/{task}_{subexperimentName}_exp_result_{seed}.txt")
            update_parameter(parameter_file, "preTextInResultFile", f"seed_{seed}")
            print("Launching ARE in experiment_simulatedARE.py ...")
            exec_res=subprocess.run(f"bash launch.sh --coppelia -e=nipes --sequential",shell=True, capture_output=True)
            with open(f"{task}_simulatedARE_logs_{seed}.txt", "w") as f:
                f.write("------------------")
                f.write("OUT: ")
                f.write(exec_res.stdout.decode("utf-8"))
                f.write("ERR: ")
                f.write(exec_res.stderr.decode("utf-8"))
                f.write("------------------")

        for method in method_list:
            for seed in seeds:
                run_with_seed_and_runtime(seed, method)


    #endregion


    #region launch_cluster

    if sys.argv[1] == "--launch_cluster":
        import itertools
        import time


        def run_with_seed_and_runtime(seed, subexperimentName, port):

            time.sleep(0.5)

            update_parameter(parameter_file, "init_x", str(initial_positions[index][0]))
            update_parameter(parameter_file, "init_y", str(initial_positions[index][1]))
            update_parameter(parameter_file, "initPosition", ";".join([str(el) for el in initial_positions[index] + [0.12]]))

            # Update parameters to the paper "Sample and time efficient policy learning with CMA-ES and Bayesian Optimisation"
            # envType 0 is get to objective, and envType 1 is exploration.
            if "MAZE" in scene:
                update_parameter(parameter_file, "maxEvalTime", str(120.0))
                update_parameter(parameter_file, "bestasrefGrace", str(24.0))
                update_parameter(parameter_file, "envType", str(0))
                update_parameter(parameter_file, "target_x", str(targets_MAZE[index][0]))
                update_parameter(parameter_file, "target_y", str(targets_MAZE[index][1]))
            else:
                update_parameter(parameter_file, "maxEvalTime", str(30.0))
                update_parameter(parameter_file, "bestasrefGrace", str(6.0))
                update_parameter(parameter_file, "envType", str(1))


            update_parameter(parameter_file, "subexperimentName", subexperimentName)
            update_parameter(parameter_file, "seed", str(seed))
            update_parameter(parameter_file, "resultFile", f"../results/data/{subexperimentName}_results/{task}_{subexperimentName}_exp_result_{seed}.txt")
            update_parameter(parameter_file, "preTextInResultFile", f"seed_{seed}")
            print("Launching ARE in experiment_simulatedARE.py ...")
            # # Parallel
            # subprocess.run(f"bash launch.sh -e=nipes --vrep --cluster --parallel --port={port} > {task}_{subexperimentName}_logs_{seed}.txt 2>&1",shell=True)

            # Sequential
            subprocess.run(f"bash launch.sh -e=nipes --vrep --cluster --port={port} --sequential",shell=True)

        for method in method_list:
            for seed in seeds:
                time.sleep(0.5)
                run_with_seed_and_runtime(seed, method, port)
                port += int(10e4)

        print("Last port = ", port)



    #endregion






    #region plot

    if sys.argv[1] == "--plot":
        import itertools
        import pandas as pd
        from matplotlib import pyplot as plt
        import numpy as np
        from scipy.stats import mannwhitneyu


        
        df_row_list = []
        for method in method_list:
            for seed in seeds:
                i = 0
                res_filepath = f"results/data/simulatedARE/{task}_{method}_exp_result_{seed}.txt"
                if exists(res_filepath):
                    with open(res_filepath, "r") as f:
                        all_text = f.readlines()
                        for line in all_text:
                            split_line = line.strip("\n").split(",")
                            fitness = float(split_line[1])
                            clock_time = float(split_line[2])
                            rw_time = float(split_line[3])
                            evals = int(split_line[4])
                            maxevaltimes_each_controller = [float(el) for el in split_line[5].strip("()").split(";") if len(el) > 0]
                            if float(fitness) < -10e200:
                                continue
                            df_row_list.append([seed, evals, rw_time, fitness, maxevaltimes_each_controller, clock_time, method, task])
                            i += 1
                print(i, "rows:", res_filepath)
        df_all = pd.DataFrame(df_row_list, columns=["seed", "evals", "rw_time", "fitness", "maxevaltimes_each_controller", "simulated_time", "method", "task"])

        import pandas as pd
        pd.set_option('display.max_rows', 1000)
  
        def get_xy_from_df(time_name_in_df, x_min, x_max, x_nsteps, df: pd.DataFrame):
            x = []
            y_median = []
            y_lower = []
            y_upper = []
            every_y = []

            for runtime in np.linspace(x_min, x_max, num=x_nsteps):
                if df[df[time_name_in_df] <= runtime].shape[0] <= 2:
                    continue
                
                fitnesses = []
                for seed in seeds:
                    f_with_seed_and_runtime_leq = df[(df[time_name_in_df]<=runtime) & (df["seed"]==seed)]["fitness"]
                    f_with_runtime_geq = df[df[time_name_in_df]>runtime]
                    if len(f_with_runtime_geq) < 4:
                        return x, y_median, y_lower, y_upper, every_y
                    if len(f_with_seed_and_runtime_leq) >= 1:
                        fitnesses.append(max(f_with_seed_and_runtime_leq))
                    else:
                        continue

                if len(fitnesses) < len(seeds):
                    continue
                
                x.append(runtime)
                every_y.append(fitnesses)
                y_median.append(np.quantile(np.array(fitnesses), 0.5))
                y_lower.append(np.quantile(np.array(fitnesses), 0.25))
                y_upper.append(np.quantile(np.array(fitnesses), 0.75))
            
            return x, y_median, y_lower, y_upper, every_y



        x_min = 0.0
        x_max = 100.0
        x_nsteps = 1000
        statistical_test_alpha = 0.05

        x_max_suggested = 10e10
        for method in method_list:
            x_max_value_list = []
            sub_f = df_all[df_all["method"] == method]

            for seed in seeds:
                x_max_value_list.append(max(sub_f[sub_f["seed"] == seed]["simulated_time"]))

            x_max_suggested = min(x_max_suggested, np.quantile(x_max_value_list, 0.25))


        print("x_max_suggested:", x_max_suggested)
        

        x_max = x_max_suggested

        x_list = []
        y_median_list = []
        y_lower_list = []
        y_upper_list = []
        every_y_halve_list = []
        for method in method_list:

            x, y_median, y_lower, y_upper, every_y_halve = get_xy_from_df("simulated_time", x_min, x_max, x_nsteps, df_all[df_all["method"] == method])
            x_list.append(x)
            y_median_list.append(y_median)
            y_lower_list.append(y_lower)
            y_upper_list.append(y_upper)
            every_y_halve_list.append(every_y_halve)


        def get_test_result(x, y, alpha):
            x = x[0:min(len(x), len(y))]
            y = y[0:min(len(x), len(y))]
            if len(x) < 5:
                print("WARNING: statistical test with less than 5 samples. Probably wont be significant.")
            return mannwhitneyu(x, y, alternative='two-sided')[1] < alpha


        # # This assertion is required for doing the tests. We are comparing the samples based on the samples
        # # in every_y_halve and every_y_constant. Consequently, the indexes in these samples need to correspond 
        # # to the same x values.

        x_max_of_the_lowest_for_test = max((min(el) for el in x_list))
        index_constant_max_of_lowest = x_list[0].index(x_max_of_the_lowest_for_test)
        index_bestasref_max_of_lowest = x_list[1].index(x_max_of_the_lowest_for_test)

        y_constant_test  = every_y_halve_list[0][index_constant_max_of_lowest:]
        y_bestasref_test = every_y_halve_list[1][index_bestasref_max_of_lowest:]
        x_test = x_list[0][index_constant_max_of_lowest:]

        print(len(y_constant_test) , len(y_bestasref_test) , len(x_test))
        assert len(y_constant_test) == len(y_bestasref_test) == len(x_test)


        test_results_true = np.where([get_test_result(y_constant_test[i], y_bestasref_test[i], statistical_test_alpha) for i in range(len(y_constant_test))])[0]



        # import code
        # code.interact(local=locals())



        plt.figure()
        plt.xlim((0, x_max))
        for x, y_median, y_lower, y_upper, every_y_halve, method, color in zip(x_list, y_median_list, y_lower_list, y_upper_list, every_y_halve_list, method_list, ["red", "green", "blue"]):
            plt.plot(x, y_median, marker="", label=f"{method}", color=color)
            plt.fill_between(x, y_lower, y_upper, color=color, alpha=.1)
        y_min = plt.gca().get_ylim()[0]
        plt.plot(np.array(x_test)[test_results_true], np.repeat(y_min, len(test_results_true)), linestyle="None", marker = "_", color="black", label=f"$p < {statistical_test_alpha}$")
            # plt.scatter(df_halve_maxevaltime["rw_time"], df_halve_maxevaltime["fitness"], marker="o", label = "halve runtime", alpha=0.5, color="red")
        plt.legend()
        for path in savefig_paths:
            plt.savefig(path + f"/{task}_exp_line.pdf")
        plt.close()

        # pd.set_option('display.max_rows', None)
        # pd.set_option('display.max_columns', None)
        # pd.set_option('display.width', None)
        # pd.set_option('display.max_colwidth', -1)

        # print(df_halve_maxevaltime)
        # print(df_maxevaltime30_evaluations)
    #endregion


    print("done.")
