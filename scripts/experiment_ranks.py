from argparse import ArgumentError
from asyncio import tasks

from numpy import average

from utils.UpdateParameter import *
import subprocess
import time
import re
from os.path import exists
import sys

seeds=list(range(2,23))
port = int(10e4)
savefig_paths = ["results/figures/are_project", "/home/paran/Dropbox/BCAM/07_estancia_1/paper/images/are_project"]


n_tasks = 3
task_list = ["ExploreObstacles", "ExploreObstaclesDistanceBonus", "ExploreHardRace"]
scene_list = ["shapes_exploration.ttt", "shapes_exploration_bounus_4_distance.ttt", "hard_race.ttt"]


for index, task, scene in zip(range(n_tasks), task_list, scene_list):

    if len(sys.argv) != 2:
        raise ArgumentError("this script requires only one argument --plot --launch_local or --launch_cluster")

    if sys.argv[1] not in ("--plot", "--launch_local", "--launch_cluster"):
        raise ArgumentError("this script requires only one argument --plot --launch_local or --launch_cluster")


    # update parameters
    if sys.argv[1] in ("--launch_local", "--launch_cluster"):
        parameter_file = "experiments/nipes/parameters.csv"
        parameter_text = f"""
#experimentName,string,nipes
#subexperimentName,string,measure_ranks
#preTextInResultFile,string,seed_202
#resultFile,string,../results/data/ranks_results/ranks_exp_result_202.txt


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
#seed,int,202

#populationSize,int,40
#maxEvalTime,float,30.0
#maxNbrEval,int,2000
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
#incrPop,bool,0
#lengthOfStagnation,int,200
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
"""

        mass_update_parameters(parameter_file, parameter_text)




    #region local_launch

    if sys.argv[1] == "--launch_local":
        from joblib import Parallel, delayed

        n_jobs = 5

        def run_with_seed(seed):
            time.sleep(int(1.25*seed) % n_jobs)
            update_parameter(parameter_file, "seed", str(seed))
            update_parameter(parameter_file, "resultFile", f"../results/data/ranks_results/{task}_ranks_exp_result_{seed}.txt")
            update_parameter(parameter_file, "preTextInResultFile", f"seed_{seed}")
            exec_res=subprocess.run(f"bash launch.sh --coppelia --parallel -e=nipes",shell=True, capture_output=True)
            with open(f"{task}_ranks_logs_{seed}.txt", "w") as f:
                f.write("------------------")
                f.write("OUT: ")
                f.write(exec_res.stdout.decode("utf-8"))
                f.write("ERR: ")
                f.write(exec_res.stderr.decode("utf-8"))
                f.write("------------------")
            

        Parallel(n_jobs=n_jobs, verbose=12)(delayed(run_with_seed)(i) for i in seeds)



    #endregion


    #region launch_cluster

    if sys.argv[1] == "--launch_cluster":


        def run_with_seed(seed, port):
            update_parameter(parameter_file, "seed", str(seed))
            update_parameter(parameter_file, "resultFile", f"../results/data/ranks_results/{task}_ranks_exp_result_{seed}.txt")
            update_parameter(parameter_file, "preTextInResultFile", f"seed_{seed}")
            # Parallel
            subprocess.run(f"bash launch.sh -e=nipes --vrep --cluster --parallel --port={port} > {task}_ranks_logs_{seed}.txt 2>&1",shell=True)
            # # Sequential
            # subprocess.run(f"bash launch.sh -e=nipes --cluster --port={port} --sequential",shell=True)
            print(f"Launched experiment with seed {seed} in port {port}.")

        for i in seeds:
            time.sleep(1.0)
            run_with_seed(i, port)
            port += int(10e4)



    #endregion






    #region plot

    if sys.argv[1] == "--plot":


        import matplotlib.pyplot as plt
        from matplotlib.ticker import StrMethodFormatter
        from scipy.stats import rankdata

        from cycler import cycler
        line_cycler   = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442", "#000000"]) +
                        cycler(linestyle=["-", "--", "-.", ":", "-", "--", "-.","-"]))
        marker_cycler = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442", "#000000"]) +
                        cycler(linestyle=["none", "none", "none", "none", "none", "none", "none","none"]) +
                        cycler(marker=["4", "2", "3", "1", "+", "x", ".",'*']))
        plt.rc("axes", prop_cycle=line_cycler)


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


        def get_ranks_resolve_ties_with_best(array):
            return rankdata(-array, method="average") - 1

        # get the avg. rank that [the best controllers in fitness_array_ref] have in fitness_array 
        def get_avg_ranks_of_best(fitness_array_ref, fitness_array):
            ranks_ref = get_ranks_resolve_ties_with_best(fitness_array_ref)
            ranks = get_ranks_resolve_ties_with_best(fitness_array)
            return np.average(ranks[ranks_ref == 0])

        def the_best_has_enough_fitness_to_make_it_to_next_halving(fitness_array_ref, fitness_array, i):
            if i==-1:
                return 1
            if sum(abs(fitness_array_ref - fitness_array)) < 0.5:
                return 1
            position_just_enough_in_sorted = int(  (1.0 - pow(0.5,i+1)) * float(len(fitness_array)-1) )
            positin_best_in_ref = np.argmax(fitness_array_ref)

            return int(fitness_array[positin_best_in_ref] >= sorted(fitness_array)[position_just_enough_in_sorted])

        dataframe_data = []
        for i in seeds:
            if not exists(f"results/data/ranks_results/{task}_ranks_exp_result_{i}.txt"):
                continue

            with open(f"results/data/ranks_results/{task}_ranks_exp_result_{i}.txt") as f:
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

        did_best_make_it_to_next_halving = []
        rank_distance_column = []
        avg_distance_of_best_column = []
        random_rank_distance_column = []
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
                ref_fitness = row.loc["fitness"][:]
                count_n_with_this_ref = 0
                
            else:
                assert _reference_runtime >= row.loc["runtime"]
                is_reference_column.append(False)
                count_n_with_this_ref += 1

            n_with_this_ref[i-count_n_with_this_ref:i+1] = count_n_with_this_ref
            rank_distance_column.append(distance_between_orders(ref_rank, row.loc["ranks"]))
            avg_distance_of_best_column.append(get_avg_ranks_of_best(ref_fitness, row.loc["fitness"]))

            copy_array = np.copy(row.loc["ranks"])
            np.random.shuffle(copy_array)
            random_rank_distance_column.append(distance_between_orders(ref_rank, copy_array))
            did_best_make_it_to_next_halving.append(
                the_best_has_enough_fitness_to_make_it_to_next_halving(ref_fitness, row.loc["fitness"], count_n_with_this_ref-1)
            )

        df['is_reference'] = is_reference_column
        df['distance_to_ref'] = rank_distance_column
        df['distance_to_random_ref'] = random_rank_distance_column
        df['n_with_this_ref'] = n_with_this_ref
        df['avg_distance_of_best'] = avg_distance_of_best_column
        df['did_best_make_it_to_next_halving'] = did_best_make_it_to_next_halving
        # print(df[df["seed"]==20])


        # Discard incomplete files: files with not enough lines (or too many), 
        # files with not enough values per ref. permu (or too many). 

        # Most frequent number of rows with the same ref permu
        most_frequent_n_with_this_ref = int(df.mode(axis='index', numeric_only=True).loc[0,"n_with_this_ref"])

        # Most frequent number of rows with the same seed
        usuall_number_of_rows = int(df["seed"].value_counts().mode()[0])

        # Seeds that have these number of rows.
        seeds_with_usual_number_of_rows = np.array(df["seed"].value_counts()[df["seed"].value_counts() == usuall_number_of_rows].index, dtype=np.int64)

        df = df[df["n_with_this_ref"] == most_frequent_n_with_this_ref]
        df = df[df["seed"].isin(seeds_with_usual_number_of_rows)]


        print("There were",len(df["seed"].unique()),"valid result files.")

        evaluations = sorted(df["evals"].unique())
        runtimes = sorted(df["runtime"].unique())

        y_random_list = []
        for runtime in runtimes:
            x = []
            y = []
            y_random = []
            for evals in evaluations:
                sub_df_with_specific_evals_runtime = df[(df["runtime"] == runtime) & (df["is_reference"] == False) & (df["evals"] == evals)]
                x += [evals]
                y += [sub_df_with_specific_evals_runtime["distance_to_ref"].mean()]
                y_random += [sub_df_with_specific_evals_runtime["distance_to_random_ref"].mean()]
            y_random_list.append(np.array(y_random))
            plt.plot(x,y, label=str(runtime))
        


        plt.plot(x,np.quantile(np.array(y_random_list), 0.5, 0), color="gray", label=str("random"))
        plt.fill_between(x, np.quantile(np.array(y_random_list), 0.05, 0), np.quantile(np.array(y_random_list), 0.95, 0), color='b', alpha=.1)

        plt.ylabel("Distance to reference")
        plt.xlabel("Evaluations")
        plt.legend()
        plt.tight_layout()
        for path in savefig_paths:
            plt.savefig(path + f"/{task}_distance_to_true_ranking_with_respect_to_maxevaltime.pdf")
        plt.close()

        x = runtimes
        y = [df[(df["runtime"]==el) & (df["is_reference"] == False)]['distance_to_ref'].mean() for el in runtimes]

        plt.plot(x,y, marker='.')
        mean_random_distance = (sum(y_random_list) / len(y_random_list)).mean()
        plt.plot([x[0], x[-1]], [mean_random_distance]*2, color = "gray")
        plt.xlabel("Runtime")
        plt.ylabel("Average distance")
        plt.tight_layout()
        for path in savefig_paths:
            plt.savefig(path + f"/{task}_distance_to_true_ranking_with_respect_to_maxevaltime_AVGD.pdf")
        plt.close()


        for runtime in runtimes:
            x = []
            y = []
            for evals in evaluations:
                sub_df_with_specific_evals_runtime = df[(df["runtime"] == runtime) & (df["is_reference"] == False) & (df["evals"] == evals)]
                x += [evals]
                y += [sub_df_with_specific_evals_runtime["avg_distance_of_best"].mean()]
            y_random_list.append(np.array(y_random))
            plt.plot(x,y, label=str(runtime))
        

        plt.ylabel("avg_distance_of_best")
        plt.xlabel("Evaluations")
        plt.text(0,0,"""First we compute which are the best 
controllers with maxEvalTime=30.
Then, we see what is the average
rank of these best with different
(lower) maxEvalTimes.""")
        plt.legend()
        plt.tight_layout()
        for path in savefig_paths:
            plt.savefig(path + f"/{task}_avg_distance_of_best.pdf")
        plt.close()



        x = []
        y = []
        for evals in evaluations:
            y_list = []
            for seed in seeds:
                sub_df_with_specific_evals_runtime = df[(df["is_reference"] == False) & (df["evals"] == evals) & (df["seed"] == seed)]
                y_list.append(sub_df_with_specific_evals_runtime["did_best_make_it_to_next_halving"].product())
            x += [evals]
            y += [average(y_list)]
        plt.plot(x,y)
        
        plt.gca().set_ylim((0,1))
        plt.ylabel("Probability best makes it through last halving")
        plt.xlabel("Evaluations")
        plt.text(0,0,"""First we compute which is the best 
controllers with maxEvalTime=30 (only one best chosen).
Then, we see what is the probability
that with the halvings, this
controller still makes it to be
evaluated for 30s.
""")
        for path in savefig_paths:
            plt.savefig(path + f"/{task}_prob_best_makes_it_to_last_halving.pdf")
        plt.close()



    #endregion


    print("done.")
