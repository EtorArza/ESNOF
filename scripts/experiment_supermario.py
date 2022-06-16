from argparse import ArgumentError

from utils.UpdateParameter import *
import subprocess
import time
import re
from os.path import exists
import sys
from tqdm import tqdm as tqdm
from joblib import Parallel, delayed

gens = 100
seeds = list(range(2,16))
gracetime = 40
fincrementsize = 150
parallel_threads = 7

savefig_paths = ["results/figures", "/home/paran/Dropbox/BCAM/07_estancia_1/paper/images"]

methods = ["constant", "nokill", "bestasref"]
level_list = ["1-1","1-2"]

index = -1
for level in level_list:
    index += 1

    if len(sys.argv) != 2:
        raise ArgumentError("this script requires only one argument --plot --launch_local")

    if sys.argv[1] not in ("--plot", "--launch_local"):
        raise ArgumentError("this script requires only one argument --plot --launch_local")


    #region local_launch

    if sys.argv[1] == "--launch_local":
        import itertools
        import time


        def run_with_seed(seed):

            time.sleep(0.5)
            print(f"Launching with seed {seed} in experiment_halveruntime.py ...")
            for method in methods:
                # # Reduce evaluations not needed in nokill if we kill all experiments in 1k iterations.
                #new_gens = str(gens if method != "nokill" else gens // 6)
                new_gens = gens
                print(f"python3 other_RL/super-mario-neat/src/main.py train --gen {new_gens} --level {level} --seed {seed} --method {method} --resultfilename results/data/super_mario/level_{level}_{method}_{seed}.txt --gracetime {gracetime}")
                # exec_res=subprocess.run(f"python3 other_RL/super-mario-neat/src/main.py train --gen {new_gens} --level {level} --seed {seed} --method {method} --resultfilename results/data/super_mario/level_{level}_{method}_{seed}.txt --gracetime {gracetime}",shell=True, capture_output=True)
                # exec_res=subprocess.run(f"python3 other_RL/super-mario-neat/src/main.py train --gen {new_gens} --level {level} --seed {seed} --method {method} --resultfilename results/data/super_mario/level_{level}_{method}fincrementsize_{seed}.txt --gracetime {gracetime} --fincrementsize {fincrementsize}",shell=True, capture_output=True)
            
        # for seed in seeds:
        #     run_with_seed_and_runtime(seed, "halving")
        Parallel(n_jobs=parallel_threads, verbose=12)(delayed(run_with_seed)(i) for i in seeds)


    #endregion






    #region plot

    if sys.argv[1] == "--plot":
        import itertools
        import pandas as pd
        from matplotlib import pyplot as plt
        import numpy as np
        from scipy.stats import mannwhitneyu

        savefig_paths = ["results/figures", "/home/paran/Dropbox/BCAM/07_estancia_1/paper/images"]

        subexperimentName="bestasref"

        df_row_list = []
        for seed in seeds:
            res_filepath = f"results/data/{subexperimentName}_results/{task}_{subexperimentName}_exp_result_{seed}.txt"
            if exists(res_filepath):
                with open(res_filepath, "r") as f:
                    all_text = f.readlines()
                    for line in all_text:
                        split_line = line.strip("\n").split(",")
                        fitness = float(split_line[1])
                        clock_time = float(split_line[2])
                        rw_time = float(split_line[3])
                        _ = float(split_line[4])
                        evals = int(split_line[5])
                        simulated_time = clock_time # evals * sim_time_coefs[0] + sim_time_coefs[1] * rw_time
                        physical_time = evals * physical_time_coefs[0] + physical_time_coefs[1] * rw_time
                        maxevaltimes_each_controller = [float(el) for el in split_line[6].strip("()").split(";") if len(el) > 0]
                        if float(fitness) < -10e200:
                            continue
                        df_row_list.append([seed, evals, rw_time, fitness, maxevaltimes_each_controller, physical_time, simulated_time])
        df_halve_maxevaltime = pd.DataFrame(df_row_list, columns=["seed", "evals", "rw_time", "fitness", "maxevaltimes_each_controller", "physical_time", "simulated_time"])

        # Discard incomplete files: files with not enough lines (or too many).
        def discard_seeds_with_diff_n_lines(df, df_name=""):


            # Most frequent number of rows with the same seed
            usuall_number_of_rows = int(df["seed"].value_counts().mode()[0])

            # Seeds that have these number of rows.
            seeds_with_usual_number_of_rows = np.array(df["seed"].value_counts()[df["seed"].value_counts() == usuall_number_of_rows].index, dtype=np.int64)

            res = df[df["seed"].isin(seeds_with_usual_number_of_rows)]
            print("Reduced " + str(df_name) + " from " + str(len(df["seed"].unique())) + " rows to " + str(len(res["seed"].unique())) + " rows.")

            return res


        df_row_list = []
        for seed in seeds:
            res_filepath = f"results/data/runtimewrtmaxevaltime_results/{task}_runtimewrtmaxevaltime_exp_result_{seed}_maxEvalTime_{30.0}.txt"
            if exists(res_filepath):
                with open(res_filepath, "r") as f:
                    all_text = f.readlines()
                    for line in all_text:
                        split_line = line.strip("\n").split(",")
                        fitness = float(split_line[1])
                        clock_time = float(split_line[2])
                        rw_time = float(split_line[3])
                        maxEvalTime = float(split_line[4])
                        evals = int(split_line[5])
                        simulated_time = clock_time # evals * sim_time_coefs[0] + sim_time_coefs[1] * rw_time
                        physical_time = evals * physical_time_coefs[0] + physical_time_coefs[1] * rw_time                        
                        if float(fitness) < -10e200:
                            continue
                        df_row_list.append([seed, evals, rw_time, fitness, physical_time, simulated_time])
        df_maxevaltime30_evaluations = pd.DataFrame(df_row_list, columns=["seed", "evals", "rw_time", "fitness", "physical_time", "simulated_time"])


        if df_maxevaltime30_evaluations.empty or df_halve_maxevaltime.empty:
            if df_maxevaltime30_evaluations.empty:
                print("Skipping task", task,", the dataframe df_maxevaltime30_evaluations.empty is empty.")
            if df_halve_maxevaltime.empty:
                print("Skipping task", task,", the dataframe df_halve_maxevaltime.empty is empty.")
            continue

        df_maxevaltime30_evaluations = discard_seeds_with_diff_n_lines(df_maxevaltime30_evaluations, "df_maxevaltime30_evaluations")
        df_halve_maxevaltime = discard_seeds_with_diff_n_lines(df_halve_maxevaltime, "df_halve_maxevaltime")

        for time_mode in ["rw_time", "physical_time", "simulated_time"]:

            # plt.figure()
            # plt.xlim((0, max((max(df_maxevaltime30_evaluations[time_mode]),max(df_halve_maxevaltime[time_mode])))))

            # plt.scatter(df_maxevaltime30_evaluations[time_mode], df_maxevaltime30_evaluations["fitness"], marker="x", label="Constant runtime", alpha=0.5, color="green")
            # plt.scatter(df_halve_maxevaltime[time_mode], df_halve_maxevaltime["fitness"], marker="o", label = "halve runtime", alpha=0.5, color="red")
            # plt.legend()
            # for path in savefig_paths:
            #     plt.savefig(path + f"/{task}_{subexperimentName}_{time_mode}_exp_scatter.pdf")
            # plt.close()

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
                        if len(f_with_seed_and_runtime_leq) != 0:
                            fitnesses.append(max(f_with_seed_and_runtime_leq))
                    if len(fitnesses) < 4:
                        continue
                    x.append(runtime)
                    every_y.append(fitnesses)
                    y_median.append(np.quantile(np.array(fitnesses), 0.5))
                    y_lower.append(np.quantile(np.array(fitnesses), 0.25))
                    y_upper.append(np.quantile(np.array(fitnesses), 0.75))
                
                return x, y_median, y_lower, y_upper, every_y



            x_min_halve = np.nanquantile([df_halve_maxevaltime[df_halve_maxevaltime["seed"] == s][time_mode].min() for s in range(seeds[-1])], 0.5)
            x_min_30s = np.nanquantile([df_maxevaltime30_evaluations[df_maxevaltime30_evaluations["seed"] == s][time_mode].min() for s in range(seeds[-1])], 0.5)

            x_min = max(x_min_halve, x_min_30s)
            y_min = min(min(df_halve_maxevaltime["fitness"]), min(df_maxevaltime30_evaluations["fitness"]))

            x_max = max(np.quantile(df_halve_maxevaltime[time_mode],0.9), np.quantile(df_maxevaltime30_evaluations[time_mode],0.9))
            x_nsteps = 200


            x_halve, y_halve_median, y_halve_lower, y_halve_upper, every_y_halve = get_xy_from_df(time_mode, x_min, x_max, x_nsteps, df_halve_maxevaltime)
            x_constant, y_constant_median, y_constant_lower, y_constant_upper, every_y_constant = get_xy_from_df(time_mode, x_min, x_max, x_nsteps, df_maxevaltime30_evaluations)

            def get_test_result(x, y, alpha = 0.05):
                x = x[0:min(len(x), len(y))]
                y = y[0:min(len(x), len(y))]
                if len(x) < 5:
                    print("WARNING: statistical test with less than 5 samples. Probably wont be significant.")
                return mannwhitneyu(x, y, alternative='two-sided')[1] < alpha


            # This assertion is required for doing the tests. We are comparing the samples based on the samples
            # in every_y_halve and every_y_constant. Consequently, the indexes in these samples need to correspond 
            # to the same x values.
            assert x_constant == x_halve             
            test_results_true = np.where([get_test_result(every_y_halve[i], every_y_constant[i]) for i in range(len(every_y_halve))])[0]



            # import code
            # code.interact(local=locals())



            plt.figure()
            plt.xlim((0, max((max(df_maxevaltime30_evaluations[time_mode]),max(df_halve_maxevaltime[time_mode])))))
            plt.plot(x_halve, y_halve_median, marker="", label=f"{subexperimentName} runtime", color="red")
            plt.fill_between(x_halve, y_halve_lower, y_halve_upper, color='red', alpha=.1)
            plt.plot(x_constant, y_constant_median, marker="", label="Constant runtime", color="green")
            plt.fill_between(x_constant, y_constant_lower, y_constant_upper, color='green', alpha=.1)
            plt.plot(np.array(x_halve)[test_results_true], np.repeat(y_min, len(test_results_true)), linestyle="None", marker = "_", color="black", label="$p < 0.05$")
            #plt.scatter(df_halve_maxevaltime["rw_time"], df_halve_maxevaltime["fitness"], marker="o", label = "halve runtime", alpha=0.5, color="red")
            plt.legend()
            for path in savefig_paths:
                plt.savefig(path + f"/{task}_{subexperimentName}_{time_mode}_exp_line.pdf")
            plt.close()
            




        
        x_maxevaltime = []
        y_maxevaltime_median = []
        y_maxevaltime_lower = []
        y_maxevaltime_upper = []
        y_n_that_have_maximum_maxevaltime_median = []
        y_n_that_have_maximum_maxevaltime_lower = []
        y_n_that_have_maximum_maxevaltime_upper = []
        vertical_lines_30s_iteration = []

        for evals in sorted(df_halve_maxevaltime["evals"].unique()):
            y_n_that_have_maximum_maxevaltime = []
            maxevaltimes_each_controller_list = []
            for seed in seeds:
                if len(df_halve_maxevaltime[(df_halve_maxevaltime["evals"]==evals) & (df_halve_maxevaltime["seed"]==seed)]) != 1:
                    continue
                runtimes_with_certain_evals_and_seed = df_halve_maxevaltime[(df_halve_maxevaltime["evals"]==evals) & (df_halve_maxevaltime["seed"]==seed)]["maxevaltimes_each_controller"].iloc[0]
                
                y_mean = np.mean(runtimes_with_certain_evals_and_seed)
                maxevaltimes_each_controller_list.append(y_mean)

                n_y_max = sum((1 if el > 29.5 else 0 for el in runtimes_with_certain_evals_and_seed))
                y_n_that_have_maximum_maxevaltime.append(n_y_max)

            if np.quantile(maxevaltimes_each_controller_list,0.5) > 29.5:
                vertical_lines_30s_iteration.append(evals)
                continue
            x_maxevaltime.append(evals)
            y_maxevaltime_median.append(np.quantile(np.array(maxevaltimes_each_controller_list), 0.5))
            y_maxevaltime_lower.append(np.quantile(np.array(maxevaltimes_each_controller_list), 0.25))
            y_maxevaltime_upper.append(np.quantile(np.array(maxevaltimes_each_controller_list), 0.75))
            y_n_that_have_maximum_maxevaltime_median.append(np.quantile(np.array(y_n_that_have_maximum_maxevaltime), 0.5))
            y_n_that_have_maximum_maxevaltime_lower.append(np.quantile(np.array(y_n_that_have_maximum_maxevaltime), 0.25))
            y_n_that_have_maximum_maxevaltime_upper.append(np.quantile(np.array(y_n_that_have_maximum_maxevaltime), 0.75))


        plt.figure()
        plt.xlim((0, max(df_halve_maxevaltime["evals"])))
        plt.plot(x_maxevaltime, y_n_that_have_maximum_maxevaltime_median, marker="", label="Avg. maxEvalTime", color="red")


        plt.fill_between(x_maxevaltime, y_n_that_have_maximum_maxevaltime_lower, y_n_that_have_maximum_maxevaltime_upper, color='red', alpha=.1)

        if len(vertical_lines_30s_iteration) != 0:
            ymin, ymax = plt.gca().get_ylim() 
            plt.vlines(x=vertical_lines_30s_iteration, ymin=ymin, ymax=ymax, colors='black', ls='--', lw=1, label='reset stopping criterion')


        #plt.scatter(df_halve_maxevaltime["rw_time"], df_halve_maxevaltime["fitness"], marker="o", label = "halve runtime", alpha=0.5, color="red")
        plt.legend()
        for path in savefig_paths:
            plt.savefig(path + f"/{task}_{subexperimentName}_exp_number_with_maximumMaxEvalTime_line.pdf")
        plt.close()

        plt.figure()
        plt.xlim((0, max(df_halve_maxevaltime["evals"])))
        plt.plot(x_maxevaltime, y_maxevaltime_median, marker="", label="Avg. maxEvalTime", color="red")


        plt.fill_between(x_maxevaltime, y_maxevaltime_lower, y_maxevaltime_upper, color='red', alpha=.1)
        #plt.scatter(df_halve_maxevaltime["rw_time"], df_halve_maxevaltime["fitness"], marker="o", label = "halve runtime", alpha=0.5, color="red")

        if len(vertical_lines_30s_iteration) != 0:
            ymin, ymax = plt.gca().get_ylim() 
            plt.vlines(x=vertical_lines_30s_iteration, ymin=ymin, ymax=ymax, colors='black', ls='--', lw=1, label='reset stopping criterion')


        plt.legend()
        for path in savefig_paths:
            plt.savefig(path + f"/{task}_{subexperimentName}_exp_avgMaxEvalTime_line.pdf")
        plt.close()

        # pd.set_option('display.max_rows', None)
        # pd.set_option('display.max_columns', None)
        # pd.set_option('display.width', None)
        # pd.set_option('display.max_colwidth', -1)

        # print(df_halve_maxevaltime)
        # print(df_maxevaltime30_evaluations)
    #endregion


    print("done.")
