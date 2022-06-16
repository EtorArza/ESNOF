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
level_list = ["1-4","2-1","3-1", "4-1", "4-2", "5-1", "6-2", "6-4"]

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
                exec_res=subprocess.run(f"python3 other_RL/super-mario-neat/src/main.py train --gen {new_gens} --level {level} --seed {seed} --method {method} --resultfilename results/data/super_mario/level_{level}_{method}_{seed}.txt --gracetime {gracetime}",shell=True, capture_output=True)
                #exec_res=subprocess.run(f"python3 other_RL/super-mario-neat/src/main.py train --gen {new_gens} --level {level} --seed {seed} --method {method} --resultfilename results/data/super_mario/level_{level}_{method}fincrementsize_{seed}.txt --gracetime {gracetime} --fincrementsize {fincrementsize}",shell=True, capture_output=True)
            
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

        savefig_paths = ["results/figures/super_mario/", "/home/paran/Dropbox/BCAM/07_estancia_1/paper/images/super_mario/"]

        
        df_row_list = []
        for fincrementsize in ("", "fincrementsize"):
            for method in methods:
                for seed in seeds:
                    i = 0
                    res_filepath = f"results/data/super_mario/level_{level}_{method}{fincrementsize}_{seed}.txt"
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
                                df_row_list.append([seed, evals, rw_time, fitness, maxevaltimes_each_controller, clock_time, method, level, fincrementsize])
                                i += 1
                    print(i, "rows:", res_filepath)
        df_all = pd.DataFrame(df_row_list, columns=["seed", "evals", "rw_time", "fitness", "maxevaltimes_each_controller", "simulated_time", "method", "level", "fincrementsize"])

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
                    if len(f_with_seed_and_runtime_leq) > 4:
                        fitnesses.append(max(f_with_seed_and_runtime_leq))
                    else:
                        fitnesses.append(0)
                x.append(runtime)
                every_y.append(fitnesses)
                y_median.append(np.quantile(np.array(fitnesses), 0.5))
                y_lower.append(np.quantile(np.array(fitnesses), 0.25))
                y_upper.append(np.quantile(np.array(fitnesses), 0.75))
            
            return x, y_median, y_lower, y_upper, every_y



        x_min = 0.0
        y_min = 5

        x_max = 58616.0
        x_nsteps = 200


        for fincrementsize in ("", "fincrementsize"):

            x_list = []
            y_median_list = []
            y_lower_list = []
            y_upper_list = []
            every_y_halve_list = []
            for method in methods:

                x, y_median, y_lower, y_upper, every_y_halve = get_xy_from_df("simulated_time", x_min, x_max, x_nsteps, df_all[(df_all["method"] == method) & (df_all["fincrementsize"] == fincrementsize)])
                x_list.append(x)
                y_median_list.append(y_median)
                y_lower_list.append(y_lower)
                y_upper_list.append(y_upper)
                every_y_halve_list.append(every_y_halve)


            def get_test_result(x, y, alpha = 0.05):
                x = x[0:min(len(x), len(y))]
                y = y[0:min(len(x), len(y))]
                if len(x) < 5:
                    print("WARNING: statistical test with less than 5 samples. Probably wont be significant.")
                return mannwhitneyu(x, y, alternative='two-sided')[1] < alpha


            # # This assertion is required for doing the tests. We are comparing the samples based on the samples
            # # in every_y_halve and every_y_constant. Consequently, the indexes in these samples need to correspond 
            # # to the same x values.
            # assert x_constant == x_halve             
            # test_results_true = np.where([get_test_result(every_y_halve[i], every_y_constant[i]) for i in range(len(every_y_halve))])[0]



            # import code
            # code.interact(local=locals())



            plt.figure()
            plt.xlim((0, x_max))
            for x, y_median, y_lower, y_upper, every_y_halve, method, color in zip(x_list, y_median_list, y_lower_list, y_upper_list, every_y_halve_list, methods, ["red", "green", "blue"]):
                plt.plot(x, y_median, marker="", label=f"{method}", color=color)
                plt.fill_between(x, y_lower, y_upper, color=color, alpha=.1)
                # plt.plot(np.array(x_halve)[test_results_true], np.repeat(y_min, len(test_results_true)), linestyle="None", marker = "_", color="black", label="$p < 0.05$")
                # plt.scatter(df_halve_maxevaltime["rw_time"], df_halve_maxevaltime["fitness"], marker="o", label = "halve runtime", alpha=0.5, color="red")
            plt.legend()
            for path in savefig_paths:
                plt.savefig(path + f"/level_{level}_{fincrementsize}_exp_line.pdf")
            plt.close()
    #endregion


    print("done.")
