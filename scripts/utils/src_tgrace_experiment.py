import csv
import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from typing import Callable, Any, Iterable, Tuple
from tqdm import tqdm as tqdm
from copy import deepcopy
from termcolor import colored



def exit_after_k():
    exit_after_k.exit_call_counter += 1
    if exit_after_k.exit_call_counter == 2:
        exit(0)
exit_after_k.exit_call_counter = 0

def print_array_with_highlight(arr, color, position):
    assert position >= 0
    """
    Print the numpy array in a single line with values separated by spaces,
    highlighting the element at the specified position with the given color.

    Parameters:
    - arr (numpy.ndarray): The input numpy array.
    - color (str): The color for highlighting (e.g., 'red', 'green', 'blue', etc.).
    - position (int): The position of the element to be highlighted.

    Returns:
    - None
    """
    formatted_array = " ".join(map(str, arr))
    formatted_array = formatted_array.split()
    formatted_array[position] = colored(formatted_array[position], color)
    formatted_array = " ".join(formatted_array)
    print(f"[{formatted_array}]") 

class ObjectiveLogger:
    def __init__(self, file_path, replace_existing=False, logevery=1):
        # Create the directory if it doesn't exist
        log_dir = os.path.dirname(file_path)
        self.logevery=logevery
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Check if the log file already exists
        if os.path.exists(file_path):
            if replace_existing:
                os.remove(file_path)
            else:
                raise FileExistsError(f"The log file '{file_path}' already exists.")

        self.file_path = file_path
        self.row_count = 0  # Initialize row count

        # Open the file and create the CSV writer
        self.csvfile = open(self.file_path, 'a', newline='')
        self.writer = csv.writer(self.csvfile)

        # self.header_written = False


    def log_values(self, time, values):

        # # We use no header for compatibility with different number of rows.
        # if not self.header_written:
        #     # Write header row
        #     self.writer.writerow(["time"] + list(range(len(values[::self.logevery]))))
        #     self.header_written = True


        # Increment row count and add it as the first value
        self.row_count += 1

        # # Round objective values to 3 decimals
        # rounded_values = [round(val, 3) for val in values]

        values = values[::self.logevery]

        # Write the row
        self.writer.writerow([time] + list(values))

    def close(self):
        # Close the CSV file
        self.csvfile.close()

class tgrace_exp_figures():

    def _get_seed_from_filepath(self, filepath:str):
        return int(filepath.split("_")[-1].removesuffix(".txt"))

    def _get_filepath_list(self, experiment_name, experiment_result_path):
        res = []
        filename_list = os.listdir(experiment_result_path)
        filename_list = filter(lambda x: ".txt" in x, filename_list)
        filename_list = sorted(filename_list, key=lambda x: self._get_seed_from_filepath(x))

        for filename in filename_list:
            if filename.startswith(experiment_name):
                file_path = os.path.join(experiment_result_path, filename)
                res.append(file_path)
        return res

    def _get_max_row_length(self, filename_list):
        largest_column_count = 0
        for filepath in tqdm(filename_list, desc="counting n-columns"):
            with open(filepath, 'r') as temp_f:
                lines = temp_f.readlines()
                for l in lines:
                    column_count = len(l.split(",")) + 1
                    largest_column_count = max(column_count, largest_column_count)
        return largest_column_count


    def __init__(self, experiment_name, experiment_result_path):


        print("TODO: Reporting objective value when environment terminates the evaluation.")
        self.combined_df: pd.DataFrame = None
        self.seed_list = []
        self.experiment_name = experiment_name
        filepath_list = self._get_filepath_list(experiment_name, experiment_result_path)
        self.largest_column_count = self._get_max_row_length(filepath_list)
        column_names = ["time"]+[i for i in range(0, self.largest_column_count-2)]

        for filepath in filepath_list:
            seed = self._get_seed_from_filepath(filepath)
            self.seed_list.append(seed)
            df = pd.read_csv(filepath, header=None, delimiter=",", names=column_names)
            df.insert(0, "seed", seed)
            self.combined_df = pd.concat([self.combined_df, df], axis=0).reset_index(drop=True)
        self.drop_seeds_with_low_time()
        self.reset_refs_stopping()
        self.ref_len = None




    def drop_seeds_with_low_time(self):
        prop_maxtime_drop = 0.9
        max_time_indices = self.combined_df.groupby('seed')['time'].idxmax()
        self.t_max = max(self.combined_df.loc[max_time_indices]["time"]) * prop_maxtime_drop
        max_time_rows = self.combined_df.loc[max_time_indices]["time"]


        print(f"Drop rows with a total time lower than {prop_maxtime_drop} * max_time.")
        filtered_series = max_time_rows[max_time_rows < self.t_max]
        indexes_below_0_9 = filtered_series.index.tolist()
        for idx_to_drop in indexes_below_0_9:
            seed = int(self.combined_df.loc[idx_to_drop]["seed"])
            self.combined_df = self.combined_df[self.combined_df['seed'] != seed]
        self.combined_df.reset_index(drop=True)
        self.seed_list = list(self.combined_df["seed"].unique())
        print(f"{len(indexes_below_0_9)} seeds where discarded, and a total of {len(self.seed_list)} valid seeds are left.")


    def reset_refs_stopping(self):
        self.gesp_refs = None
        self.gesp_current_steps = None
        self.gesp_current_steps_w_gesp = None
        self.gesp_current_best_f = None
        self.gesp_current_best_f_w_gesp = None


    def _get_ratio_where_gesp_worse(self):
        # It needs to be the value at the end. Otherwise the time graph makes no sense.
        w_gesp_f = self.gesp_current_best_f_w_gesp[-1]
        w_gesp_step = self.gesp_current_steps_w_gesp[-1]
        wo_gesp_f = None

        for step,f in zip(self.gesp_current_steps, self.gesp_current_best_f):
            if step > w_gesp_step:
                break
            wo_gesp_f = f
        assert wo_gesp_f != None
        comp = lambda a, b: 0.5 if a == b else 1 if a > b else 0
        return comp(wo_gesp_f, w_gesp_f) # without gesp better -> 1.0 

        # This is the old code, in which we compute which is better for every time step
        # is_with_gesp_better = []
        # i_gesp=0
        # for step, f in zip(self.gesp_current_steps, self.gesp_current_best_f):
        #     if step > self.gesp_current_steps_w_gesp[-1]:
        #         break
        #     while i_gesp < len(self.gesp_current_steps_w_gesp) and self.gesp_current_steps_w_gesp[i_gesp] <= step:
        #         comp = lambda a, b: 0.5 if a == b else 1 if a > b else 0
        #         is_gesp_better = comp(self.gesp_current_best_f_w_gesp[i_gesp], f) 
        #         i_gesp+=1
        #     is_with_gesp_better.append(is_gesp_better)
        # return is_with_gesp_better[-1] 

    def when2stopGESP(self, f_array, t_grace_proportion):
        assert t_grace_proportion <= 1.0


        # Create copy where nans are -infty
        f_array_no_nans = np.copy(f_array)
        f_array_no_nans[np.where(np.isnan(f_array_no_nans))[0]] = np.float64(-1e10)

        # First call: fill self.gesp_refs 
        if self.gesp_refs is None:
            self.gesp_refs = np.copy(f_array_no_nans)

        # Get episode length
        max_episode_length = len(f_array)
        episode_length_wo_gesp = len(f_array) - len(np.where(np.isnan(f_array))[0])

        _grace_steps = round(t_grace_proportion * len(f_array))

        # print("np.max(",f_array[:max_episode_length-_grace_steps], "|",       f_array[_grace_steps:], ") < ")
        # print("np.max(",self.gesp_refs[:max_episode_length-_grace_steps], "|", self.gesp_refs[_grace_steps:], ")")
        _episode_length_w_gesp =  np.where(
            np.maximum(       f_array[:max_episode_length-_grace_steps],        f_array[_grace_steps:]) < 
            np.minimum(self.gesp_refs[:max_episode_length-_grace_steps], self.gesp_refs[_grace_steps:])
        )[0]
        episode_length_w_gesp = (_episode_length_w_gesp[0] + 1 + _grace_steps) if len(_episode_length_w_gesp) > 0 else episode_length_wo_gesp

        # print(episode_length_w_gesp)
        # print(episode_length_wo_gesp)
        # print("---")
        # exit_after_k()

        # Find why the evaluation stopped assuming gesp is applied.
        if episode_length_w_gesp == max_episode_length:
            why_evaluation_stopped = "maxepisodelength" # If episode evaluated until the end
        elif episode_length_w_gesp == episode_length_wo_gesp and episode_length_wo_gesp < max_episode_length: 
            why_evaluation_stopped = "problemspecific" # Evaluation stopped by environment (i.e mario died)
        elif episode_length_w_gesp < episode_length_wo_gesp:
            why_evaluation_stopped = "gesp" # Gesp early stopped the evaluation.
        else:
            raise ValueError(f"One of the previous three conditions should be True.\n episode_length_w_gesp = {episode_length_w_gesp}\n episode_length_wo_gesp = {episode_length_wo_gesp}\n max_episode_length = {max_episode_length}\n")

        assert ("supermario" in self.experiment_name) or (episode_length_wo_gesp == max_episode_length), "The environment needs to have a monotone increasing f when there are problem specific early stopping criteria."

        if "supermario" in self.experiment_name:
            is_better_than_best_found_wo_gesp = np.max(f_array_no_nans) > np.max(self.gesp_refs)
        else:
            is_better_than_best_found_wo_gesp = f_array_no_nans[-1] > self.gesp_refs[-1]


        if is_better_than_best_found_wo_gesp:
            self.gesp_refs = f_array_no_nans[:]

        was_early_stopped = not (why_evaluation_stopped != "gesp")

        if self.gesp_current_steps is None:
            self.gesp_current_steps = [episode_length_wo_gesp]
            self.gesp_current_steps_w_gesp = [episode_length_w_gesp]
            self.gesp_current_best_f = [f_array[episode_length_wo_gesp-1]]
            self.gesp_current_best_f_w_gesp = [f_array[episode_length_w_gesp-1]]
        else:
            self.gesp_current_steps.append(       self.gesp_current_steps[-1]        + episode_length_wo_gesp)
            self.gesp_current_steps_w_gesp.append(self.gesp_current_steps_w_gesp[-1] + episode_length_w_gesp)
            self.gesp_current_best_f.append(       max(self.gesp_current_best_f[-1],        f_array[episode_length_wo_gesp-1]))
            self.gesp_current_best_f_w_gesp.append(max(self.gesp_current_best_f_w_gesp[-1], f_array[episode_length_w_gesp-1]))


        # print("max_episode_length",max_episode_length)
        # print("episode_length_wo_gesp",episode_length_wo_gesp)
        # print("episode_length_w_gesp",episode_length_w_gesp)
        # print_array_with_highlight(f_array,"yellow", episode_length_wo_gesp-1)
        # print("f_wo_gesp",f_array[episode_length_wo_gesp-1])
        # print_array_with_highlight(f_array,"yellow", episode_length_w_gesp-1)
        # print("f_w_gesp",f_array[episode_length_w_gesp-1])
        # print("why_evaluation_stopped",why_evaluation_stopped)
        # print("---")
        # input("Press Enter key to continue...")


        res = {
            "episode_length_w_gesp": self.gesp_current_steps_w_gesp[-1],
            "episode_length_wo_gesp": self.gesp_current_steps[-1],
            "is_better_than_best_found_wo_gesp": is_better_than_best_found_wo_gesp,
            "was_early_stopped": was_early_stopped,
        }

        return res



    def get_proportion_timesaved_bestsolsmised(self, when2stopfunc):
        """
        when2stopfunc: Given a array of f-values, it tells you the index in which the evaluation would be stopped. 
        """
        proportion_best_missed_list = []
        proportion_frames_evaluated_list = []
        proportion_with_gesp_worse_list = []
        assert self.combined_df.shape[0] > 2, "Dataframe is empty or has only one row."
        for seed in self.seed_list:
            self.reset_refs_stopping()
            iterable_of_observed_f = [el[1][2:].to_numpy() for el in self.combined_df[self.combined_df['seed'] == seed].iterrows()]
            res_list = [when2stopfunc(row) for row in iterable_of_observed_f]
            was_best_found_missed = [res["is_better_than_best_found_wo_gesp"] and res["was_early_stopped"] for res in res_list if res["is_better_than_best_found_wo_gesp"]]
            assert len(was_best_found_missed) > 0
            proportion_best_missed = np.mean(np.int16(was_best_found_missed))
            proportion_best_missed_list.append(proportion_best_missed)
            proportion_frames_evaluated = res_list[-1]["episode_length_w_gesp"] / res_list[-1]["episode_length_wo_gesp"]
            proportion_frames_evaluated_list.append(proportion_frames_evaluated)
            proportion_with_gesp_worse_list.append(self._get_ratio_where_gesp_worse())
        return proportion_best_missed_list, proportion_frames_evaluated_list, proportion_with_gesp_worse_list


    def plot_tgrace_param(self):
        
        x = np.linspace(0.0, 1.0, 31, endpoint=True)
        y_missed_median = np.zeros_like(x, dtype=np.float64)
        y_missed_upper_75 = np.zeros_like(x, dtype=np.float64)
        y_missed_lower_75 = np.zeros_like(x, dtype=np.float64)
        y_frames_median = np.zeros_like(x, dtype=np.float64)
        y_frames_upper_75 = np.zeros_like(x, dtype=np.float64)
        y_frames_lower_75 = np.zeros_like(x, dtype=np.float64)
        y_better_median = np.zeros_like(x, dtype=np.float64)
        y_better_upper_75 = np.zeros_like(x, dtype=np.float64)
        y_better_lower_75 = np.zeros_like(x, dtype=np.float64)

        for i, t_grace in tqdm(list(enumerate(x))):
            proportion_best_missed_list, proportion_frames_evaluated_list, proportion_with_gesp_worse_list = self.get_proportion_timesaved_bestsolsmised(lambda x: self.when2stopGESP(x, t_grace))
            x[i] = t_grace
            y_missed_median[i] = np.mean(proportion_best_missed_list)
            y_frames_median[i] = np.mean(proportion_frames_evaluated_list)
            y_better_median[i] = np.mean(proportion_with_gesp_worse_list)

        plt.plot(x, y_missed_median, linestyle="-", color="#1f77b4", label="missed new best solution")
        plt.plot(x, y_frames_median, linestyle="--", color="#ff7f0e", label="steps computed")
        plt.plot(x, y_better_median, linestyle="-.", color="#2ca02c", label="worse with gesp")

        plt.xlabel(r"$t_{grace}$")
        plt.legend(title="Proportion of...")
        plt.savefig(f"results/figures/tgrace_experiment/{self.experiment_name}_proportion_average_no_runtime.pdf")
        plt.close()


    def plot_tgrace_param_with_time(self):

        print("Calculations on intervals during the opitmization procedure.")
        n_time_partition = 11
        n_tgrace_partitions = 11
        res_matrix = np.zeros(shape=(n_tgrace_partitions, n_time_partition))
        original_df = deepcopy(self.combined_df)
        progress_bar = tqdm(total=n_tgrace_partitions*n_time_partition*3)


        for plotname, residx in zip(["bestmissed","framesevalutaed","withgespbetter"], [0,1,2]):
            t_partition_values = list(np.linspace(0.0,self.t_max, num=n_time_partition+2))[1:-1]
            t_grace_values = list(np.linspace(0.0, 1.0, num=n_tgrace_partitions, endpoint=True))
            for j, t_partition in enumerate(t_partition_values):
                self.combined_df = self.combined_df[(self.combined_df['time'] < t_partition)]
                for i, t_grace in enumerate(t_grace_values):
                    res = self.get_proportion_timesaved_bestsolsmised(lambda x: self.when2stopGESP(x, t_grace))[residx]
                    res_matrix[(i,j)] = np.mean(res)
                    progress_bar.update(1)
                self.combined_df = deepcopy(original_df)
            vmin = min((1.0-np.max(res_matrix), np.min(res_matrix)))
            vmax = 1.0 - vmin
            plt.imshow(res_matrix, cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
            plt.colorbar(label='Values')
            plot_titles = ['Proportion in which with gesp missed \nan actually better solution.',
                           'Proportion of frames evaluated when \nusing GESP.',
                           'Probability that with gesp the score is \nworse for the same amount of steps.\n$t_{max}$ es distinto dependiendo de $t_{grace}$',]
            plt.title(plot_titles[residx])
            plt.yticks(range(len(t_grace_values)), ["{:.2f}".format(x) for x in t_grace_values])

            if residx == 2:
                plt.xticks(range(len(t_partition_values)), ["{:.1f}".format(x / (len(t_partition_values)-1)) for x in range(len(t_partition_values))])
                plt.xlabel(r"time with respect to $t_{max}$")
            else:
                plt.xticks(range(len(t_partition_values)), ["{:.1f}".format(x/3600) for x in t_partition_values])
                plt.xlabel("time (hours)")
            plt.ylabel(r"$t_{grace}$")
            plt.grid(visible=False)
            plt.tight_layout()
            plt.savefig(f"results/figures/tgrace_experiment/{self.experiment_name}_proportion_{plotname}.pdf")
            plt.close()




if __name__ == "__main__":
    # Call the function with default parameters
    exp = tgrace_exp_figures("veenstra", "results/data/tgrace_experiment/")
    exp.plot_tgrace_param_with_time()
    exp.plot_tgrace_param()
