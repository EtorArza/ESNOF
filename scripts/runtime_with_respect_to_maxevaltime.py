from matplotlib import pyplot as plt
import numpy as np
from cycler import cycler
from statistics import mean,median
from pylab import polyfit
import subprocess
from utils.UpdateParameter import update_parameter

savefigpath = "/home/paran/Dropbox/BCAM/07_estancia_1/code/results/figures/"
experimentCsvPath="/home/paran/Dropbox/BCAM/07_estancia_1/code/experiments/nipes/parameters.csv"
experimentName = "nipes"

def get_evaluation_runtimes(maxEvalTime):
    print("Computing runtimes for MaxEvalTime =", maxEvalTime)
    update_parameter(experimentCsvPath, "maxEvalTime", maxEvalTime)
    update_parameter(experimentCsvPath, "maxNbrEval", 100)
    exec_res=subprocess.run(f"bash launch.sh -e={experimentName}",shell=True, capture_output=True)  
    txt_res= str(exec_res.stdout).strip("'b").strip('"').strip("'")

    def extract_runtimes(all_lines_string: str):
        res=[]
        line_list = all_lines_string.split(r"\n")
        for line in line_list:
            if "update() " in line:
                res.append(float(line.strip("update() ")))
        return res
    return extract_runtimes(txt_res)


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



max_eval_time_120 = get_evaluation_runtimes(120.0)
max_eval_time_90 = get_evaluation_runtimes(90.0)
max_eval_time_60 = get_evaluation_runtimes(60.0)
max_eval_time_30 = get_evaluation_runtimes(30.0)
max_eval_time_15 = get_evaluation_runtimes(15.0)
max_eval_time_5 = get_evaluation_runtimes(5.0)
max_eval_time_0_5 = get_evaluation_runtimes(0.5)

average_runtimes = [mean(ds) for ds in [max_eval_time_0_5,max_eval_time_5,max_eval_time_15,max_eval_time_30,max_eval_time_60,max_eval_time_90,max_eval_time_120]]


fig, ax = plt.subplots(1,1)
load_bw_theme(ax)

ax.boxplot(max_eval_time_0_5, positions=[5],widths=[40])
ax.boxplot(max_eval_time_5,positions=[50],widths=[40])
ax.boxplot(max_eval_time_15,positions=[150],widths=[40])
ax.boxplot(max_eval_time_30,positions=[300],widths=[40])
ax.boxplot(max_eval_time_60,positions=[600],widths=[40])
ax.boxplot(max_eval_time_90,positions=[900],widths=[40])
ax.boxplot(max_eval_time_120,positions=[1200],widths=[40])


x = np.array([5,50,150,300,600,900,1200])
y = np.array(average_runtimes) 


m,b = polyfit(x, y, 1)
ax.plot([4,1200], [5*m+b, 1200*m+b], label=f"$f(x) = {m:2f} \cdot x  + {b:2f}$", linestyle="--")

ax.scatter([5,50,150,300,600,900,1200], average_runtimes, marker="+", label="Average")
ax.scatter([5],median(max_eval_time_0_5),marker="_", color="orange", label="Median")
ax.legend()

plt.xlabel("Number of ticks, x")
plt.ylabel("t(x) Time for evaluating a controller in seconds.")
plt.savefig(savefigpath + "runtime_of_one_controller_evaluation_with_respect_to_max_eval_time.pdf")
plt.close()
