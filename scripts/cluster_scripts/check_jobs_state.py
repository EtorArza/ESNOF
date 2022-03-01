import os
import sys


def usage():
    print("""Usage:
# To get job_id
python3 scripts/cluster_scripts/check_jobs_runing.py /home/paran/Dropbox/BCAM/07_estancia_1/code/client_03_01_15_48_44_484453.out job_id
# To get time since last 
python3 scripts/cluster_scripts/check_jobs_runing.py /home/paran/Dropbox/BCAM/07_estancia_1/code/client_03_01_15_48_44_484453.out time
""")   

def check_slurm_out_files_for_errors():

    dir_list = list(filter(lambda x: "slurm" in x, os.listdir("./")))
    for file_path in dir_list:
        faulty_count = 0
        error_count = 0
        with open(file_path, "r") as f:
            lines = f.readlines()
            lines = [el.strip("\n\t") for el in lines]
            for line in lines:
                if "One V-REP instance is faulty since I tried to connect to it for more than 3 times." in line:
                    faulty_count+=1
                if "ERROR :" in line:
                    error_count += int(line.split("ERROR :")[-1])
        status = ("ERROR", "OK")[int(error_count+faulty_count == 0)]
        print("{file_path} -> {status} ({faulty_count}, {error_count})".format(file_path=file_path, status=status, faulty_count=faulty_count, error_count=error_count))



def check_client_files(file_path, mode):
    with open(file_path, "r") as f:
        lines = f.readlines()
        lines = [el.strip("\n") for el in lines if "epoch" in el]
    linux_time_last=None
    linux_time=None
    for line in lines[-2:]: # check last two lines
        if len(line) < 5:
            continue
        line = line.removeprefix("- epoch(), ")
        line = line.replace(" ", "")
        line = line.replace("=", ",")
        line = line.split(",")
        line_dict = dict(zip(*[iter(line)]*2))
        linux_time_last = linux_time
        linux_time = int(line_dict["time"])
        job_id = int(line_dict["preTextInResultFile"].split("_")[-1])

    if mode == "job_id":
        return job_id
    elif mode == "time":
        if linux_time_last == None:
            return 0
        else:
            return linux_time - linux_time_last
    else:
        print(f"ERROR: mode = {mode} not recognized. parameter 2 must be either job_id or time")
        usage()
        exit(1)
        
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("ERROR: two arguments expected.")
        usage()
        exit(1)
    print(check_client_files(sys.argv[1], sys.argv[2]))