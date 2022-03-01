import os

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