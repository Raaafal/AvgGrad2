import copy
import ctypes
import subprocess
from time import sleep
import sys
import os



python_cmd="python"

err_file_name='errors.txt'
def save_errors(str):
    with open(err_file_name, "a") as text_file:
        text_file.write(str + "\n")

# first arg is the number of concurrent processes, then file names
def main():
    if ctypes.windll.shell32.IsUserAnAdmin()!=1:
        print('No Admin Rights')
        return
    args = sys.argv
    args.pop(0)
    p_num = int(args.pop(0))
    processes = args
    actual = []
    dir=None
    processes_from_dir_at_start=None

    if len(processes)==1 and os.path.isdir(processes[0]):
        dir=processes.pop(0)
        processes=[os.path.join(dir, f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
        processes_from_dir_at_start=copy.deepcopy(processes)

    pipe=open('errors.txt','a')
    while len(processes) != 0 or len(actual) != 0:
        while len(actual) < p_num and len(processes) != 0:
            name = processes.pop(0)
            try:
                #x=os.popen(name)
                #actual.append((name,subprocess.Popen([python_cmd, name], stdout=None, stderr=subprocess.PIPE, shell=True)))
                #actual.append((name,subprocess.Popen([python_cmd, name], stdout=None, stderr=None, shell=False)))
                actual.append((name,subprocess.Popen([python_cmd, name], stdout=subprocess.DEVNULL, stderr=pipe, shell=False)))
                print('started new process: '+name)
                sleep(10)
            except:
                output='failed to start process: '+name+'\n'
                print(output)
                save_errors(output)
            #sleep(30.0)

        for i in range(len(actual) - 1, -1, -1):
            processEnded = actual[i][1].poll() is not None
            if processEnded:
                # stdout, stderr = actual[i][1].communicate()
                print('process ended: '+actual[i][0])
                # #print(stdout)
                # err=str(stderr)
                # if len(err)>5:
                #     print(err)
                #     save_errors('Errors of '+actual[i][0]+':\n'+err)
                actual.pop(i)

        sleep(10.0)

        print('CHECK')
        if dir is not None:
            processes_new = [os.path.join(dir, f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
            #if len(processes_new)>len(processes_from_dir_at_start):
            for p in processes_new:
                if p not in processes_from_dir_at_start:
                    processes.append(p)
                    processes_from_dir_at_start.append(p)
                    print('Mew file detected in the directory: '+p)
            processes_to_remove=[]
            for p in processes_from_dir_at_start:
                if p not in processes_new:
                    processes_to_remove.append(p)
            for p in processes_to_remove:
                processes_from_dir_at_start.remove(p)
                try:
                    processes.remove(p)
                    print('Removed process: ' + p)
                except:
                    print("removing process " + p + " is impossible because it has already started")
    pipe.close()

    # d1 = subprocess.run(["python", "python1.py"], capture_output=True)
    # d2 = subprocess.run(["python", "python2.py"], capture_output=True)
    # d3 = subprocess.run(["python", "python3.py"], capture_output=True)
    #
    # print(d1.stdout)
    # print(d2.stdout)
    # print(d3.stdout)


if __name__ == "__main__":
    main()
