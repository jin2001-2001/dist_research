import sys
import math
import os
import shutil
import csv
sys.path.append("./RCPSP")
sys.path.append("./Simulator")
sys.path.append("./Visiualization")
import test
import sample_generation as sg
import classical_jobshop as cj
import visiualize_pip as vp
import visiualize_pipsub as vps

def clear_scratch_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove directory
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")


def test_stramline(k_candidate, s_candidate, subtasksfb, subtasksg,rank_rule = 2):
    #import test case, and now generate DP stage 1 sols:
    (layers,ndevice, nmbatch,mbatchsize,band), score_list, plan_list, allo_list, profiler = test.test_DP_solver_onlytime_prime(k_candidate, s_candidate)

    band = band / 1024    #kB as unit

    current_amount_sol = len(score_list)

    score_dy_list = [[x] for x in score_list] ##important!: need further prove in the future... here we assue that socre is just time latency...
    best_score_dy_index = 0

    for idx, sublist,candidate, allo_plan in zip(range(current_amount_sol), score_dy_list, plan_list,allo_list):

        #print(candidate)
        B_ft, B_bt, B_fe, B_be, T_gathering, E_gathering, BatchAllocateList = profiler.getall(candidate)
        #print(B_ft, B_bt, B_fe, B_be, T_gathering, E_gathering)
        nsteps = len(B_ft)

        #get real score:
        real_score = vp.pip_ploting(num_stages = nsteps, num_microbatches = nmbatch,
                    forward_times = B_ft,
                    backward_times = B_bt,
                    gathering_times = T_gathering, enableshare = False, enablegraph = False)
        sublist.append(real_score)
        real_score = vp.pip_ploting(num_stages = nsteps, num_microbatches = nmbatch,
                    forward_times = B_ft,
                    backward_times = B_bt,
                    gathering_times = T_gathering, enableshare = True, enablegraph = False)
        #assert(int(real_score) == int(sublist[0]))

        sublist.append(real_score)


        #construct profile format:
        tprofile = list(zip([math.ceil(x) for x in B_ft], [math.ceil(y) for y in B_bt]))
        rprofile = [(int(band),int(band))]*(nsteps//2)
        grprofile = [int(band) if x > 0 else x for x in T_gathering]

        #call generation to generate an RCPSP problem input file:
        #print(tprofile,rprofile,T_gathering, grprofile)


        sg.generatesm(pfile = "./scratch/scratchtest.sm",s= nsteps, b=nmbatch, a=subtasksfb,
                    TProfile=tprofile, RProfile = rprofile, UnitR = int(band),
                    gatheringTprofile= T_gathering,gatheringRprofile= grprofile,
                      aa = subtasksg)

        #call RCPSP solver:
        model, result = cj.RCPSP_solver("./scratch/scratchtest.sm")
        #print(result)
        cj.RCPSP_plot(model, result)
        if result.objective>0:
            #get current best solver:
            best_score = result.objective
            opt_score = result.lower_bound
            sublist.append(best_score)
            sublist.append(opt_score)

            if(best_score<score_dy_list[best_score_dy_index][rank_rule]):
                best_score_dy_index = idx

            #print("RESULT:\n")
            #print(candidate)
            ###ploting work:
            vp.pip_ploting(num_stages = nsteps, num_microbatches = nmbatch,
                        forward_times = B_ft,
                        backward_times = B_bt,
                        gathering_times = T_gathering, enableshare = False, enablegraph = True,
                        storage = f"./scratch/plot{idx}_d2d.png",
                        group_plan = candidate,
                        ) 

            vp.pip_ploting(num_stages = nsteps, num_microbatches = nmbatch,
                        forward_times = B_ft,
                        backward_times = B_bt,
                        gathering_times = T_gathering, enableshare = True, enablegraph = True,
                        storage = f"./scratch/plot{idx}_shared.png",
                        group_plan = candidate)   
            vps.pip_ploting_direct(result.best.tasks,
                               s=nsteps, a=subtasksfb, b=nmbatch, aa=subtasksg,
                               storage = f"./scratch/plot{idx}_sharedOPT.png",
                               group_plan = candidate) 


    #print(score_dy_list)
    #print(best_score_dy_index)
    return score_dy_list, best_score_dy_index
                        

if __name__ == "__main__":
    scratch_dir = "./scratch"
    clear_scratch_folder(scratch_dir)

    k_candidate = 5
    s_candidate = 1
    subtasksfb = 5
    subtasksg = 6

    data = []
    data.append(["kcandidate", "scontainer", "bestTime","actualTime", "stage2Time"])
    os.makedirs("./record", exist_ok=True)
    with open("./record/ksresults.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data[0])
        for k_candidate in range(1,5):
            for s_candidate in range(1,2,10):
                print(f"recording k:{k_candidate},s:{s_candidate}")
                score_dy_list, best_score_dy_index = test_stramline(k_candidate, s_candidate, subtasksfb, subtasksg, rank_rule=2)
                if (score_dy_list==[]):
                    print(f"opps, for k:{k_candidate},s:{s_candidate} no sols...")
                    continue
                print([k_candidate, s_candidate,
                              score_dy_list[best_score_dy_index][1],
                              score_dy_list[best_score_dy_index][2],
                              score_dy_list[best_score_dy_index][3]
                              ])
                writer.writerow([k_candidate, s_candidate,
                              score_dy_list[best_score_dy_index][1],
                              score_dy_list[best_score_dy_index][2],
                              score_dy_list[best_score_dy_index][3]
                              ])
                
    
    
    print("CSV file written successfully.")