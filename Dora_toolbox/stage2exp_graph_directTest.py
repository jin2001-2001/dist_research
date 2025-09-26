import sys
import math
import os
import shutil
import csv
import numpy as np
sys.path.append("./RCPSP")
sys.path.append("./Simulator")
sys.path.append("./Visiualization")
import test
import sample_generation as sg
import sample_generation_shift as sgf
import sample_generation_graph as sgg
import classical_jobshop as cj
import visiualize_pip as vp
import visiualize_pipsub as vps
import visiualize_pip_shift as vpf
import visiualize_pip_graph as vpg

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


def test_stramline(ratio1=0,ratio2=0,ratio3=0,B_ft = [1,3,5], B_bt = [2,4,6], rprofile = [(15,17)],T_gathering= [20,0,20], grprofile = [20,0,20],subtasksfb=5, UnitR = 20, subtasksg=6,nmbatch = 8,pct = 0.6):

    sd_list = [1,4,3,4,-1]

    nsteps = len(B_ft)
    candidate = [{'device':[0],'layer':[0]}]*((nsteps+1)//2)
    pp=pct
    tprofile = list(zip([math.ceil(x) for x in B_ft], [math.ceil(y) for y in B_bt]))
    score_list = []
    sgg.generatesm_graph(pfile = "./scratch/scratchtest_graph.sm",s= nsteps, b=nmbatch, a=subtasksfb,
                    TProfile=tprofile, RProfile = rprofile, UnitR = UnitR,
                    gatheringTprofile= T_gathering,gatheringRprofile= grprofile,
                      aa = subtasksg, percent = pp, sd_list = sd_list)

    #call RCPSP solver:
    model, result = cj.RCPSP_solver("./scratch/scratchtest_graph.sm")
    #print(result)
    #cj.RCPSP_plot(model, result)
    if result.objective>0:
        enable = True
        score = vpg.pip_ploting_graph(num_stages = nsteps, num_microbatches = nmbatch,
                    forward_times = B_ft,
                    backward_times = B_bt,
                    gathering_times = T_gathering,
                    steps_dlist = sd_list,
                    enableshare = False, enablegraph = enable,
                    storage = f"./scratch/plot{ratio1:.1f},{ratio2:.1f},{ratio3:.1f}_d2d.png",
                    group_plan = candidate,
                    percentage = pp
                    ) 
        score_list.append(score)
        score = vpg.pip_ploting_graph(num_stages = nsteps, num_microbatches = nmbatch,
                    forward_times = B_ft,
                    backward_times = B_bt,
                    gathering_times = T_gathering, 
                    steps_dlist = sd_list,
                    enableshare = True, enablegraph = enable,
                    storage = f"./scratch/plot{ratio1:.1f},{ratio2:.1f},{ratio3:.1f}_shared.png",
                    group_plan = candidate,
                    percentage = pp)   
        score_list.append(score)
        score = vpg.pip_ploting_graph_real(result.best.tasks,
                           ns=nsteps, a=subtasksfb, b=nmbatch, aa=subtasksg, 
                           forward_times = B_ft,
                           backward_times = B_bt,
                           gathering_times = T_gathering,
                           steps_dlist = sd_list,
                           RProfile = rprofile, UnitR = UnitR,
                           gatheringRprofile= grprofile,
                           enablegraph = enable,
                           storage = f"./scratch/plot{ratio1:.1f},{ratio2:.1f},{ratio3:.1f}_OPTreal.png",
                           group_plan = candidate,
                           percentage = pp)
        score_list.append(score)
        #vps.pip_ploting_direct(result.best.tasks,
        #                       s=nsteps, a=subtasksfb, b=nmbatch, aa=subtasksg,
        #                       storage = f"./scratch/plot{ratio1:.1f},{ratio2:.1f},{ratio3:.1f}_OPTdirect.png",
        #                       group_plan = candidate) 

    #print(score_dy_list)
    #print(best_score_dy_index)
    return score_list
                        

if __name__ == "__main__":
    scratch_dir = "./scratch"
    clear_scratch_folder(scratch_dir)

    r1= 100
    r2 = 100
    r3 = 100

    number_stage = 3
    unit_time = 100
    UnitR = 250
    back_ratio = 1.2
    gather_ratio = 1
    subtasksfb=8
    subtasksg=8
    nmbatch = 4
#test_stramline(ratio1=0,ratio2=0,ratio3=0,B_ft = [1,3,5], B_bt = [2,4,6], rprofile = [(15,17)],
#T_gathering= [20,0,20], grprofile = [20,0,20],subtasksfb=5, UnitR = 20, subtasksg=6,nmbatch = 8)
    data = []
    data.append(["ratio_of_compute", "ratio_of_comm", "ratio_of_between","d2dTime","actualTime", "OPTTime","ratio"])
    os.makedirs("./record", exist_ok=True)
    with open(f"./record/test_improve_b{nmbatch}_sub{subtasksfb}.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data[0])
        percentage = 0
        gather_ratio = 1.0
        #980,180,600,5752.0,7136.0,5776.0,0.8094170403587444 mbatch = 2
        #780,380,300,7064.0,8296.0,7216.0,0.8698167791706847 mbatch = 4
        fo1 = 780
        fo2 = 380
        comm = 300
        i1 = fo1
        i2 = fo2
        i3 = comm
        r_between = 1.4
        commun_time = r_between*unit_time
        current_compt = unit_time
        current_commu = commun_time
        B_ft = []
        B_bt = []
        rprofile = [(UnitR,UnitR)]*(number_stage-1)
        T_gathering = []
        grprofile = []
        B_ft = [fo1, comm, fo2, comm, 500]
        B_bt = [fo1*back_ratio, comm*back_ratio, fo2*back_ratio, comm*back_ratio, 500*back_ratio]
        T_gathering = [0,0,0,0,0]
        grprofile = [0,0, 0, 0 , 0]
        score_list = test_stramline(ratio1=i1,ratio2=i2,ratio3=i3,
                                    B_ft = B_ft, B_bt = B_bt, rprofile = rprofile,
                                    T_gathering= T_gathering, grprofile = grprofile,
                                    subtasksfb=subtasksfb, UnitR = UnitR, subtasksg=subtasksg,
                                    nmbatch = nmbatch,pct= percentage)
        print([i1, i2, i3,
                      score_list[0],
                      score_list[1],
                      score_list[2],
                      score_list[2]/score_list[1]
                      ])
        writer.writerow([i1, i2, i3,
                      score_list[0],
                      score_list[1],
                      score_list[2],
                      score_list[2]/score_list[1]
                      ])
                
    
    
    print("CSV file written successfully.")