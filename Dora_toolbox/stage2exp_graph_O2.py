import sys
import math
import os
import shutil
import csv
import numpy as np
sys.path.append("./RCPSP")
sys.path.append("./Simulator")
sys.path.append("./Visiualization")
import test_real as tr
import sample_generation_graph as sgg
import classical_jobshop as cj
import visiualize_pip_graph as vpg
from profile_real import Device, Profilelor
from DPsolver import dynamic_programming_planning
from collections import Counter
import utils

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


def test_stramline(ratio1=0,ratio2=0,ratio3=0,
                   B_ft = [1,3,5], B_bt = [2,4,6], 
                   rprofile = [(15,17)],T_gathering= [20,0,20], 
                   grprofile = [20,0,20],subtasksfb=5, 
                   UnitR = 20, subtasksg=6,
                   nmbatch = 8,pct = 0.0,
                   sd_list = [1,4,3,4,-1]):
    

    ##we try to get a good ratio...
    smallest = min(B_ft)
    ratio = int((10+smallest-1)/smallest)
    ratio = min(ratio, 1)
    ##we get the ratio, 


    nsteps = len(B_ft)
    candidate = [{'device':[0],'layer':[0]}]*((nsteps+1)//2)
    pp=pct
    tprofile = list(zip([math.ceil(x*ratio) for x in B_ft], [math.ceil(y*ratio) for y in B_bt]))
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

    ks = 5 #top ks plans: over all possible permutation 
    ss = 1 # DP cell room...
    ndevice = 4
    nmbatch = 20
    mbatchsize = 5
    layers = 28
    test_list = ["CPU100"]*4 + ["cpu60"]*0+["cpu30"]*0
    score_list = []
    plan_list = []
    allo_list = []
    device_order_list = []
    topK = utils.TopKContainer(ks)

    for device_order in tr.unique_permutations(test_list):
        simprofile, band = tr.generate_profiler_samples_nolimit(
            n = ndevice,
            type_list = device_order,  
            MbatchSize=mbatchsize,
            profilehome="../Profile",  #where you store corresponding profile
            band = 100)   #bandwidth in mbps
        #print("Communication" ,simprofile.communication_solver(10))
        #print("computation:", simprofile.DList[0].computeprofile.batchFuncforward(5), simprofile.DList[0].computeprofile.batchFuncbackward(5))

        result = dynamic_programming_planning(L = layers, N= ndevice , M = nmbatch, k = ks, s = ss,
                                          Profilelor = simprofile, 
                                          alpha = 1, SLO = 0)
        topK.merge_with_device_type(result, device_order)
        

    for j in range(len(topK.scores)):
        score_list.append(topK.scores[j])
        plan_list.append(topK.plans[j])
        allo_list.append(topK.allocateplans[j])
        device_order_list.append(topK.device_orders[j])

    score_list_after = []

    for j in range(len(score_list)):
        B_ft, B_bt, B_fe, B_be, T_gathering, E_gathering, BatchAllocateList = simprofile.getall(plan_list[j])

        ##Actually, the UnitR's value doesn't matter...

        UnitR = 10000

        subtasksfb = mbatchsize+2    
        subtasksg = 1+2
        percentage = 0

        rprofile = [(UnitR, UnitR) for i in range(int((len(B_ft)-1)/2))]
        grprofile = [UnitR if x != 0 else 0 for x in T_gathering]
        sd_list = [i+1 if i < len(B_ft)-1 else -1 for i in range(len(B_ft))]
        #print(sd_list)

        score_compare = test_stramline(ratio1=j,ratio2=0,ratio3=0,
                                    B_ft = B_ft, B_bt = B_bt, rprofile = rprofile,
                                    T_gathering= T_gathering, grprofile = grprofile,
                                    subtasksfb=subtasksfb, UnitR = UnitR, subtasksg=subtasksg,
                                    nmbatch = nmbatch,pct= percentage,
                                    sd_list=sd_list)
        score_list_after.append(score_compare)

    print(score_list_after)
                
    
