import sys
import math
import os
import shutil
import csv
import numpy as np
import itertools
sys.path.append("./RCPSP")
sys.path.append("./Simulator")
sys.path.append("./Visiualization")
import test_real as tr
import sample_generation_graph as sgg
import classical_jobshop as cj
import visiualize_pip_graph as vpg
from profile_real import Device, Profilelor, GraphProfilelor
from DPsolver import dynamic_programming_planning, dynamic_programming_planning_MM
from collections import Counter
import utils
import time
import random


class Bandwidth_str:

    def __init__(self, type, shardband):
        self.type = type
        self.shardband = shardband
        self.LAN_resource = []
        self.LAN_link = {}

    def resource_ratio_list(self):
        L = self.LAN_resource.copy()

        for i in range(len(L)):
            L[i] = L[i]/self.shardband
        return L

    

    def give_cost_option(self, index_from, index_to):
        for key, value in self.LAN_link.items():
            if (index_from,index_to) == key or (index_to,index_from) == key:
                return value
    
    def path_band(self,tlist):
        min = float('inf')
        #print(tlist)
        for rindex in tlist:
            if self.LAN_resource[rindex]<min:
                min = self.LAN_resource[rindex]
        return min


    def available_bw(self, index_from, index_to):
        best_length = 999
        best_band = 0
        if sum(self.LAN_resource) == 0: # no LAN setting
            return self.shardband, []

        best_path = []
        for key, value in self.LAN_link.items():
            if (index_from,index_to) == key or (index_to,index_from) == key:
                for per_path in value:
                    if len(per_path)<best_length:
                        best_band = self.path_band(per_path)
                        best_length = len(per_path)
                        best_path = per_path
                    if len(per_path) == best_length:
                        buffer= self.path_band(per_path)
                        if best_band < buffer:
                            best_band = buffer
                            best_path = per_path

        return self.shardband+best_band, best_path
    




    def available_group_lan_bw(self, index_list):
        lowest_band = float('inf')
        ss= set()
        for key, value in self.LAN_link.items():
            if key[0] in index_list and key[1] in index_list:
                ss.add(key[0])
                ss.add(key[1])
                for per_path in value:
                    b = self.path_band(per_path)
                    if b < lowest_band:
                        lowest_band = b
        if len(ss) == len(index_list):
            return lowest_band
        else:
            return 0
    
    

    




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


def simplified_FIFO_modify(B_ft,B_bt,sd_list,plan_list,  band_str, BatchAllocateList):
    bf = B_ft.copy()
    bb = B_bt.copy()
    for List in (bf, bb):
        for idx, val in enumerate(List):
            if idx %2 == 1:
                to_index = sd_list[idx]
                to_device_tuple = plan_list[to_index//2]['device']
                from_device_tuple = plan_list[idx//2]['device']

                lan_bw, _ = band_str.available_bw(from_device_tuple[0],to_device_tuple[0])
                sbw = band_str.shardband
                r = sbw/(lan_bw)
                #print("r", r)
                List[idx] *= r
    #print(B_ft, bf)
    return bf, bb



def test_stramline(ratio1=0,ratio2=0,ratio3=0,
                   B_ft = [1,3,5], B_bt = [2,4,6], 
                   rprofile = [(15,17)],T_gathering= [20,0,20], 
                   grprofile = [20,0,20],subtasksfb=5, 
                   UnitR = 20, subtasksg=6,
                   nmbatch = 8,pct = 0.0,
                   sd_list = [1,4,3,4,-1],
                   plan_list = None,
                   BatchAllocateList = None,
                   band_str = None,
                   jmode = "training",
                   simprofile = None,
                   cmode = None,
                   Tlimit = 10):
    

    ##we try to get a good ratio...
    smallest = min(B_ft)
    ratio = int((1000+smallest-1)/smallest)
    ratio = max(ratio, 1)
    #print("time span ratio we use:", ratio)
    ##we get the ratio, 


    nsteps = len(B_ft)
    candidate = [{'device':[0],'layer':[0]}]*((nsteps+1)//2)
    pp=pct
    tprofile = list(zip([math.ceil(x*ratio) for x in B_ft], [math.ceil(y*ratio) for y in B_bt]))
    gathering_amp = [math.ceil(x*ratio) for x in T_gathering]
    score_list = []

    sgg.generatesm_graph(pfile = "./scratch/scratchtest_graph.mm",s= nsteps, b=nmbatch, a=subtasksfb,
                    TProfile=tprofile, RProfile = rprofile, UnitR = UnitR,
                    gatheringTprofile= gathering_amp,gatheringRprofile= grprofile,
                      aa = subtasksg, percent = pp, sd_list = sd_list,
                    plan_list = plan_list,
                    BatchAllocateList = BatchAllocateList,
                    band_str = band_str,
                    jmode = jmode  )


    #call RCPSP solver:
    start_T = time.time()
    model, result = cj.RCPSP_solver("./scratch/scratchtest_graph.mm", t = Tlimit)
    #cj.RCPSP_plot(model, result)
    time_cost = time.time()-start_T
    #print("successfully get RCPSP results...")
    return time_cost, result.objective, result.lower_bound
    #print(result)
    #cj.RCPSP_plot(model, result)
    if result.objective>0:
        enable = True
        B_ft, B_bt, B_fe, B_be, T_gathering, E_gathering, BatchAllocateList = simprofile.getall(plan_list, mode= "not shared...")
        score = vpg.pip_ploting_graph(num_stages = nsteps, num_microbatches = nmbatch,
                    forward_times = B_ft,
                    backward_times = B_bt,
                    gathering_times = T_gathering,
                    steps_dlist = sd_list,
                    enableshare = False, enablegraph = enable,
                    storage = f"./scratch/plot{ratio1:.1f},{ratio2:.1f},{ratio3:.1f}_d2d.png",
                    group_plan = candidate,
                    percentage = pp,
                    trivial_mode = "fifo",
                    jmode = jmode
                    ) 
        score_list.append(score)

        
        B_ft, B_bt, B_fe, B_be, T_gathering, E_gathering, BatchAllocateList = simprofile.getall(plan_list)

        score = vpg.pip_ploting_graph(num_stages = nsteps, num_microbatches = nmbatch,
                    forward_times = B_ft,
                    backward_times = B_bt,
                    gathering_times = T_gathering,
                    steps_dlist = sd_list,
                    enableshare = True, enablegraph = enable,
                    storage = f"./scratch/plot{ratio1:.1f},{ratio2:.1f},{ratio3:.1f}_shared_average.png",
                    group_plan = candidate,
                    percentage = pp,
                    trivial_mode = "average",
                    plan_list = plan_list,
                    BatchAllocateList = BatchAllocateList,
                    band_str = band_str,
                    jmode = jmode
                    ) 
        score_list.append(score)


        bftt, bbtt = simplified_FIFO_modify(B_ft,B_bt,sd_list,plan_list,  band_str, BatchAllocateList)

        #score = vpg.pip_ploting_graph(num_stages = nsteps, num_microbatches = nmbatch,
        #            forward_times = bftt,
        #            backward_times = bbtt,
        #            gathering_times = T_gathering, 
        #            steps_dlist = sd_list,
        #            enableshare = True, enablegraph = enable,
        #            storage = f"./scratch/plot{ratio1:.1f},{ratio2:.1f},{ratio3:.1f}_shared_FIFO.png",
        #            group_plan = candidate,
        #            percentage = pp,
        #            trivial_mode = "fifo")   
        #score_list.append(score)



        #score = vpg.pip_ploting_graph(num_stages = nsteps, num_microbatches = nmbatch,
        #            forward_times = B_ft,
        #            backward_times = B_bt,
        #            gathering_times = T_gathering, 
        #            steps_dlist = sd_list,
        #            enableshare = True, enablegraph = enable,
        #            storage = f"./scratch/plot{ratio1:.1f},{ratio2:.1f},{ratio3:.1f}_shared_even.png",
        #            group_plan = candidate,
        #            percentage = pp,
        #            trivial_mode = "even")   
        #score_list.append(score)
        if result.best.tasks!=[]:
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
                           percentage = pp,
                           ratio =ratio,
                           jmode = jmode)
        else:
            print("error, no enough time for RCPSP to get a good result...")
            score = -1
        score_list.append(score)
        #vps.pip_ploting_direct(result.best.tasks,
        #                       s=nsteps, a=subtasksfb, b=nmbatch, aa=subtasksg,
        #                       storage = f"./scratch/plot{ratio1:.1f},{ratio2:.1f},{ratio3:.1f}_OPTdirect.png",
        #                       group_plan = candidate) 

    #print(score_dy_list)
    #print(best_score_dy_index)
    return score_list, time_cost

def structure_generator_for_DPsolver(M, L):

    ##examples...
    #L = [10, 20, 30]
    #M = {(0,0): 0, (0,1): 1, (1,0): 2}

    # find how many layers are needed
    max_i = max(i for (i, j) in M.keys())

    # initialize 2D list with empty lists
    result = [[] for _ in range(max_i + 1)]
    counter = [0 for _ in range(max_i + 1)]

    # fill in according to mapping
    for (i, j), idx in M.items():
        counter[i]+=1
        # make sure inner list is large enough
        while len(result[i]) <= j:
            result[i].append(None)
        result[i][j] = L[idx]         
    return counter, result    


def dependency_generator_for_drawing(plan):  ## the same one in profile_real,,, if we need to do modification, we need change  one as well

    L = []
    biggest_phase = 0
    for shard in plan:
        index = shard['phase']
        L.append(index)
        if shard !=plan[-1]:
            L.append(index)
        if index[0]>=biggest_phase:
            biggest_phase = index[0]
    #print(L)
    #L = [(0,0), (0,0), (0,1), (0,1), (1,0), (1,0)]
    n = len(L)
    deps = [-1] * n  # initialize all with -1
    # 1Rule 1: identical neighbors
    for i in range(n - 1):
        if L[i] == L[i + 1]:
            deps[i] = i + 1
    if  biggest_phase>0:
        # Rule 2: last (0,*) → first (1,0)
        first_1_idx = next(i for i, t in enumerate(L) if t == (1,0))
        # find last index of each unique (0, y)
        seen = set()
        for i in range(n - 1, -1, -1):
            if L[i][0] == 0 and L[i][1] not in seen:
                deps[i] = first_1_idx
                seen.add(L[i][1])
    if biggest_phase>1:
        for j in range(biggest_phase-1):
            lower = 1+j
            higher = lower +1
            first_higher_idx= next(i for i, t in enumerate(L) if t == (higher,0))
            deps[first_higher_idx-1] = first_higher_idx
            

    return(deps)




def pareto_min_min(x, y):
    """Return indices of Pareto-optimal points for minimizing both x and y."""
    points = list(zip(x, y))
    pareto_idx = []

    for i, (xi, yi) in enumerate(points):
        dominated = False
        for j, (xj, yj) in enumerate(points):
            # j dominates i if j is <= on both axes and strictly < on one axis
            if (xj <= xi and yj <= yi) and (xj < xi or yj < yi):
                dominated = True
                break
        if not dominated:
            pareto_idx.append(i)

    return pareto_idx


def sample_pareto_points(x, y, max_points=5):
    idx = pareto_min_min(x, y)       # Pareto indices
    idx = list(idx)

    # Case 1: Too many Pareto points → downsample
    if len(idx) > max_points:
        idx = random.sample(idx, max_points)

    # Case 2: Too few Pareto points → randomly add more indices
    elif len(idx) < max_points:
        all_idx = list(range(len(x)))
        remaining = [i for i in all_idx if i not in idx]

        needed = max_points - len(idx)

        if len(remaining) >= needed:
            idx += random.sample(remaining, needed)
        else:
            # If there are not enough remaining points, take all
            idx += remaining

    # Now idx has ≤ max_points, ensure unique
    idx = idx[:max_points]

    # Return in x,y,index format
    return [x[i] for i in idx], [y[i] for i in idx], idx


def dora_best_MM(
                model_maping, #dict
                model_names,
                ndevice,
                nmbatch,
                mbatchsize,
                hidden_size, #list
                seq,         #list ## should calculate by oursevlves:: For different encoder and the backbone...
                layers,      #list
                band_str,
                profilehome,
                test_list = [],
                util_list = [],
                mem_list = [],
                ks=10, ss = 1,
                alpha = 0,
                SLO = 0,
                jmode = "training",
                cmode = "shared"):
    scratch_dir = "./scratch"
    clear_scratch_folder(scratch_dir)

    print("begin MM model algorithm...")
    start1 = 0
    score_list = []
    plan_list = []
    allo_list = []
    device_order_list = []

    topK = utils.TopKContainer(ks)

    Permuaccounts = 0

    perms_ = list(itertools.permutations(range(len(test_list))))
    random.shuffle(perms_)   # now order is randomized
    ##currently, we don't consider exploring possible shuffled orders...
    perms = [[i for i in reversed(range(ndevice))],[i for i in range(ndevice)],perms_[0] ,perms_[1]]

    Tpart1 = 0
    for perm_indices in perms[:]:
        ##adjust
        if Permuaccounts>8:
            continue
        device_order = [test_list[i] for i in perm_indices]
        mem_order    = [mem_list[i] for i in perm_indices]
        util_order = [util_list[i] for i in perm_indices]
        print("generate profiler:")
        simprofile, _ = tr.generate_profiler_samples_nolimit_MM(
            model_names = model_names,
            n = ndevice,
            hidden_size = hidden_size,
            seq = seq, 
            layers = layers,
            type_list = device_order,  
            MbatchSize=mbatchsize,
            profilehome=profilehome,
            band_str = band_str,
            mem_list = mem_order,
            util_list = util_order,
            map_dict = model_maping,
            jmode = jmode)
        #print("Communication" ,simprofile.communication_solver(10))
        #print("computation:", simprofile.DList[0].computeprofile.batchFuncforward(5), simprofile.DList[0].computeprofile.batchFuncbackward(5))

        Structure,layer_Structure = structure_generator_for_DPsolver(model_maping, layers)
        
        print("begin solving...")
        start = time.time()
        result = dynamic_programming_planning_MM(Structure=Structure,Layer_structure = layer_Structure, N= ndevice , M = nmbatch, k = ks, s = ss,
                                          Profilelor = simprofile, 
                                          alpha = alpha, SLO = SLO, jmode = jmode)
        tcost = time.time() - start
        Tpart1+=tcost
        #print("the iteraion's result is ", result.amount())
        topK.merge_with_device_type(result, perm_indices)  # notice here that we use it to store perm_indices, not actual device name anymore...
        Permuaccounts+=1

    for j in range(len(topK.scores)):
        score_list.append(topK.scores[j])
        plan_list.append(topK.plans[j])
        allo_list.append(topK.allocateplans[j])
        device_order_list.append(topK.device_orders[j])  
    
    #print("score_list:", score_list)

    score_list_after = []
    final_score = []
    print(score_list)
    print(plan_list)
    print(allo_list)
    print(device_order_list)


    Tpart2 = 0

    ###for test...###
    print("Drawing begins...")


    buf = utils.graph_plan_estimator(0 ,plan_list[0], nmbatch, 0, simprofile, alpha)
    #print("tested value:::",buf)
    Energy_list = []
    for j in range(len(score_list)):
        #noticed: we should regenerate the simprofile...
        perm_indices = device_order_list[j]
        device_order = [test_list[i] for i in perm_indices]
        mem_order    = [mem_list[i] for i in perm_indices]
        util_order = [util_list[i] for i in perm_indices]
        simprofile, _= tr.generate_profiler_samples_nolimit_MM(
            model_names = model_names,
            n = ndevice,
            hidden_size = hidden_size,
            seq = seq, 
            layers = layers,
            type_list = device_order,  
            MbatchSize=mbatchsize,
            profilehome=profilehome,
            band_str = band_str,
            mem_list = mem_order,
            util_list = util_order,
            map_dict = model_maping,
            jmode = jmode)

        B_ft, B_bt, B_fe, B_be, T_gathering, E_gathering, BatchAllocateList = simprofile.getall(plan_list[j])

        if jmode == "training":
            #print(B_fe)
            E_consumption = sum(
            nmbatch * (B_fe[i] + B_be[i]) + E_gathering[i]
            for i in range(len(B_fe))
        )
        else:
            E_consumption = sum(
            nmbatch * (B_fe[i]) 
            for i in range(len(B_fe))
        )

        print("Energy cost:::", E_consumption)
        Energy_list.append(E_consumption)
        #print(B_ft, B_bt,T_gathering)
        res = utils.graph_plan_estimator(0 ,plan_list[j], nmbatch, 0, simprofile, alpha, jmode)
        ##Actually, the UnitR's value doesn't matter...

        UnitR = 10000

        bestam = mbatchsize*2 if mbatchsize<=6 else mbatchsize

        subtasksfb = bestam+2    
        subtasksg = bestam*4+2
        percentage = 0

        rprofile = [(UnitR, UnitR) for i in range(int((len(B_ft)-1)/2))]
        grprofile = [UnitR if x != 0 else 0 for x in T_gathering]
        #sd_list = [i+1 if i < len(B_ft)-1 else -1 for i in range(len(B_ft))]
        sd_list = dependency_generator_for_drawing(plan_list[j])
        
        #print(B_ft, B_bt)
        #print(sd_list)
        #print(plan_list[j])

        score_compare, timecost = test_stramline(ratio1=j,ratio2=0,ratio3=0,
                                    B_ft = B_ft, B_bt = B_bt, rprofile = rprofile,
                                    T_gathering= T_gathering, grprofile = grprofile,
                                    subtasksfb=subtasksfb, UnitR = UnitR, subtasksg=subtasksg,
                                    nmbatch = nmbatch,pct= percentage,
                                    sd_list=sd_list,
                                    plan_list = plan_list[j],
                                    BatchAllocateList = BatchAllocateList,
                                    band_str = band_str,
                                    jmode = jmode,
                                    simprofile = simprofile,
                                    cmode = cmode
                                    )
        Tpart2+=timecost
        score_list_after.append(score_compare)
        if score_compare[2]!=-1:
            final_score.append(score_compare[2])
        else:
            final_score.append(score_compare[1])

    print(score_list_after)
    print("energy_list:", Energy_list)
    print(f"time cost: T1:{Tpart1}, T2:{Tpart2}. Ttotal{Tpart1+Tpart2}")

    final_x, final_y, _ = sample_pareto_points(final_score, Energy_list, max_points=5)
    return final_x, final_y






def simulator_eval(  
                model_maping, 
                model_names,
                ndevice,
                nmbatch,
                mbatchsize,
                hidden_size,
                seq,
                layers,
                band_str,
                profilehome,
                test_plan,
                test_list = [],
                util_dlist= [],
                mem_list = []
                ,ks=10, ss = 1,
                alpha = 0,
                jmode = None,
                Tlimit = 10):
    scratch_dir = "./scratch"
    clear_scratch_folder(scratch_dir)



    simprofile, band = tr.generate_profiler_samples_nolimit_MM(
            model_names = model_names,
            n = ndevice,
            hidden_size = hidden_size,
            seq = seq, 
            layers = layers,
            type_list = test_list,  
            MbatchSize=mbatchsize,
            profilehome=profilehome,
            band_str = band_str,
            mem_list = mem_list,
            util_list =util_dlist,
            map_dict = model_maping,
            jmode = jmode)
    B_ft, B_bt, B_fe, B_be, T_gathering, E_gathering, BatchAllocateList = simprofile.getall(test_plan)

    if jmode == "training":
        print(B_fe)
        E_consumption = sum(
        nmbatch * (B_fe[i] + B_be[i]) + E_gathering[i]
        for i in range(len(B_fe))
    )
    else:
        print()
        E_consumption = sum(
        nmbatch * (B_fe[i]) 
        for i in range(len(B_fe))
    )

    #print(B_ft, B_bt)
    ##Actually, the UnitR's value doesn't matter...
    UnitR = 10000


    bestam =  mbatchsize
    subtasksfb = bestam*4    +2
    subtasksg = bestam*4 +2


    percentage = 0
    rprofile = [(UnitR, UnitR) for i in range(int((len(B_ft)-1)/2))]
    grprofile = [UnitR if x != 0 else 0 for x in T_gathering]
    sd_list = dependency_generator_for_drawing(test_plan)
    #print(sd_list)
    time_cost, best, lowerbound = test_stramline(ratio1=-1,ratio2=0,ratio3=0,
                                    B_ft = B_ft, B_bt = B_bt, rprofile = rprofile,
                                    T_gathering= T_gathering, grprofile = grprofile,
                                    subtasksfb=subtasksfb, UnitR = UnitR, subtasksg=subtasksg,
                                    nmbatch = nmbatch,pct= percentage,
                                    sd_list=sd_list,
                                    plan_list = test_plan,
                                    BatchAllocateList = BatchAllocateList,
                                    band_str = band_str,
                                    jmode = jmode,
                                    simprofile = simprofile,
                                    Tlimit = Tlimit
                                    )
    return time_cost, best, lowerbound


def construct_band(name, band, lanband):
    if name == "mesh4":
        structure = Bandwidth_str("mesh4", band)
        ch = [0,2,1,3]
        #  0  --- 0 ---- 1
        #  |             |
        #  3             1 
        #  |             |
        #  3  --- 2 ---- 2
        structure.LAN_resource = [lanband]*4
        structure.LAN_link = {(ch[0],ch[1]):[[0],[3,2,1]], (ch[0],ch[2]):[[0,1],[2,3]], (ch[0],ch[3]):[[3],[0,1,2]],
                              (ch[1],ch[2]):[[1],[0,2,3]], (ch[1],ch[3]):[[1,2],[0,3]],
                              (ch[2],ch[3]):[[2],[0,1,3]]      
        }
        return structure
    2
    if name == "mesh5":
        structure = Bandwidth_str("mesh5", band)
        ch = [0,1,2,3,4]
        #   0 -----  1 ---------  2

        #      4----------30
        structure.LAN_resource = [lanband]*5
        structure.LAN_link = {(ch[0],ch[1]):[[0],[4,3,2,1]], (ch[0],ch[2]):[[0,1],[2,3,4]], (ch[0],ch[3]):[[3,4],[0,1,2]], (ch[0],ch[4]):[[4],[0,1,2,3]],
                              (ch[1],ch[2]):[[1],[0,2,3,4]], (ch[1],ch[3]):[[1,2],[0,4,3]], (ch[1],ch[4]):[[1,2,3],[0,4]],
                              (ch[2],ch[3]):[[2],[0,1,3,4]], (ch[2],ch[4]):[[2,3],[0,1,4]],     
                              (ch[3],ch[4]):[[3],[0,1,2,4]]
        }
        return structure
    if name == "mesh3dead":
        structure = Bandwidth_str("mesh5", band)
        ch = [0,1,2]
        #   0 -----  1 ---------  2

        #      4----------3
        structure.LAN_resource = [lanband]*3
        structure.LAN_link = {
        }
        return structure


if __name__ == "__main__":
    ndevice = 4
    nmbatch = 5
    mbatchsize = 4
    choice0= ["home1", "home2", "traffic", "station", "motivation_graph","motivation_3b"]
    choice1= ["bert", "0.6", "1.7", "omni","serial_omni_0", "serial_omni_1","serial_omni_2" ]
    util_dlist = [1]*10
    work_mode = "trainingxx"
    device_setting = choice0[2]
    model_setting = choice1[2]
    #model_setting = "1.7_for_motivation"
    #device_setting = "home1_for_motivation"
    f_able = 0
    v_able = 1
    alpha= 0.0
    SLO  = 0.0
    ##notice line 380 around for the 

    plan1 = [{'phase': (0, 0), 'layer': (0, 32), 'device': (1, 2), 'inver_internal_stage_idx': 0},
            {'phase': (0, 1), 'layer': (0, 32), 'device': (2, 3), 'inver_internal_stage_idx': 0}, 
            {'phase': (1, 0), 'layer': (0, 36), 'device': (3, 4), 'inver_internal_stage_idx': 0}]  # for omni...
    
    plan1 = [{'phase': (0, 0), 'layer': (0, 5), 'device': (0, 1), 'inver_internal_stage_idx': 4},
             {'phase': (0, 0), 'layer': (5, 10), 'device': (1, 2), 'inver_internal_stage_idx': 3},
             {'phase': (0, 0), 'layer': (10, 16), 'device': (2, 3), 'inver_internal_stage_idx': 2},
             {'phase': (0, 0), 'layer': (16, 21), 'device': (3, 4), 'inver_internal_stage_idx': 1},
             {'phase': (0, 0), 'layer': (21, 28), 'device': (4, 5), 'inver_internal_stage_idx': 0}]  #for 5 steps
    

    plan1 = [{'phase': (0, 0), 'layer': (0, 7), 'device': (0, 1), 'inver_internal_stage_idx': 3},
             {'phase': (0, 0), 'layer': (7, 14), 'device': (1, 2), 'inver_internal_stage_idx': 2},
             {'phase': (0, 0), 'layer': (14, 21), 'device': (2, 3), 'inver_internal_stage_idx': 1},
             {'phase': (0, 0), 'layer': (21, 28), 'device': (3, 4), 'inver_internal_stage_idx': 0}]  #for 4 steps

    '''
    plan1 = [
             {'phase': (0, 0), 'layer': (0, 25), 'device': (0, 1), 'inver_internal_stage_idx': 0}, 
             {'phase': (1, 0), 'layer': (25, 50), 'device': (1, 2), 'inver_internal_stage_idx': 0},
             {'phase': (2, 0), 'layer': (50, 75), 'device': (2, 3), 'inver_internal_stage_idx': 0}, 
             {'phase': (3, 0), 'layer': (75, 100), 'device': (3, 4), 'inver_internal_stage_idx': 0}]
    '''
   # plan1 = [{'phase': (0, 0), 'layer': (0,28), 'device': (0,4), 'inver_internal_stage_idx': 1},]
    #         {'phase': (1, 0), 'layer': (47, 100), 'device': (2,4), 'inver_internal_stage_idx': 0}]        
    #plan1 =[{'phase': (0, 0), 'layer': (0, 11), 'device': (2, 3), 'inver_internal_stage_idx': 2}, {'phase': (0, 0), 'layer': (11, 22), 'device': (3, 4), 'inver_internal_stage_idx': 1}, {'phase': (0, 0), 'layer': (22, 28), 'device': (4, 5), 'inver_internal_stage_idx': 0}]
    #plan1 = [{'phase': (0, 0), 'layer': (0, 32), 'device': (0, 1), 'inver_internal_stage_idx': 0}, {'phase': (0, 1), 'layer': (0, 32), 'device': (1, 2), 'inver_internal_stage_idx': 0}, {'phase': (1, 0), 'layer': (0, 20), 'device': (2, 3), 'inver_internal_stage_idx': 1}, {'phase': (1, 0), 'layer': (20, 36), 'device': (3, 4), 'inver_internal_stage_idx': 0}]
    if model_setting == "omni":
        nmbatch = 5
        mbatchsize = 4
        layers = [32,32,36]  ## should be a list 
        hidden_size = [1280,1280,3584] ## should be a list 
        seq = [768, 512, 768+512+256] ## should be a list
        profilehome="../Profile_exp_1.7"
        profilemaping = {(0,0):0, (0,1):1, (1,0):2}   # 0: vision, 1: audio, 2: backbone
        #profilemaping = {(0,0):0}   # 0: vision, 1: audio, 2: backbone
        model_names = [""]*len(profilemaping)
        #model_names = ["vision", "audio", "thinker"]



    if model_setting == "serial_omni_0":
        nmbatch = 5
        mbatchsize = 4
        layers = [100]  ## should be a list 
        hidden_size = [2109*1] ## should be a list 
        seq = [1208*1] ## should be a list       #768+768 = 1536
        profilehome="../Profile_exp_1.7"
        profilemaping = {(0,0):0}   # 0: vision, 1: audio, 2: backbone
        #profilemaping = {(0,0):0}   # 0: vision, 1: audio, 2: backbone
        model_names = [""]*len(profilemaping)
        #model_names = ["vision", "audio", "thinker"]
    if model_setting == "serial_omni_1":
        nmbatch = 5
        mbatchsize = 4
        layers = [20,20,20,20]  ## should be a list 
        hidden_size = [1280,1280, 3584,3584] ## should be a list 
        seq = [768+512+256, 768+512+256, 768+512+256, 768+512+256, 768+512+256] ## should be a list
        profilehome="../Profile_exp_1.7"
        profilemaping = {(0,0):0, (1,0):1, (2,0):2,(3,0):3}   # 0: vision, 1: audio, 2: backbone
        #profilemaping = {(0,0):0}   # 0: vision, 1: audio, 2: backbone
        model_names = [""]*len(profilemaping)
        #model_names = ["vision", "audio", "thinker"]
    if model_setting == "serial_omni_2":
        nmbatch = 5
        mbatchsize = 4
        layers = [32,32,36]  ## should be a list 
        hidden_size = [1280,2816] ## should be a list 
        seq = [ 768+512+256,  768+512+256, 768+512+256] ## should be a list
        profilehome="../Profile_exp_1.7"
        profilemaping = {(0,0):0, (1,0):1}   # 0: vision, 1: audio, 2: backbone
        #profilemaping = {(0,0):0}   # 0: vision, 1: audio, 2: backbone
        model_names = [""]*len(profilemaping)
        #model_names = ["vision", "audio", "thinker"]



    if model_setting == "0.6":
        nmbatch = 10
        mbatchsize = 12
        layers = [28]  ## should be a list 
        hidden_size = [1024] ## fixed to be 2048
        seq = [512] ## should be a list
        profilehome="../Profile_exp_0.6"
        profilemaping = {(0,0):0}   # 0: vision, 1: audio, 2: backbone
        #profilemaping = {(0,0):0}   # 0: vision, 1: audio, 2: backbone
        model_names = [""]*(len(profilemaping))
        #model_names = ["vision", "audio", "thinker"]

    if model_setting == "1.7":
        nmbatch = 9   #9
        mbatchsize = 8  #8
        layers = [28]  ## should be a list 
        hidden_size = [2048] ## should be a list 
        seq = [512] ## should be a list
        profilehome="../Profile_exp_1.7"
        profilemaping = {(0,0):0}   # 0: vision, 1: audio, 2: backbone
        #profilemaping = {(0,0):0}   # 0: vision, 1: audio, 2: backbone
        model_names = [""]*(len(profilemaping))
        #model_names = ["vision", "audio", "thinker"]

    if model_setting == "1.7_for_motivation":
        nmbatch = 5   #9
        mbatchsize = 8  #8
        layers = [28]  ## should be a list 
        hidden_size = [2048] ## should be a list 
        seq = [512] ## should be a list
        profilehome="../Profile_exp_1.7"
        profilemaping = {(0,0):0}   # 0: vision, 1: audio, 2: backbone
        #profilemaping = {(0,0):0}   # 0: vision, 1: audio, 2: backbone
        model_names = [""]*(len(profilemaping))
        #model_names = ["vision", "audio", "thinker"]

    if model_setting == "bert":
        nmbatch = 12
        mbatchsize = 12
        layers = [12]  ## should be a list 
        hidden_size = [768*2] ## should be a list :for bert we consider a full -presion model...
        seq = [1024] ## should be a list
        profilehome="../Profile_exp_bert"
        profilemaping = {(0,0):0}   # 0: vision, 1: audio, 2: backbone
        #profilemaping = {(0,0):0}   # 0: vision, 1: audio, 2: backbone
        model_names = [""]*(len(profilemaping))
        #model_names = ["vision", "audio", "thinker"]

    band = 250
    lanband = 0
    name = "mesh4"
    band_Str = construct_band(name,band,lanband)





    if device_setting == "home1":
        ndevice = 5
        band = 1000
        lanband = 0
        name = "mesh4"
        band_Str = construct_band(name,band,lanband)
        if "omni" in model_setting:
            set_list =  ["4050INT8"]*2+ ["4060INT8"]*3
        else:
            set_list = ["2630"]*0 + ["A40"]*0+ ["V100"]*0  + ["CameraINT8"]*0 + ["Xiaomi"]*0  + ["Samsung"]*0+["4050"]*2+ ["4060"]*3
        util_list = [1]*5
        mem_list = [32*2]*0   + [48*2]*0+    [32*2]*0+     [16*2]*0+            [12*2]*0+  [12*2]*0+   [12*2]*2+    [12*2]*3 #12+24 =36
        mem_list = [x*1024 for x in mem_list]

    if device_setting == "home1_for_motivation":
        ndevice = 3
        band = 500
        lanband = 0
        name = "mesh4"
        band_Str = construct_band(name,band,lanband)
        set_list = ["2630"]*0 + ["A40"]*0+ ["V100"]*0  + ["CameraINT8"]*0 + ["Xiaomi"]*3  + ["Samsung"]*0+["4050"]*0+ ["4060"]*0
        util_list = [0.5,0.18,0.50]
        mem_list = [32*2]*0   + [48*2]*0+    [32*2]*0+     [16*2]*0+            [12*2]*0+  [12*2]*0+   [12*2]*2+    [12*2]*3 #12+24 =36
        mem_list = [x*1024 for x in mem_list]

    if device_setting == "home2":
        ndevice = 5
        band = 600
        lanband = 0
        name = "mesh4"
        band_Str = construct_band(name,band,lanband)
        if "omni" in model_setting:
            #set_list =  ["XiaomiINT8"]*2  + ["SamsungINT8"]*1+ ["4050INT8"]*2
            set_list =  ["XiaomiINT8"]*1  + ["SamsungINT8"]*1+ ["4050INT8"]*1 +["XiaomiINT8"]*1+ ["4050INT8"]*1
        else:
            #set_list = ["2630"]*0 + ["A40"]*0+ ["V100"]*0  + ["CameraINT8"]*0 + ["Xiaomi"]*2  + ["Samsung"]*1+["4060"]*0+ ["4050"]*2
            set_list =  ["Xiaomi"]*1  + ["Samsung"]*1+ ["4050"]*1 + ["Xiaomi"]*1 + ["4050"]*1
        util_list = [0.85,0.7,0.85,0.75, 0.8]
        mem_list = [32*2]*0   + [48*2]*0+    [32*2]*0+     [16*2]*0+            [12*2]*2+  [12*2]*1+   [12*2]*0+    [12*2]*2
        mem_list = [x*1024 for x in mem_list]

    if device_setting == "traffic":
        ndevice = 4
        band = 1
        lanband = 50 ##actual situation is *2(as INT8 model)
        name = "mesh4"
        band_Str = construct_band(name,band,lanband)
        if "omni" in model_setting:
            set_list = ["2630"]*0 + ["A40"]*0+ ["V100"]*0  + ["CameraINT8"]*4 + ["Xiaomi"]*0  + ["Samsung"]*0+["4060"]*0+ ["4050"]*0
        else:
            set_list = ["Camera"]*4
        util_list = [0.55,0.6,0.55,1]
        #util_list = [0.7,0.7,1, 1]
        mem_list = [32*2]*0   + [48*2]*0+    [32*2]*0+     [16*2]*4+            [12*2]*0+  [12*2]*0+   [12*2]*0+    [8*2]*0
        mem_list = [x*1024 for x in mem_list]






    if device_setting == "station":
        ndevice = 4
        band = 40000
        lanband = 0
        name = "mesh4"
        band_Str = construct_band(name,band,lanband)

        if "omni" in model_setting:
            set_list = ["A40INT8"]*2+ ["V100INT8"]*2
        else:
            set_list = ["2630"]*0 + ["A40"]*2+ ["V100"]*2  + ["CameraINT8"]*0 + ["Xiaomi"]*0  + ["Samsung"]*0+["4060"]*0+ ["4050"]*0
        util_list = [1]*5
        mem_list = [32*2]*0   + [48*2]*2+    [32*2]*2+     [16*2]*0+            [12*2]*0+  [12*2]*0+   [12*2]*0+    [8*2]*0
        mem_list = [x*1024 for x in mem_list]

    if device_setting == "motivation_graph":
        ndevice = 5
        band = 500
        lanband = 0 ##actual situation is *2(as INT8 model)
        name = "mesh4"
        band_Str = construct_band(name,band,lanband)
        if model_setting == "omni":
            set_list = ["2630"]*0 + ["A40"]*0+ ["V100"]*0  + ["CameraINT8"]*0 + ["Xiaomi"]*0  + ["Samsung"]*0+["4060"]*0+ ["4050INT8"]*5
        else:
            set_list = ["4050"]*5
        util_list = [1]*5
        mem_list = [32*2]*0   + [48*2]*0+    [32*2]*0+     [16*2]*5+            [12*2]*0+  [12*2]*0+   [12*2]*0+    [8*2]*0
        mem_list = [x*1024 for x in mem_list] 

    if device_setting == "motivation_3b":
        ndevice = 3
        band = 500
        lanband = 0 ##actual situation is *2(as INT8 model)
        name = "mesh3dead"
        band_Str = construct_band(name,band,lanband)
        set_list =  ["Xiaomi"]*3
        util_list = [0.50,0.50,0.50]
        util_dlist = util_list
        mem_list = [16*2]*3
        mem_list = [x*1024 for x in mem_list] 
        plan1 = [{'phase': (0, 0), 'layer': (0, 28), 'device': (2, 3), 'inver_internal_stage_idx': 0}]
    if model_setting == "serial_omni_0":
        mem_list = [x*1.5 for x in mem_list]
    
    test_dlist = set_list
    #test_dlist = ["CPU60"]*4
    mem_tlist = mem_list
    mem_tlist = [x*1024 for x in mem_tlist]
    util_dlist = util_list


    print(set_list)
  
    with open("RCPSP_perform_new.txt", "w") as f:
        f.write("T,mbatch,time_cost,best,lowerbound\n")  # header
        for T in [2,4,8, 12, 16]:
        
            for mbatchsize in range(1,10):
                for repeat in range(10):
                    time_cost, best, lowerbound = simulator_eval(
                    profilemaping, 
                    model_names,
                    ndevice,
                    nmbatch,
                    mbatchsize,
                    hidden_size,
                    seq,
                    layers,
                    band_Str,
                    profilehome,
                    plan1,
                    test_dlist,
                    util_dlist,
                    mem_tlist,ks = 5, ss = 4,
                    alpha=alpha,
                    jmode = work_mode,
                    Tlimit=T
                    )

                    #print(best.__dict__)
                    f.write(f"{T},{mbatchsize},{time_cost},{best},{lowerbound}\n")
                    print(T, mbatchsize, time_cost, best, lowerbound )

