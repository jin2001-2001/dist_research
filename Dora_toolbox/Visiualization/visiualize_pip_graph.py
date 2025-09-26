import sys
sys.path.append("../RCPSP")

#import sample_generation

import classical_jobshop as cj
import sample_generation as sg
import sample_generation_graph as sgg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#import random

def pip_ploting_graph(num_stages = 5, num_microbatches = 10,
                forward_times = [6.6, 2.5, 2.5, 4.0, 5.6],
                backward_times = [3.0, 2.0, 4.0, 4,  5.7],
                gathering_times = [9,0,6,0,8],
                steps_dlist = [1,4,3,4,-1],
                enableshare = False, enablegraph = True,
                storage = "./scratch",
                group_plan = [],percentage = 0.6):
    # ----------------------
    # Parameters
    # ----------------------

    #print(num_stages,num_microbatches,forward_times,backward_times,gathering_times)
    comm_delay = 0.2    
    start_percent = 1-percentage
    # ----------------------
    # Schedule initialization
    # ----------------------
    fp_schedule = {}
    bp_schedule = {}
    gathering_schedule = {}
    #step 0: for each stage, generate a intra-dependency list   

    oIdependent = [] 
    Idependent = []
    for s in range(num_stages): 

        ini = []
        if(num_stages-s>num_microbatches): #no enouch microbatches we can use...
            for i in range(num_microbatches):
                ini.append((i,'f'))
            
            for i in range(num_microbatches):
                ini.append((i,'b'))
            oIdependent.append(ini)
            continue
            
        for i in range(num_stages-s):
            ini.append((i,'f'))
        back = 1
        ini_back = 0
        ini_for = num_stages-s
        for i in range(2*(num_microbatches - (num_stages-s))):
            if back == 1:
                ini.append((ini_back,'b'))
                ini_back+=1
                back = 0
            else:
                ini.append((ini_for,'f'))
                ini_for+=1
                back = 1
        for i in range(num_stages-s):
            ini.append((ini_back,'b'))
            ini_back+=1
        oIdependent.append(ini)  
    
    rank_list = sgg.get_rank(steps_dlist)
    for i in range(num_stages):
        Idependent.append(oIdependent[num_stages - 1 - rank_list[i]])


    #finish construction
    #print(Idependent)  

    def cal_index(s, m, fb):
        for i, v in enumerate(Idependent[s]):
            if (m, fb) == v:
                return i


            
    steps_plist = sgg.steps_precedence_list(steps_dlist)

    #print(steps_plist)

    communication_recorder_end = 0
    while(1):  #brute force update:
        #print(len(fp_schedule)+len(bp_schedule))
        upi = 0 
        candidates = None
        candidaterecord = 0
        current_smallest_s = float("inf")

        for m in range(num_microbatches):
            for s in range(num_stages):
                if (s, m) in fp_schedule:
                    continue
                if m== 0 and steps_plist[s]==[]: # first microbatches without precedence..
                    fp_schedule[(s, m)] = (0, forward_times[s])
                    upi = 1
                    continue
                if m == 0 and s!=0: # no intra-dependency

                    #all precedence are finished...
                    indicator = True
                    for p_item in steps_plist[s]:
                        if (p_item,m) not in fp_schedule:
                            indicator = False
                    if indicator == False: continue
                    start = 0

                    for p_item in steps_plist[s]:
                        if (p_item,m) in fp_schedule:
                            start = max(start,
                                        fp_schedule[(p_item, m)][1]) # end time
                            if s%2 == 1:    # new added, for the percentage advance for communication...
                                start = max(start,
                                    fp_schedule[(p_item, m)][0]+start_percent*(fp_schedule[(p_item, m)][1]-fp_schedule[(p_item, m)][0]))
                            if s%2 == 1 and enableshare == True: ####jin: added shared bandwidth version
                                if start<current_smallest_s:
                                    candidaterecord+=1
                                    current_smallest_s = start
                                    candidates = ('f', s, m)                                
                            else:    
                                fp_schedule[(s, m)] = (start, start+forward_times[s])
                            upi = 1
                        else:
                            continue
                if m>0: #
                    start = 0
                    if s>0: # there is inter-dependency
                        indicator = True
                        #all precedence are finished...
                        for p_item in steps_plist[s]:
                            if (p_item,m) not in fp_schedule:
                                indicator = False
                        if indicator == False: continue

                        for p_item in steps_plist[s]:
                            if (p_item,m) in fp_schedule:
                                k = fp_schedule[(p_item, m)][1]
                                if s%2 == 1:
                                    k = fp_schedule[(p_item, m)][0]+start_percent*(fp_schedule[(p_item, m)][1]-fp_schedule[(p_item, m)][0])

                                start = max(start, k)
                            else:
                                continue
                    #then is intra-dependency
                    #print(s,m, cal_index(s, m, 'f'))
                    #print(Idependent)
                    pref = Idependent[s][cal_index(s, m, 'f') - 1]
                    #print(pref)
                    if pref[1] == 'f' and (s,pref[0]) in fp_schedule:
                        start = max(start, fp_schedule[(s,pref[0])][1])
                    elif pref[1] == 'b' and (s,pref[0]) in bp_schedule:
                        start = max(start, bp_schedule[(s,pref[0])][1])
                    else:
                        continue


                    if s%2 == 1 and enableshare == True: ####jin: added shared bandwidth version
                        if start<current_smallest_s:
                            candidaterecord+=1
                            current_smallest_s = start
                            candidates = ('f', s, m)       
                    else:
                        fp_schedule[(s, m)] = (start, start+forward_times[s])
                    upi = 1
        #search for backward updating...          
        for m in range(num_microbatches):
            for s in range(num_stages-1, -1,-1):
                if (s, m) in bp_schedule:
                    continue
                if m>=0: #
                    start = 0
                    if s<num_stages-1: # there is inter-dependency

                                          
                        if (steps_dlist[s],m) in bp_schedule:
                            k = bp_schedule[(steps_dlist[s], m)][1]
                            if s%2 == 1:
                                k = bp_schedule[(steps_dlist[s], m)][0]+start_percent*(bp_schedule[(steps_dlist[s], m)][1]-bp_schedule[(steps_dlist[s], m)][0])
                            start = max(start, k)
                        else:
                            continue
                    #then is intra-dependency
                    pref = Idependent[s][cal_index(s, m, 'b') - 1]
                    if pref[1] == 'f' and (s,pref[0]) in fp_schedule:
                        start = max(start, fp_schedule[(s,pref[0])][1])
                    elif pref[1] == 'b' and (s,pref[0]) in bp_schedule:
                        start = max(start, bp_schedule[(s,pref[0])][1])
                    else:
                        continue

                    if s%2 == 1 and enableshare == True: ####jin: added shared bandwidth version
                        if start<current_smallest_s:
                            candidaterecord+=1
                            current_smallest_s = start
                            candidates = ('b', s, m)  
                    else:
                        bp_schedule[(s, m)] = (start, start+backward_times[s])
                    upi = 1

        #each time, we only update one communication task...
        for s in range(num_stages): # updatind the gathering shedule times:
            if gathering_times[s] == 0:
                continue
            if s in gathering_schedule:
                continue
            if (s, num_microbatches-1) not in bp_schedule:
                continue
            start = bp_schedule[(s, num_microbatches-1)][1]
            if enableshare == True:
                if start<current_smallest_s:
                    candidaterecord+=1
                    current_smallest_s = start
                    candidates = ('g', s)  
            else:
                gathering_schedule[s] = (start, start+gathering_times[s])
            upi = 1



        if candidates != None:
            start = current_smallest_s
            if start<communication_recorder_end:
                start = communication_recorder_end
            if candidates[0] == 'f':
                lasting = forward_times[candidates[1]]
                fp_schedule[(candidates[1],candidates[2])] = (start, start+lasting)
            elif candidates[0] == 'b':
                lasting = backward_times[candidates[1]]
                bp_schedule[(candidates[1],candidates[2])] = (start, start+lasting)
            elif candidates[0] == 'g':
                lasting = gathering_times[candidates[1]]
                gathering_schedule[candidates[1]] = (start, start+lasting)
            communication_recorder_end = start + lasting    
            #candidaterecord-=1
            upi = 1
        if upi == 0: # no update happened ...
            break   
    
    #print(len(bp_schedule), len(fp_schedule),len(gathering_schedule))

    #important: generate max_time:

    #print(fp_schedule,bp_schedule,gathering_schedule)

    max_time = 0

    max_time = max(bp_schedule[(i, num_microbatches - 1)][1]for i in range(0,num_stages,2))
    for s in range(num_stages): #searching for biggest time...
        if gathering_times[s] == 0:
            continue
        if(gathering_schedule[s][1]>max_time):
            max_time = gathering_schedule[s][1]
    
        
    if enablegraph == False:
        return max_time
    #print(max_time)

    #print(fp_schedule)
    #print(bp_schedule)
    # ----------------------
    # Visualization
    # ----------------------
    stage_y = {s: 3.3 * (num_stages - s - 1) for s in range(num_stages)}
    fig, ax = plt.subplots(figsize=(12, 5)) 

    # Forward Passes (green)
    for (s, m), (start, end) in fp_schedule.items():
        if s%2 == 0:
            ax.add_patch(Rectangle((start, stage_y[s]), end - start, 3, facecolor='limegreen', edgecolor='black', linewidth=1.0))
            ax.text(start + (end - start)/2, stage_y[s] + 1.5, f"F{m}", ha='center', va='center', fontsize=8)
        else:
            ax.add_patch(Rectangle((start, stage_y[s]), end - start, 3, facecolor='lightgreen', edgecolor='black', linewidth=1.0))
            ax.text(start + (end - start)/2, stage_y[s] + 1.5, f"↓{m}↓", ha='center', va='center', fontsize=8)  

    # Backward Passes (blue)
    for (s, m), (start, end) in bp_schedule.items():
        if s%2 == 0:
            ax.add_patch(Rectangle((start, stage_y[s]), end - start, 3, facecolor='skyblue', edgecolor='black', linewidth=1.0))
            ax.text(start + (end - start)/2, stage_y[s] + 1.5, f"B{m}", ha='center', va='center', fontsize=8)
        else: 
            ax.add_patch(Rectangle((start, stage_y[s]), end - start, 3, facecolor='lightblue', edgecolor='black', linewidth=1.0))
            ax.text(start + (end - start)/2, stage_y[s] + 1.5, f"↑{m}↑", ha='center', va='center', fontsize=8)

    # Gathering cluster (blue)
    for s, (start, end) in gathering_schedule.items():
        ax.add_patch(Rectangle((start, stage_y[s]), end - start, 3, facecolor='yellow', edgecolor='black', linewidth=1.0))
        ax.text(start + (end - start)/2, stage_y[s] + 1.5, f"Gather", ha='center', va='center', fontsize=8)
    
    # Axes
    ax.set_ylim(0, 3.3 * num_stages)
    #print(gathering_schedule)
    if(gathering_schedule == {}):
        gathering_schedule = {0:(-1,0)}
    ax.set_xlim(0, max(
        max(et for (_, et) in gathering_schedule.values()),
        max(et for (_, et) in bp_schedule.values())
    ) + 2)  

    

    #print([f"Step{s}\nGroup:{group_plan[s//2]}" if s %2 == 0 else f"Step {s}\nCommunication" for s in range(num_stages)])
    ax.set_yticks([stage_y[s] + 2 for s in range(num_stages)])
    ax.set_yticklabels([f"Step{s}\nDevices:{group_plan[s//2]['device']}\nLayers:{group_plan[s//2]['layer']}" if s %2 == 0 else f"Step {s}\nCommunication" for s in range(num_stages)])
    ax.set_xlabel("Time")
    if enableshare == False:
        ax.set_title("1F1B Interleaved Pipeline Schedule (D2D communication)")
    else:
        ax.set_title("1F1B Interleaved Pipeline Schedule (shared communication)")
    ax.grid(False)
    plt.tight_layout()
    #plt.show()
    plt.savefig(storage)
    return max_time


def pip_ploting_graph_real(RCPSP_tasks, TaskL = None,
                        ns=3, a=4, b=3, aa=5, 
                        forward_times = [1, 3.0, 5],
                        backward_times = [2, 4.0, 6.0],
                        gathering_times = [20,0,20],
                        steps_dlist = [1,4,3,4,-1],
                        RProfile = [(15,17)]*1, UnitR = 7,
                        gatheringRprofile= [20,0,20],
                        enablegraph = True,
                        storage = "./scratch",
                        group_plan = [],
                        percentage = 0.6):
    fp_schedule = {}
    bp_schedule = {}
    gathering_schedule = {}
    start_percent = 1-percentage
    fb_amount = (ns//2+1)*b*2*2+(ns//2)*b*2*a
    total_amount = fb_amount+ns*aa

    #step -1: updating the time latency:
    for i in range(len(forward_times)):
        if i%2 == 1:
            forward_times[i] = forward_times[i]*(RProfile[i//2][i%2])/(a-2)/UnitR
            backward_times[i] = backward_times[i]*(RProfile[i//2][i%2])/(a-2)/UnitR
        if gathering_times[i] !=0:
            gathering_times[i] = gathering_times[i]*gatheringRprofile[i]/(aa-2)/UnitR

    #step 0: for each stage, generate a intra-dependency list   

    Idependent = []
    oIdependent = [] 

    for s in range(ns): 

        ini = []
        if(ns-s>b): #no enouch microbatches we can use...
            for i in range(b):
                ini.append((i,'f'))
            
            for i in range(b):
                ini.append((i,'b'))
            oIdependent.append(ini)
            continue
            
        for i in range(ns-s):
            ini.append((i,'f'))
        back = 1
        ini_back = 0
        ini_for = ns-s
        for i in range(2*(b - (ns-s))):
            if back == 1:
                ini.append((ini_back,'b'))
                ini_back+=1
                back = 0
            else:
                ini.append((ini_for,'f'))
                ini_for+=1
                back = 1
        for i in range(ns-s):
            ini.append((ini_back,'b'))
            ini_back+=1
        oIdependent.append(ini)  

    rank_list = sgg.get_rank(steps_dlist)
    for i in range(ns):
        Idependent.append(oIdependent[ns - 1 - rank_list[i]])
    #finish construction
    #print(Idependent)  

    #cal_index: used for calculate intra-stage dependence index for microbatches...
    def cal_index(s, m, fb):
        for i, v in enumerate(Idependent[s]):
            if (m, fb) == v:
                return i
    steps_plist = sgg.steps_precedence_list(steps_dlist)
    #now, the only thing is that we try to extract out the dependencies of the sub communication task...
    comm_Independent = []

    for i in range(fb_amount): 
        if_compute,if_first_part, nsteps, nmicrobatch, ntasks, if_back = sgg.partitioner_shift(i,s,b,a,aa)
        if if_compute == 0:
            start, end = cj.RCPSP_sol_parser_shift(RCPSP_tasks,s, a,b,
                            nsteps, nmicrobatch, 
                            ntasks,
                            aa,if_gathering = False)
            if start == end: #dummy tasks
                continue
            comm_Independent.append([0,nsteps, nmicrobatch, ntasks,if_back, start])
    for i in range(fb_amount, total_amount):
        nsteps, ntasks = sgg.partitioner_shift(i,s,b,a,aa)
        start, end = cj.RCPSP_sol_parser_shift(RCPSP_tasks,s, a,b,
                                         nsteps, nmicrobatch, 
                                         ntasks,
                                         aa,if_gathering = True)
        #print(i, start, end )
        if start == end: #dummy tasks
            continue
        comm_Independent.append([1,nsteps,-1, ntasks,-1, start])

    comm_Independent.sort(key=lambda x: x[-1])
    #print(comm_Independent)
    ##now, we want to reindex the ntasks value, so that biggest ntask is always the final one to be finished...
    counter_dict = {}
    for s in range(ns):
        for j in range(2*b):
            for ib in range(2):
                counter_dict[(0,s,j,ib)] = 0
    for s in range(ns):
        counter_dict[(1,s)] = 0

    for item in comm_Independent:
        if item[0] == 0:
            counter_dict[(0,item[1],item[2],item[4])]+=1
            item[3] = counter_dict[(0,item[1],item[2],item[4])]
        if item[0] == 1:
            counter_dict[(1,item[1])]+=1
            item[3] = counter_dict[(1,item[1])]            
    #print(comm_Independent)
    max_time = 0
    comm_ava_start = 0
#######code ready for update......        
    while(1):  #brute force update:
        #print(len(fp_schedule)+len(bp_schedule))
        upi = 0 
    
        for m in range(b):
            for s in range(ns):
                ##
                #consider if it is communication:
                comm_available = 0
                if s%2 == 1:
                    if len(comm_Independent) == 0:
                        continue
                    (is_g,cnsteps, cnmicrobatch, cntasks, if_back,_) = comm_Independent[0]
                    if is_g == 0 and cnsteps == s and cnmicrobatch == m and if_back == 0:
                        comm_available = 1
                    else:
                        continue
                ##over
                ##

                if comm_available == 0 and (s, m, -1) in fp_schedule:
                    continue
                if m== 0 and steps_plist[s]==[]: # first microbatches
                    fp_schedule[(s, m, -1)] = (0, forward_times[s])
                    upi = 1
                    continue
                if m == 0 and s!=0: # no intra-dependency

                    #all precedence are finished...
                    indicator = True
                    for p_item in steps_plist[s]:
                        if (p_item,m, -1) not in fp_schedule and (p_item,m, a-2) not in fp_schedule:
                            indicator = False
                    if indicator == False: continue

                    start = 0

                    for p_item in steps_plist[s]:
                        if (p_item,m,-1) in fp_schedule or (p_item,m,a-2) in fp_schedule: # interdepence
                            if s%2 == 1:
                                start = max(start,
                                            fp_schedule[(p_item, m, -1)][0]+start_percent*(fp_schedule[(p_item, m, -1)][1]-fp_schedule[(p_item, m, -1)][0])) # end time
                                ##noticed that communication can start earlier now...
                            else:
                                start = max(start,
                                            fp_schedule[(p_item, m, a-2)][1]) # end time
                            if s%2 == 1 and comm_available == 1: ####meaning it is a commun tasks..
                                if start < comm_ava_start:
                                    start = comm_ava_start
                                fp_schedule[(s, m, cntasks)] = (start, start+forward_times[s])
                                comm_ava_start = start+forward_times[s]
                                del comm_Independent[0]  
                                upi = 1                        
                            elif s%2 == 0:    
                                fp_schedule[(s, m, -1)] = (start, start+forward_times[s])
                                upi = 1

                        else: 
                            continue
                if m>0: #
                    start = 0
                    if s>0: # there is inter-dependency
                        indicator = True
                        #all precedence are finished...
                        for p_item in steps_plist[s]:
                            if (p_item,m, -1) not in fp_schedule and (p_item,m, a-2) not in fp_schedule:
                                indicator = False
                        if indicator == False: continue
                        
                        for p_item in steps_plist[s]:
                            if (p_item,m,-1) in fp_schedule or (p_item,m,a-2) in fp_schedule:
                                if s%2 == 1:
                                    k = fp_schedule[(p_item, m, -1)][0]+start_percent*(fp_schedule[(p_item, m, -1)][1]-fp_schedule[(p_item, m, -1)][0])
                                    start = max(start, k)
                                else:
                                    start = max(start, fp_schedule[(p_item, m, a-2)][1])
                            else:
                                continue
                    #then is intra-dependency


                    pref = Idependent[s][cal_index(s, m, 'f') - 1]
                    #print(pref)
                    if pref[1] == 'f' and (s,pref[0],-1) in fp_schedule:
                        start = max(start, fp_schedule[(s,pref[0], -1)][1])
                    elif pref[1] == 'b' and (s,pref[0],-1) in bp_schedule:
                        start = max(start, bp_schedule[(s,pref[0],-1)][1])
                    else:
                        if comm_available == 0:
                            continue


                    if s%2 == 1 and comm_available == 1: ####
                        if start < comm_ava_start:
                            start = comm_ava_start
                        fp_schedule[(s, m, cntasks)] = (start, start+forward_times[s])
                        comm_ava_start = start+forward_times[s]
                        del comm_Independent[0]   
                        upi = 1    
                    elif s%2 == 0:
                        fp_schedule[(s, m, -1)] = (start, start+forward_times[s])
                        upi = 1
        #search for backward updating...          
        for m in range(b):
            for s in range(ns-1, -1,-1):
                ##
                #consider if it is communication:
                comm_available = 0
                if s%2 == 1:
                    if len(comm_Independent) == 0:
                        continue
                    (is_g,cnsteps, cnmicrobatch, cntasks, if_back, _) = comm_Independent[0]
                    if is_g == 0 and cnsteps == s and cnmicrobatch == m+b and if_back == 1:
                        comm_available = 1
                    else:
                        continue
                ##over
                ##

                if comm_available == 0 and (s, m, -1) in bp_schedule:
                    continue
                if m>=0: #
                    start = 0
                    if s<ns-1: # there is inter-dependency
                        
                        if (steps_dlist[s],m,-1) in bp_schedule or (steps_dlist[s],m,a-2) in bp_schedule:
                            if s%2 == 1:
                                k = bp_schedule[(steps_dlist[s], m, -1)][0]+start_percent*(bp_schedule[(steps_dlist[s], m, -1)][1]-bp_schedule[(steps_dlist[s], m, -1)][0])
                                start = max(start, k)
                            else:
                                start = max(start, bp_schedule[(steps_dlist[s], m, a-2)][1])
                        else:
                            continue
                    #then is intra-dependency
                    pref = Idependent[s][cal_index(s, m, 'b') - 1]
                    if pref[1] == 'f' and (s,pref[0],-1) in fp_schedule:
                        start = max(start, fp_schedule[(s,pref[0], -1)][1])
                    elif pref[1] == 'b' and (s,pref[0],-1) in bp_schedule:
                        start = max(start, bp_schedule[(s,pref[0], -1)][1])
                    else:
                        if comm_available == 0:
                            continue

                    if s%2 == 1 and comm_available == 1: 
                        if start < comm_ava_start:
                            start = comm_ava_start
                        bp_schedule[(s, m, cntasks)] = (start, start+backward_times[s])
                        comm_ava_start = start+backward_times[s]
                        del comm_Independent[0]
                        upi = 1   
                    elif s%2 == 0:
                        bp_schedule[(s, m, -1)] = (start, start+backward_times[s])
                        upi = 1

        #each time, we only update one communication task...
        for s in range(ns): # updatind the gathering shedule times:
            comm_available = 0
            if gathering_times[s] == 0:
                continue
            if s%2 == 0: ##only computation has gathering...
                if len(comm_Independent) == 0:
                    continue
                (is_g,cnsteps,_, cntasks,_, _) = comm_Independent[0]
                if is_g == 1 and cnsteps == s:
                    comm_available = 1
                else:
                    continue

            if (s, b-1, -1) not in bp_schedule:
                continue
            start = bp_schedule[(s, b-1, -1)][1]
            if comm_available == 1:
                if start<comm_ava_start:
                    start = comm_ava_start
                gathering_schedule[(s, cntasks)] = (start, start+gathering_times[s])
                comm_ava_start = start+gathering_times[s]
                if len(comm_Independent) == 1:
                    max_time = comm_ava_start
                del comm_Independent[0]
                upi = 1

        if upi == 0: # no update happened ...
            break   
    
    #print(len(bp_schedule), len(fp_schedule),len(gathering_schedule))
    #print(bp_schedule)
    #print(comm_Independent)
    max_time1 = max(bp_schedule[(i, b - 1, -1)][1]for i in range(0,ns,2))
    max_time = max(max_time1, max_time)

    #print(max_time)
    if enablegraph == False:
        return max_time

    #print(fp_schedule)
    #print(bp_schedule)
    #print(gathering_schedule)
    # ----------------------
    # Visualization
    # ----------------------
    s = ns
    stage_y = {ss: 3.3 * (s - ss - 1) for ss in range(s)}
    fig, ax = plt.subplots(figsize=(12, 5)) 

    # Forward Passes (green)
    for (ss, m, subidx), (start, end) in fp_schedule.items():
        #print(s,m)

        interval = 3/(a-2)
        ys = stage_y[ss]
        yl = 3
        yll= 1.5
        if subidx != -1:
            ys = stage_y[ss] + (subidx-1)*interval 
            yl = interval
            yll = interval/2


        if ss%2 == 0:
            ax.add_patch(Rectangle((start, ys), end - start, yl, facecolor='limegreen', edgecolor='black', linewidth=1.0))
            ax.text(start + (end - start)/2, ys + yll, f"F{m}", ha='center', va='center', fontsize=8)
        else:
            ax.add_patch(Rectangle((start, ys), end - start, yl, facecolor='lightgreen', edgecolor='black', linewidth=1.0))
            ax.text(start + (end - start)/2, ys + yll, f"↓{m}↓", ha='center', va='center', fontsize=8)  

    # Backward Passes (blue)
    for (ss, m, subidx), (start, end) in bp_schedule.items():

        interval = 3/(a-2)
        ys = stage_y[ss]
        yl = 3
        yll= 1.5
        if subidx != -1:
            ys = stage_y[ss] + (subidx-1)*interval 
            yl = interval
            yll = interval/2

        if ss%2 == 0:
            ax.add_patch(Rectangle((start, ys), end - start, yl, facecolor='skyblue', edgecolor='black', linewidth=1.0))
            ax.text(start + (end - start)/2, ys + yll, f"B{m}", ha='center', va='center', fontsize=8)
        else: 
            ax.add_patch(Rectangle((start, ys), end - start, yl, facecolor='lightblue', edgecolor='black', linewidth=1.0))
            ax.text(start + (end - start)/2, ys + yll, f"↑{m}↑", ha='center', va='center', fontsize=8)

    # Gathering cluster (blue)
    for (ss,subidx), (start, end) in gathering_schedule.items():

        interval = 3/(aa-2)
        ys = stage_y[ss]
        yl = 3
        yll= 1.5
        if subidx != -1:
            ys = stage_y[ss] + (subidx-1)*interval 
            yl = interval
            yll = interval/2

        ax.add_patch(Rectangle((start, ys), end - start, yl, facecolor='yellow', edgecolor='black', linewidth=1.0))
        ax.text(start + (end - start)/2, ys + yll, f"gather", ha='center', va='center', fontsize=8)
    
    # Axes
    ax.set_ylim(0, 3.3 * s)

    if(gathering_schedule == {}):
        gathering_schedule = {0:(-1,0)}
        
    ax.set_xlim(0, max(
        max(et for (_, et) in gathering_schedule.values()),
        max(et for (_, et) in bp_schedule.values())
    ) + 2)  

    ax.set_yticks([stage_y[ss] + 2.0 for ss in range(s)])
    ax.set_yticklabels([f"Step{ss}\nDevices:{group_plan[ss//2]['device']}\nLayers:{group_plan[ss//2]['layer']}" if ss %2 == 0 else f"Step {ss}\nCommunication" for ss in range(s)])
    ax.set_xlabel("Time")
    ax.set_title("1F1B Interleaved Pipeline Schedule(shared bandwidth with stage 2 OPT Real)")
    ax.grid(False)
    plt.tight_layout()
    #plt.show()
    plt.savefig(storage)
    return max_time



if __name__ == "__main__":
    #test_Profilelor_DPsolver()
    m, r = cj.RCPSP_solver(pfile ="../RCPSP/scratch_graph.sm")
    #cj.RCPSP_plot(m, r)
    RCPSP_tasks = r.best.tasks
    #pip_ploting_graph(group_plan = [{'device':[0], 'layer':[0]}]*5,enableshare=True)
    pip_ploting_graph_real(RCPSP_tasks,
                        ns=5, a=3, b=3, aa=5, 
                        forward_times = [100, 300, 500,700,900],
                        backward_times = [200, 400,600,800,1000],
                        gathering_times = [1000,0,1000,0,0],
                        steps_dlist=[1,4,3,4,-1],
                        RProfile=[(80,80)]*2,
                        UnitR=80,
                        gatheringRprofile=[80,0,80,0,0],
                        enablegraph = True,
                        storage = "./scratch",
                        group_plan = [{'device':[0], 'layer':[0]}]*3)