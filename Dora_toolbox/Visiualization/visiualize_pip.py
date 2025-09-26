import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random

def pip_ploting(num_stages = 5, num_microbatches = 10,
                forward_times = [3.0, 6.0, 2.5, 5.5, 1.6],
                backward_times = [3.5, 4.0, 3.0, 3.4,  1.7],
                gathering_times = [9,0,6,0,8], enableshare = False, enablegraph = True,
                storage = "./scratch",
                group_plan = []):
    # ----------------------
    # Parameters
    # ----------------------

    #print(num_stages,num_microbatches,forward_times,backward_times,gathering_times)
    comm_delay = 0.2    

    # ----------------------
    # Schedule initialization
    # ----------------------
    fp_schedule = {}
    bp_schedule = {}
    gathering_schedule = {}
    #step 0: for each stage, generate a intra-dependency list   

    Idependent = [] 

    for s in range(num_stages): 

        ini = []
        if(num_stages-s>num_microbatches): #no enouch microbatches we can use...
            for i in range(num_microbatches):
                ini.append((i,'f'))
            
            for i in range(num_microbatches):
                ini.append((i,'b'))
            Idependent.append(ini)
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
        Idependent.append(ini)  

    #finish construction
    #print(Idependent)  

    def cal_index(s, m, fb):
        if fb == 'f':
            if m<num_stages-s:
                return m
            else:
                return num_stages-s-1+(m-(num_stages-s-1))*2
        if fb == 'b':
            if m<=num_microbatches-(num_stages-s):
                return num_stages-s+(m)*2
            else:
                rest = num_microbatches - 1 - m
                return 2*num_microbatches - 1 -rest 

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
                if m== 0 and s==0: # first microbatches
                    fp_schedule[(s, m)] = (0, forward_times[s])
                    upi = 1
                    continue
                if m == 0 and s!=0: # no intra-dependency
                    if (s-1,m) in fp_schedule:
                        start = fp_schedule[(s-1, m)][1] # end time
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
                        if (s-1,m) in fp_schedule:
                            start = max(start, fp_schedule[(s-1, m)][1])
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
                        
                        if (s+1,m) in bp_schedule:
                            start = max(start, bp_schedule[(s+1, m)][1])
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
    max_time = bp_schedule[(0, num_microbatches-1)][1]
    #debugg:
    #max_time = 0
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


if __name__ == "__main__":
    #test_Profilelor_DPsolver()
    pip_ploting()
    pip_ploting(enableshare = True)