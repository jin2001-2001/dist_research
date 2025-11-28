import sys
sys.path.append("../RCPSP")

#import sample_generation

import classical_jobshop as cj
import sample_generation as sg
import sample_generation_graph as sgg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#import random

class BandwidthStack:
    def __init__(self, plan_list,BatchAllocateList, band_str,steps_dlist):
        self.tasks = []  # store {"id", "start", "end", "remaining", "active"}
        self.time = 0.0
        self.next_id = 0
        self.lazy_stack = []
        self.plan_list = plan_list
        self.BatchAllocateList = BatchAllocateList
        self.band_str = band_str
        self.steps_dlist = steps_dlist
        self.resource_ratio_list = band_str.resource_ratio_list()

        n = len(band_str.LAN_resource)
        self.LAN_count = [0]*n
        self.shared_count = 0


    def init_path_extract(self, tindex,sd_list, plan_list, band_str, BatchAllocateList):


        steps_index = tindex[1]
    #print(steps_index, sd_list, plan_list)
        to_index = sd_list[steps_index]
        to_device_tuple = plan_list[to_index//2]['device']
        from_device_tuple = plan_list[steps_index//2]['device']



        _, path =  band_str.available_bw(from_device_tuple[0], to_device_tuple[0])

        if tindex[0] == 'g':
            path = [i for i in range(len(band_str.LAN_resource))]
            to_device_tuple  = None

        return from_device_tuple, to_device_tuple, path
        

    def _active_tasks(self):
        return [t for t in self.tasks if t["active"] and not t["dead"]]    

    def _update_remaining(self, delta_t):
        """Update all active tasks given time delta_t, assuming equal bandwidth share."""
        active = [t for t in self.tasks if t["active"]]
        n = len(active)
        if n == 0 or delta_t <= 0:
            return

        # Each gets 1/n bandwidth share → progress = delta_t / n
        for t in active:
            t["remaining"] -= delta_t / n
            if t["remaining"] <= 0:
                t["active"] = False
                t["end"] = self.time + delta_t  # record finish time


    def advance_to(self, new_time: float):
       
        """Advance simulated time to new_time, updating all tasks with
        correct event-based bandwidth redistribution."""
        if new_time <= self.time:
            if new_time == self.time:
                #print("happen to concatenate tight...")
                return False
            print("error operations...")
            return None

        remaining_dt = new_time - self.time
        eps = 1e-12  # numerical tolerance

        meet_end = False
        while remaining_dt > eps:
            if meet_end == True:
                return meet_end
            active = self._active_tasks()
            n = len(active)

            if n == 0:
                # No active tasks: just jump time
                self.time += remaining_dt
                remaining_dt = 0.0
                break

            # Time until each active task finishes under current n
            # time_to_finish_i = remaining_i * n
            times_to_finish = [
                self.remain_time(t) for t in active if t["remaining"] > eps
            ]
            #print("advance to time and the remaining time:", new_time,remaining_dt, times_to_finish)
            if  times_to_finish == []:
                # All effectively done, mark and exit
                for t in active:
                    t["active"] = False
                    t["end"] = self.time
                    self.r_counter_update(t, "sub")
                remaining_dt = 0.0
                meet_end = True
                break

            next_finish = min(times_to_finish)

            if next_finish <= remaining_dt + eps:
                # A task (or multiple) will finish before we reach new_time
                dt = next_finish  # actual step to next finish event

                # All active tasks progress for dt with 1/n share
                for t in active:
                    t["remaining"] -= self.cal_progress( t, dt)
                    #print("rest pregress", t["index"], t["remaining"])
                    if t["remaining"] <= eps:
                        # mark finished exactly at this event time
                        t["remaining"] = 0.0
                        t["active"] = False
                        t["end"] = self.time + dt
                        meet_end = True
                for t in active:
                    if t["active"] == False:
                        self.r_counter_update(t, "sub")
                # Move time forward and reduce remaining_dt
                self.time += dt
                remaining_dt -= dt
                if meet_end == True:
                    break
                

            else:
                # No one finishes within remaining_dt: just partial progress
                dt = remaining_dt
                for t in active:
                    t["remaining"] -= self.cal_progress( t, dt)
                self.time += dt
                remaining_dt = 0.0
        return meet_end

    def lazy_add_task(self, start,end, index):

        duration = end - start

        ftuple, ttuple, path = self.init_path_extract(index,self.steps_dlist, self.plan_list, self.band_str, self.BatchAllocateList)
        task = {
            "index": index, # index of the (mbatch, stage)...
            "id": self.next_id,
            "start": start,
            "end": None,
            "remaining": duration,
            "active": True,
            "dead": False,
            "ftuple": ftuple,
            "ttuple": ttuple,
            "path": path,
            "BatchAllocateList": self.BatchAllocateList
        }

        self.next_id += 1
        self.lazy_stack.append(task)
        #print(self.lazy_stack)

    def cal_progress(self, task, time):
        shared_resource_init = 1
        act_shared_r = 1/self.shared_count

        tmin = float("inf")
        for perLAN in task["path"]:
            LAN_resource_init = self.resource_ratio_list[perLAN]
            act_LAN = LAN_resource_init/self.LAN_count[perLAN]
            if tmin>act_LAN:
                tmin = act_LAN
        
        if len(task["path"]) == 0:
            tmin = 0

        return time*(act_shared_r + tmin)


    
    def remain_time(self, task):
        remaining_work = task["remaining"]
        shared_resource_init = 1
        act_shared_r = 1/self.shared_count

        tmin = float("inf")
        for perLAN in task["path"]:
            LAN_resource_init = self.resource_ratio_list[perLAN]
            act_LAN = LAN_resource_init/self.LAN_count[perLAN]
            if tmin>act_LAN:
                tmin = act_LAN
        
        if len(task["path"]) == 0:
            tmin = 0

        return remaining_work/(act_shared_r + tmin)



    #unused method......
    def sync_dump(self):
        """Add a new task starting at 'start' (simulate to that point first)."""
        # advance time to new task start

        self.lazy_stack.sort(key=lambda x: x["start"])

        for per_task in self.lazy_stack:
            start = per_task["start"]
            self.advance_to(start)
            self.tasks.append(per_task)

        self.lazy_stack = []

        #print(f"Task {task['id']} joined at t={start}, duration={duration}")
        return

    def r_counter_update(self,per_task, mode = "add"):
        if mode == "add":
            for LAN_r in per_task["path"]:
                self.LAN_count[LAN_r]+=1
            self.shared_count+=1
        elif mode == "sub":
            for LAN_r in per_task["path"]:
                self.LAN_count[LAN_r]-=1
                if self.LAN_count[LAN_r]<0: print("LAN_counter update error......")
            self.shared_count-=1
            if  self.shared_count<0: print("shared_counter update error......")

    def add_start_until_end(self):
        """Add a new task starting at 'start' (simulate to that point first)."""
        # advance time to new task start

        self.lazy_stack.sort(key=lambda x: x["start"])
        counter = 0
        meet_end = False
        for per_task in self.lazy_stack:
            start = per_task["start"]
            meet_end = self.advance_to(start)
            if  meet_end == True:
                break
            self.tasks.append(per_task)
            ##now we add the task into our runtime counter, so, we need to maintain the resource_counter...
            self.r_counter_update(per_task, "add")

            counter += 1

        self.lazy_stack = self.lazy_stack[counter:]

        #print(f"Task {task['id']} joined at t={start}, duration={duration}")
        return meet_end
    
    def step_until_next_finish(self):
        active = [t for t in self.tasks if t["active"]]
        if not active:
            return None
        n = len(active)
        # smallest remaining time among all * total bandwidth share*
        min_remaining_time = min(self.remain_time(t) for t in active)
        next_finish_time = self.time + min_remaining_time
        self.advance_to(next_finish_time)
        return next_finish_time
    
    ##main method used for the update...
    def pop_finished(self):
        """Pop and return all finished tasks (earliest ones)."""
        finished = [t for t in self.tasks if not t["active"] and t["dead"] == False]

        finished.sort(key=lambda x: x["remaining"])

        if len(finished) == 0:
            #before do next finish: we add starts lazily...
            result = self.add_start_until_end()
            if result == False: # add all start, but still doesn't meet the END...
                result = self.step_until_next_finish()
            #redo again...
            finished = [t for t in self.tasks if not t["active"] and t["dead"] == False]
            finished.sort(key=lambda x: x["remaining"])
            if len(finished) == 0:
                #again being zero, then no more...
                return (-1,-1, -1),0,0

        candidate = finished[0]

        candidate["dead"] = True
        #print(self.tasks)
        return candidate["index"], candidate["start"],candidate["end"]






def cal_index(s, m, fb, Idependent):
    for i, v in enumerate(Idependent[s]):
        if (m, fb) == v:
            return i

def construct_processing(num_microbatches,num_stages,
                         fp_schedule,bp_schedule,gathering_schedule,
                         steps_plist,steps_dlist,
                         forward_times, backward_times, gathering_times,
                         start_percent,
                         enableshare,
                         Idependent,
                         mode = '',
                         jmode = None
                         ):
    #print("with in precessing func:", num_stages, num_microbatches, Idependent)
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
                if m == 0 and steps_plist[s]!=[]: # no intra-dependency

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
                    pref = Idependent[s][cal_index(s, m, 'f', Idependent) - 1]
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
                if jmode != "training":
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
                    pref = Idependent[s][cal_index(s, m, 'b',Idependent) - 1]
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
            if jmode != "training":
                continue
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


def construct_processing_even_channel(num_microbatches,num_stages,
                         fp_schedule,bp_schedule,gathering_schedule,
                         steps_plist,steps_dlist,
                         forward_times, backward_times, gathering_times,
                         start_percent,
                         enableshare,
                         Idependent,
                         mode = "",
                         jmode = None,
                        plan_list = None,
                        BatchAllocateList = None,
                        band_str = None
                         ):
    communication_recorder_end = 0
    router_stack = BandwidthStack(plan_list,BatchAllocateList, band_str,steps_dlist)
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
                if m == 0 and steps_plist[s]!=[]: # no intra-dependency

                    #all precedence are finished...
                    indicator = True
                    for p_item in steps_plist[s]:
                        if (p_item,m) not in fp_schedule:
                            indicator = False
                        elif fp_schedule[(p_item,m)][0] == -1:
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
                                router_stack.lazy_add_task(start, start+forward_times[s], ('f',s,m))
                                candidaterecord+=1
                                fp_schedule[(s, m)] = (-1,-1)
                                                               
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
                            elif fp_schedule[(p_item,m)][0] == -1:
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
                    pref = Idependent[s][cal_index(s, m, 'f', Idependent) - 1]
                    #print(pref)
                    if pref[1] == 'f' and (s,pref[0]) in fp_schedule and fp_schedule[(s,pref[0])][0]!=-1:
                        start = max(start, fp_schedule[(s,pref[0])][1])
                    elif pref[1] == 'b' and (s,pref[0]) in bp_schedule and bp_schedule[(s,pref[0])][0]!=-1:
                        start = max(start, bp_schedule[(s,pref[0])][1])
                    else:
                        continue


                    if s%2 == 1 and enableshare == True: ####jin: added shared bandwidth version
                        candidaterecord+=1
                        router_stack.lazy_add_task(start, start+forward_times[s], ('f',s,m))
                        fp_schedule[(s, m)] = (-1,-1)
                    else:
                        fp_schedule[(s, m)] = (start, start+forward_times[s])
                    upi = 1
        #search for backward updating...          
        for m in range(num_microbatches):
            for s in range(num_stages-1, -1,-1):
                if (s, m) in bp_schedule:
                    continue
                if jmode != "training":
                    continue
                if m>=0: #
                    start = 0
                    if s<num_stages-1: # there is inter-dependency

                                          
                        if (steps_dlist[s],m) in bp_schedule:
                            if bp_schedule[(steps_dlist[s],m)][0] == -1: continue

                            k = bp_schedule[(steps_dlist[s], m)][1]
                            if s%2 == 1:
                                k = bp_schedule[(steps_dlist[s], m)][0]+start_percent*(bp_schedule[(steps_dlist[s], m)][1]-bp_schedule[(steps_dlist[s], m)][0])
                            start = max(start, k)
                        else:
                            continue
                    #then is intra-dependency
                    pref = Idependent[s][cal_index(s, m, 'b',Idependent) - 1]
                    if pref[1] == 'f' and (s,pref[0]) in fp_schedule and fp_schedule[(s,pref[0])][0]!=-1:
                        start = max(start, fp_schedule[(s,pref[0])][1])
                    elif pref[1] == 'b' and (s,pref[0]) in bp_schedule and bp_schedule[(s,pref[0])][0]!=-1:
                        start = max(start, bp_schedule[(s,pref[0])][1])
                    else:
                        continue

                    if s%2 == 1 and enableshare == True: ####jin: added shared bandwidth version
                        candidaterecord+=1
                        router_stack.lazy_add_task(start, start+forward_times[s], ('b',s,m))
                        bp_schedule[(s, m)] = (-1,-1)
                    else:
                        bp_schedule[(s, m)] = (start, start+backward_times[s])
                    upi = 1

        #all gathering update...
        for s in range(num_stages): # updatind the gathering shedule times:
            if jmode != "training":
                continue
            if gathering_times[s] == 0:
                continue
            if s in gathering_schedule:
                continue
            if (s, num_microbatches-1) not in bp_schedule:
                continue
            start = bp_schedule[(s, num_microbatches-1)][1]
            if enableshare == True:
                candidaterecord+=1
                router_stack.lazy_add_task(start, start+gathering_times[s], ('g',s))
                gathering_schedule[s] = (-1, -1)
            else:
                gathering_schedule[s] = (start, start+gathering_times[s])
            upi = 1



        #if candidates != None:
        #    start = current_smallest_s
        #    if start<communication_recorder_end:
        #        start = communication_recorder_end
        #    if candidates[0] == 'f':
        #        lasting = forward_times[candidates[1]]
        #        fp_schedule[(candidates[1],candidates[2])] = (start, start+lasting)
        #    elif candidates[0] == 'b':
        #        lasting = backward_times[candidates[1]]
        #        bp_schedule[(candidates[1],candidates[2])] = (start, start+lasting)
        #    elif candidates[0] == 'g':
        #        lasting = gathering_times[candidates[1]]
        #        gathering_schedule[candidates[1]] = (start, start+lasting)
        #    communication_recorder_end = start + lasting    
        #    #candidaterecord-=1
        #    upi = 1
        if upi == 0: # no update happened ... then we update the communication lazily
            # begin to update the communication task...
            #print("till communication...")
            #print("current stack's time: ", router_stack.time)
            #print(router_stack.tasks)
            #print(router_stack.lazy_stack)
            #print(fp_schedule)
            #print(bp_schedule)
            #router_stack.sync_dump()   # sync_dump is unnecessary operation...
            #lazy update all stored things...
            index, start, end = router_stack.pop_finished()
            cat = index[0]
            if cat == 'f':
                fp_schedule[(index[1],index[2])] = (start, end)
            elif cat == 'b':
                bp_schedule[(index[1],index[2])] = (start, end)
            elif cat == 'g':
                gathering_schedule[index[1]] = (start, end)
            else: #-1 situation...
                #print("at the end of processing, no update and router becomes empty?:",candidaterecord)
                break

            candidaterecord-=1
            #print("after operation:")
            #print(router_stack.tasks)
            #print(router_stack.lazy_stack)        
            #print(fp_schedule)
            #print(bp_schedule)   



def pip_ploting_graph(num_stages = 5, num_microbatches = 10,
                forward_times = [6.6, 2.5, 2.5, 4.0, 5.6],
                backward_times = [3.0, 2.0, 4.0, 4,  5.7],
                gathering_times = [9,0,6,0,8],
                steps_dlist = [1,4,3,4,-1],
                enableshare = False, enablegraph = True,
                storage = "./scratch",
                group_plan = [],percentage = 0.6,
                trivial_mode = "fifo",
                plan_list = None,
                BatchAllocateList = None,
                band_str = None,
                jmode= "training"):
    # ----------------------
    # Parameters
    # ----------------------

    #print(num_stages,num_microbatches,forward_times,backward_times,gathering_times)
    #comm_delay = 0.2    
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
        if(num_stages-s>num_microbatches or jmode != "training"): #no enouch microbatches we can use...
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



            
    steps_plist = sgg.steps_precedence_list(steps_dlist)

    #print(steps_plist)
    #print(steps_dlist)
    #print(forward_times,backward_times, gathering_times)
    #print("begin working...")
    if trivial_mode == "fifo":
        construct_processing(num_microbatches,num_stages,
                         fp_schedule,bp_schedule,gathering_schedule,
                         steps_plist,steps_dlist,
                         forward_times, backward_times, gathering_times,
                         start_percent,
                         enableshare,
                         Idependent,
                         mode = trivial_mode,
                         jmode = jmode
                          #this is a func
                         )
    else:
        construct_processing_even_channel(num_microbatches,num_stages,
                         fp_schedule,bp_schedule,gathering_schedule,
                         steps_plist,steps_dlist,
                         forward_times, backward_times, gathering_times,
                         start_percent,
                         enableshare,
                         Idependent,
                         mode = trivial_mode,
                         jmode = jmode,
                        plan_list = plan_list,
                        BatchAllocateList = BatchAllocateList,
                        band_str = band_str
                          #this is a func
                         )
    #print(len(bp_schedule), len(fp_schedule),len(gathering_schedule))

    #important: generate max_time:

    #print(fp_schedule,bp_schedule,gathering_schedule)

    max_time = 0
    try:
        if  jmode == "training":
            max_time = max(bp_schedule[(i, num_microbatches - 1)][1]for i in range(0,num_stages,2))
        else:
            max_time = max(fp_schedule[(i, num_microbatches - 1)][1]for i in range(0,num_stages,2))
    except:
        if jmode == "training":
            print(fp_schedule,bp_schedule,gathering_schedule)
            print("erro drawing, need further debug...")
        return 0
    for s in range(num_stages): #searching for biggest time...
        if jmode != "training":
            continue
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
        max(et for (_, et) in bp_schedule.values()) if bp_schedule !={} else 0,
        max(et for (_, et) in fp_schedule.values())
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
                        percentage = 0.6,
                        ratio = 0,
                         jmode= "training"):
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
    biggest_end = 0
    for i in range(fb_amount): 
        if_compute,if_first_part, nsteps, nmicrobatch, ntasks, if_back = sgg.partitioner_shift(i,s,b,a,aa)
        if True:
            start, end = cj.RCPSP_sol_parser_shift(RCPSP_tasks,s, a,b,
                            nsteps, nmicrobatch, 
                            ntasks,
                            aa,if_gathering = False)
            if start == end: #dummy tasks
                continue
            if end>biggest_end: biggest_end = end
            comm_Independent.append([0,nsteps, nmicrobatch, ntasks,if_back, start])
            if if_back ==0:
                fp_schedule[(nsteps, nmicrobatch, ntasks)] = (start/ratio, end/ratio)
            else:
                bp_schedule[(nsteps, nmicrobatch, ntasks)] = (start/ratio, end/ratio)
    for i in range(fb_amount, total_amount):
        nsteps, ntasks = sgg.partitioner_shift(i,s,b,a,aa)
        start, end = cj.RCPSP_sol_parser_shift(RCPSP_tasks,s, a,b,
                                         nsteps, nmicrobatch, 
                                         ntasks,
                                         aa,if_gathering = True)
        #print(i, start, end )
        if start == end: #dummy tasks
            continue
        if end>biggest_end: biggest_end = end
        comm_Independent.append([1,nsteps,-1, ntasks,-1, start])
        gathering_schedule[(nsteps,ntasks)] = (start/ratio, end/ratio)
    #comm_Independent.sort(key=lambda x: x[-1])
    max_time = biggest_end/ratio

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
        max(et for (_, et) in bp_schedule.values()) if bp_schedule !={} else 0,
        max(et for (_, et) in fp_schedule.values())
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
    
    #stak test::
    router = BandwidthStack()
    router.lazy_add_task(1.0,4.0,(0,0))
    router.lazy_add_task(2.0,7.0,(0,1))
    router.lazy_add_task(3.0,12.0,(0,2))
    router.sync_dump()
    print(router.pop_finished())
    print(router.pop_finished())
    print(router.pop_finished())




    #m, r = cj.RCPSP_solver(pfile ="../RCPSP/scratch_graph.sm")
    ##cj.RCPSP_plot(m, r)
    #RCPSP_tasks = r.best.tasks
    #pip_ploting_graph(group_plan = [{'device':[0], 'layer':[0]}]*5,enableshare=True)
    #pip_ploting_graph_real(RCPSP_tasks,
    #                    ns=5, a=3, b=3, aa=5, 
    #                    forward_times = [100, 300, 500,700,900],
    #                    backward_times = [200, 400,600,800,1000],
    #                    gathering_times = [1000,0,1000,0,0],
    #                    steps_dlist=[1,4,3,4,-1],
    #                    RProfile=[(80,80)]*2,
    #                    UnitR=80,
    #                    gatheringRprofile=[80,0,80,0,0],
    #                    enablegraph = True,
    #                    storage = "./scratch",
    #                    group_plan = [{'device':[0], 'layer':[0]}]*3)
    

    pip_ploting_graph(num_stages = 5, num_microbatches = 10,
                forward_times = [5, 5, 3, 3, 4],
                backward_times = [5, 5, 3, 3,  4],
                gathering_times = [7,0,7,0,7],
                steps_dlist = [1,4,3,4,-1],
                enableshare = True, enablegraph = True,
                storage = "./thescratch",
                group_plan = [],percentage = 0.6,
                trivial_mode = "notFIFO")