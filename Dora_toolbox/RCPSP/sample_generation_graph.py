import random
import sys
import math
from psplib_ import TaskStatus_prime
import random

class TaskStatus_graph(TaskStatus_prime):
    def initialize(self):
        if self.isgathering == False:
            self.isgathering = False
            self.if_compute, self.if_firstpart ,self.nsteps, self.nmicrobatch, self.ntasks, self.if_back = self.partitioner(self.index,self.s,self.b,self.a)
            #print(partitioner(index,s,b,a))
            ratio = self.percentage if self.if_firstpart == 0 else 1-self.percentage
            ratio = ratio if self.if_compute == 1 else 1
            self.T = int(self.Tprofile[self.nsteps][self.if_back]*ratio)  # get total time span
            self.R = 0 # by default
            if self.ntasks>=0: #subTask of communication...
                self.R = self.Rprofile[self.nsteps//2][self.if_back] #get overall resource needed 
                if self.ntasks ==0 or self.ntasks == self.a - 1:
                    self.T = 0
                    self.R = 0
                else: # ok, we have the subtask
                    self.T, self.R = self.suballocator(self.T, self.R, self.a-2 ,self.UnitR)
                    self.R = math.ceil(self.R)
        else:
            #print(aa)
            self.isgathering = True
            self.nsteps,self.ntasks = self.partitioner(self.index,self.s,self.b,self.a,self.aa)
            #print(self.nsteps,self.ntasks)
            self.T = 0
            self.R = 0
            if self.ntasks>0 and self.ntasks<self.aa-1 and self.gatheringTprofile[self.nsteps]>0:
                self.R = self.gatheringRprofile[self.nsteps]
                self.T = self.gatheringTprofile[self.nsteps]
                #print(self.R, self.T)
                self.T, self.R = self.suballocator(self.T, self.R, self.aa-2 ,self.UnitR)
                self.R = math.ceil(self.R)
    

def suballocator_shift(T, R, a ,UnitR):
    totalR = T*R # T should be integer
    totalunit = math.ceil(totalR /UnitR) # get necessary time unit
    pertaskunit = math.ceil(totalunit/a) # get necessart per sub task
    totalunit = pertaskunit*a
    pertaskR = totalR/a
    return pertaskunit, pertaskR/pertaskunit

def suballocator_random(T, R, a ,UnitR):

    return random.randint(10, 40), random.randint(70, 80)


def partitioner_shift( index, s,b,a, aa = 0):
    total_amount = (s//2+1)*b*2*2+(s//2)*b*2*a
    amount_comp = b*2*2
    amount_comm = b*2*a
    if index >= total_amount: #out of scope added the two step computation
        gatheri = index - total_amount
        #print(gatheri)
        nsteps = gatheri//aa
        ntasks = gatheri%aa
        return nsteps, ntasks
    if_compute = 1
    if_back = 1
    if_firstpart = -1
    ntasks = -1
    n2step = index // (amount_comm+amount_comp) #one step for compute, one step for communication
    brest = index % (amount_comm+amount_comp) #1 st reminder 
    if brest >= amount_comp:
        if_compute = 0 # you are the communication part
        nsteps = n2step*2+1
        nmicrobatch = (brest - amount_comp)//a
        ntasks = (brest - amount_comp)%a
        if_back = 1 if nmicrobatch >= b else 0        
    else:
        nsteps = n2step*2
        nmicrobatch = brest//2
        if_firstpart = (1-brest%2)
        if_back = 1 if nmicrobatch >= b else 0
    return if_compute,if_firstpart, nsteps, nmicrobatch, ntasks, if_back

def indexer_shift(s,a,b, step, nmicrobatch, ntasks = -1):
    if(step%2 == 0):
        return (step//2)*(b*2*2+b*2*a) + nmicrobatch*2
    else:
        return (step//2)*(b*2*2+b*2*a) + b*2*2 + nmicrobatch*a + ntasks

def gindexer_shift(s,a,b,step, nmicrobatch, ntasks = -1,aa = 4):
    total_amount = (s//2+1)*b*2*2+(s//2)*b*2*a
    return total_amount+step*aa + ntasks


def steps_precedence_list(steps_dependent_list):
    nsteps = len(steps_dependent_list)
    final = [[] for _ in range(nsteps)]
    for i, v in enumerate(steps_dependent_list):
        #print(i,v)
        if v != -1 and v != None:
            final[v].append(i)
    return final


def get_rank(steps_dependent_list):
    nsteps = len(steps_dependent_list)
    final = [-1]*nsteps
    for i, value in enumerate(list(reversed(steps_dependent_list))):
        idx = nsteps-1-i
        if value == -1 or value == None:
            final[idx] = 0
        else:
            final[idx] = final[value]+1
    return final



def OFOB_graph_scheduler(steps, b, steps_dependent_list):

    rank_list = get_rank(steps_dependent_list)

    #first, generate default 1F1B order schedule:
    order = []
    final_order = []
    for s in range(steps):
        ini = []
        if(steps-s>b): #no enouch microbatches we can use...
            for i in range(b):
                ini.append(i)
            
            for i in range(b):
                ini.append(i+b)
            order.append(ini)
            continue

        for i in range(steps-s):
            ini.append(i)
        back = 1
        ini_back = 0
        ini_for = steps-s
        for i in range(2*(b - (steps-s))):
            if back == 1:
                ini.append(ini_back+b)
                ini_back+=1
                back = 0
            else:
                ini.append(ini_for)
                ini_for+=1
                back = 1
        for i in range(steps-s):
            ini.append(ini_back+b)
            ini_back+=1
        order.append(ini)

    for i in range(steps):
        final_order.append(order[steps - 1 - rank_list[i]])


    return final_order


def depedency_generate_graph(TaskL, s= 3, b=3, a=4, aa = 5,sd_list = [1,4,3,4,-1],indexer = None, indexerg = None, scheduler = None):
    total_amount = (s//2+1)*b*2*2+(s//2)*b*2*a   #one more *2 for two divided computation tasks...
    end_node_index = total_amount+s*aa
    mbatch = b*2 #microbatch
    amount = a  #smaller tasks
    steps = s
    Dep = []

    #1. append between steps:
    for i in range(steps-1): 
        #i = 0 means 0->1's dependency:
        for j in range(b):
            #now, for per line, we add dep from comp to comm, and from comm to comp
            #from comp to comm:
            if(i %2 == 0): #computation step
                next = sd_list[i]
                if next == -1 or next == None: continue
                id1 = indexer(s,a,b,i,j)
                id2 = indexer(s,a,b,next,j,0)
                TaskL[id1].add_dependency(id2)
                TaskL[id2].add_precedence(id1)

                id1 = indexer(s,a,b,next,b+j,a-1)
                id2 = indexer(s,a,b,i,b+j)

                TaskL[id1].add_dependency(id2)
                TaskL[id2].add_precedence(id1)

            else:
                next = sd_list[i]
                if next == -1 or next == None: continue
                id1 = indexer(s,a,b,i,j,a-1)
                id2 = indexer(s,a,b,next,j)

                TaskL[id1].add_dependency(id2)
                TaskL[id2].add_precedence(id1)

                id1 = indexer(s,a,b,next,b+j)
                id2 = indexer(s,a,b,i,b+j,0)


                TaskL[id1].add_dependency(id2)
                TaskL[id2].add_precedence(id1)             



    #2. append intra sub-tasks dependency within communication and computation:
    for i in range(steps): #example of astroid
        for j in range(b*2):
            if i%2 == 1:
                #work for comm
                for k in range(a-2):
                    id1 = indexer(s,a,b,i,j,0)
                    id2 = indexer(s,a,b,i,j,k+1)
                    TaskL[id1].add_dependency(id2)
                    TaskL[id2].add_precedence(id1)

                    id1 = indexer(s,a,b,i,j,k+1)
                    id2 = indexer(s,a,b,i,j,a-1)
                    TaskL[id1].add_dependency(id2)
                    TaskL[id2].add_precedence(id1)
            else:        
                #work for comp
                id1 = indexer(s,a,b,i,j)
                id2 = id1+1
                TaskL[id1].add_dependency(id2)
                TaskL[id2].add_precedence(id1)

    #new:so do for gathering tasks:
    for i in range(steps):
        for k in range(aa-2):
            
            id1 = indexerg(s,a,b,i,-1,0,aa)
            id2 = indexerg(s,a,b,i,-1,k+1,aa)
            TaskL[id1].add_dependency(id2)
            TaskL[id2].add_precedence(id1)

            id1 = indexerg(s,a,b,i,-1,k+1,aa)
            id2 = indexerg(s,a,b,i,-1,aa-1,aa)
            TaskL[id1].add_dependency(id2)
            TaskL[id2].add_precedence(id1)


    #append intra steps and intra phase's dependency...
    order = scheduler(steps, b, sd_list)
    #print(order)
    #append over scratch Dep
    for i in range(steps):
        for j in range(2*b-1):
            if i%2 == 0: # means it is computation task
                id1 = indexer(s,a,b,i,order[i][j]) + 1 #2nd computation blk
                id2 = indexer(s,a,b,i,order[i][j+1])
                TaskL[id1].add_dependency(id2)
                TaskL[id2].add_precedence(id1)
            else: #means it is communication task, we add dep btn sub small task...
                id1 = indexer(s,a,b,i,order[i][j],a-1)
                id2 = indexer(s,a,b,i,order[i][j+1],0)
                TaskL[id1].add_dependency(id2)
                TaskL[id2].add_precedence(id1)
            #Dep.append(((i,order[i][j],amount-1),(i,order[i][j+1],0)))

    #append for gathering tasks:
    for i in range(steps):
        id2 = indexerg(s,a,b,i,-1,0,aa)
        if i%2 == 0:
            id1 = indexer(s,a,b,i,mbatch-1) + 1 # 2nd computation blk
            TaskL[id1].add_dependency(id2)
            TaskL[id2].add_precedence(id1)
        else:
            id1 = indexer(s,a,b,i,mbatch-1,a-1)
            TaskL[id1].add_dependency(id2)
            TaskL[id2].add_precedence(id1)

        #add to a final dummy nodes...
        TaskL[indexerg(s,a,b,i,-1,aa-1,aa)].add_dependency(end_node_index)





def generatesm_graph(pfile = "scratch_graph.sm",s= 5, b=3, a=3,
                TProfile=[(100,200),(300,400),(500,600),(700, 800),(900,1000)], RProfile = [(80,80)]*2, UnitR = 80,
                gatheringTprofile= [1000,0,1000,0,0],gatheringRprofile= [80,0,80,0,0],
                aa = 5,percent = 1, sd_list = [1,4,3,4,-1],
                indexer = indexer_shift, indexerg = gindexer_shift,
                d_generate = depedency_generate_graph, scheduler = OFOB_graph_scheduler):
    total_amount = (s//2+1)*b*2*2+(s//2)*b*2*a   #one more *2 for two divided computation tasks...
    end_node_index = total_amount+s*aa
    #print(total_amount)
    TaskL = []
    for i in range(total_amount):
        #print(f"number: {i}")
        ptr = TaskStatus_graph(i, s,b,a , TProfile, RProfile, UnitR,
                                    isgathering = False, aa=-1,
                                    method1=partitioner_shift,method2=suballocator_shift,
                                    percentage=percent
                                    )
        ptr.initialize()
        TaskL.append(ptr)
    #print("finished")
    for i in range(total_amount, total_amount+s*aa):
        #print("for sub gathering:"+str(i))
        ptr = TaskStatus_graph(i, s, b, a,[],[],
                                UnitR, 
                                isgathering = True, aa=aa, 
                                gatheringTprofile = gatheringTprofile, 
                                gatheringRprofile = gatheringRprofile,
                                method1=partitioner_shift,method2=suballocator_shift,
                                )
        ptr.initialize()
        TaskL.append(ptr)

    mbatch = b*2 #microbatch
    amount = a  #smaller tasks
    steps = s
    Dep = []
    d_generate(TaskL, s, b, a, aa,sd_list ,indexer, indexerg, scheduler = scheduler)
    file_name = pfile
    with open(file_name, "w") as file:
        #step 0: total numbers:
        num = steps*mbatch*amount
        #file.write(str(num) + "\n")

        #1st part: PRECEDENCE RELATIONS:
        file.write("PRECEDENCE RELATIONS:"+ "\n")
        file.write("jobnr.    #modes  #successors   successors"+ "\n")
            #real work begin

        for k in range(end_node_index):
            v = TaskL[k].dependency_nodes
            line = " "*2 + str(k+1)+" "*8+"1"+" "*8+ str(len(v))+" "*8
            for i in v:
                line+=(str(i+1)+" "*3)
            file.write(line+"\n")

        #don't have to...
        #write last line in maunal
        file.write("  "+str(end_node_index+1)+"        1          0     "+"\n")        

        file.write("---------dummy line---------"+"\n")

        #2nd part: REQUESTS/DURATIONS::
        file.write("REQUESTS/DURATIONS:"+ "\n")
        file.write("jobnr. mode duration  R 1"+"\n")
        file.write("---------dummy line---------"+"\n")
            #real work begin

        if_comput = 1
        step_counter = 0
        comput_counter = 0
        #file.write("  1      1     0       0 "+"\n")
        for t in range(end_node_index):
            T = TaskL[t].T
            strline = " "*2+str(t+1)+" "*5+"1"+" "*5
            strline= strline+str(T)+" "*5

            #cost = 0
            #if if_comput == 0:
            #    cost = RProfile[step_counter//2]
            cost = TaskL[t].R
            strline= strline+str(cost)+" "
            file.write(strline + "\n")

            #if comput_counter %(mbatch*amount) == 0: # the line is over, will switch modes...
            #    if_comput = if_comput ^ 1
            #    step_counter+=1

        file.write(" "+str(end_node_index+1)+"      1     0       0 "+"\n")

        file.write("---------dummy line---------"+"\n")



        #3rd part RESOURCEAVAILABILITIES
        file.write("RESOURCEAVAILABILITIES:\n")
        file.write("   R 1   "+"\n")
            #real work begin

        file.write("   "+str(UnitR)+"   "+"\n")

        file.write("---------dummy end---------"+"\n")
    return TaskL


if __name__ == "__main__":

    #print(OFOB_graph_scheduler(5, 3, [2,3,4,4,-1]))

    if len(sys.argv) < 4:
        print("Usage: python3 s_g.py <steps> <microbatch> <subtasks_amount>")
        sys.exit(1)

    s = int(sys.argv[1])
    m = int(sys.argv[2])
    a = int(sys.argv[3])

    generatesm_graph()


    ##abandom


    
    
    

