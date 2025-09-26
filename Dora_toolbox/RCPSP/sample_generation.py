import random
import sys
import math

class TaskStatus:
    def __init__(self, index, s,b,a , Tprofile, Rprofile, UnitR, isgathering = False, aa = 0, gatheringTprofile= [20,0,20], gatheringRprofile=[20,0,20]):
        """
        Initialize the status variables based on input parameters.
        """
        if isgathering == False:
            self.isgathering = False
            self.if_compute, self.nsteps, self.nmicrobatch, self.ntasks, self.if_back = partitioner(index,s,b,a)
            #print(partitioner(index,s,b,a))
            self.T = Tprofile[self.nsteps][self.if_back]  # get total time span
            self.R = 0 # by default
            if self.ntasks>=0: #subTask
                self.R = Rprofile[self.nsteps//2][self.if_back] #get overall resource needed 
                if self.ntasks ==0 or self.ntasks == a - 1:
                    self.T = 0
                    self.R = 0
                else: # ok, we have the subtask
                    self.T, self.R = suballocator(self.T, self.R, a-2 ,UnitR)
                    self.R = math.ceil(self.R)
        else:
            #print(aa)
            self.isgathering = True
            self.nsteps,self.ntasks = partitioner(index,s,b,a,aa)
            #print(self.nsteps,self.ntasks)
            self.T = 0
            self.R = 0
            if self.ntasks>0 and self.ntasks<aa-1 and gatheringTprofile[self.nsteps]>0:
                self.R = gatheringRprofile[self.nsteps]
                self.T = gatheringTprofile[self.nsteps]
                #print(self.R, self.T)
                self.T, self.R = suballocator(self.T, self.R, aa-2 ,UnitR)
                self.R = math.ceil(self.R)
        # List to store dependency nodes
        self.dependency_nodes = []

    def add_dependency(self, node):
        """
        Add a node to the dependency list.
        """
        self.dependency_nodes.append(node)

    def __repr__(self):
        """
        For clean printing of the object state.
        """
        if self.isgathering == True:
            return(f"gathering states")

        return (
            f"PipelineStageStatus("
            f"if_compute={self.if_compute}, "
            f"nsteps={self.nsteps}, "
            f"nmicrobatch={self.nmicrobatch}, "
            f"ntasks={self.ntasks}, "
            f"if_back={self.if_back}, "
            f"ResourceT&R={self.T}, {self.R}, "
            f"dependencies={self.dependency_nodes})"
        )
    

def suballocator(T, R, a ,UnitR):
    totalR = T*R # T should be integer
    totalunit = math.ceil(totalR /UnitR) # get necessary time unit
    pertaskunit = math.ceil(totalunit/a) # get necessart per sub task
    totalunit = pertaskunit*a
    pertaskR = totalR/a
    return pertaskunit, pertaskR/pertaskunit


def partitioner( index, s,b,a, aa = 0):
    if index >= (s//2+1)*b*2+(s//2)*b*2*a: #out of scope
        gatheri = index - ((s//2+1)*b*2+(s//2)*b*2*a)
        #print(gatheri)
        nsteps = gatheri//aa
        ntasks = gatheri%aa
        return nsteps, ntasks
    if_compute = 1
    if_back = 1
    ntasks = -1
    n2step = index // (2*b*a + 2*b) #one step for compute, one step for communication
    brest = index % (2*b*a + 2*b) #1 st reminder 
    if brest >= 2*b:
        if_compute = 0 # you are the communication part
        nsteps = n2step*2+1
        nmicrobatch = (brest - 2*b)//a
        ntasks = (brest - 2*b)%a
        if_back = 1 if nmicrobatch >= b else 0        
    else:
        nsteps = n2step*2
        nmicrobatch = brest 
        if_back = 1 if nmicrobatch >= b else 0
    return if_compute, nsteps, nmicrobatch, ntasks, if_back

def indexer (s,a,b, step, nmicrobatch, ntasks = -1):
    if(step%2 == 0):
        return (step//2)*(b*2+b*2*a) + nmicrobatch
    else:
        return (step//2)*(b*2+b*2*a) + b*2 + nmicrobatch*a + ntasks

def gindexer (s,a,b,step, nmicrobatch, ntasks = -1,aa = 4):
    total_amount = (s//2+1)*b*2+(s//2)*b*2*a
    return total_amount+step*aa + ntasks


def generatesm(pfile = "scratch2.sm",s= 3, b=3, a=4,
                TProfile=[(1,2),(3,4),(5,6)], RProfile = [(15,17)]*1, UnitR = 7,
                gatheringTprofile= [20,0,20],gatheringRprofile= [20,0,20],
                  aa = 5):
    total_amount = (s//2+1)*b*2+(s//2)*b*2*a
    end_node_index = total_amount+s*aa
    #print(total_amount)
    TaskL = []
    for i in range(total_amount):
        TaskL.append(TaskStatus(i, s,b,a , TProfile, RProfile, UnitR))
    for i in range(total_amount, total_amount+s*aa):
        #print("for sub gathering:"+str(i))
        TaskL.append(TaskStatus(i, s, b, a,[],[],
                                UnitR, 
                                isgathering = True, aa=aa, 
                                gatheringTprofile = gatheringTprofile, 
                                gatheringRprofile = gatheringRprofile))

    mbatch = b*2 #microbatch
    amount = a  #smaller tasks
    steps = s
    Dep = []

    #append btn steps:
    for i in range(steps-1): 
        #i = 0 means 0->1's dependency:
        for j in range(b):
            #now, for per line, we add dep from comp to comm, and from comm to comp
            #from comp to comm:
            if(i %2 == 0):
                TaskL[indexer(s,a,b,i,j)].add_dependency(indexer(s,a,b,i+1,j,0))
                TaskL[indexer(s,a,b,i+1,b+j,a-1)].add_dependency(indexer(s,a,b,i,b+j))
            else:
                TaskL[indexer(s,a,b,i,j,a-1)].add_dependency(indexer(s,a,b,i+1,j))
                TaskL[indexer(s,a,b,i+1,b+j)].add_dependency(indexer(s,a,b,i,b+j,0))                
            #Dep.append(((i,j,amount-1),(i+1,j,0)))
            #Dep.append(((i+1,j+mbatch//2,amount-1),(i,j+mbatch//2,0)))


    #append intra sub-tasks dependency within communication only:
    for i in range(steps): #example of astroid
        if i%2==0: #this is computation, no need for intra sub-tasks dependency
            continue
        for j in range(b*2):
            for k in range(a-2):
                TaskL[indexer(s,a,b,i,j,0)].add_dependency(indexer(s,a,b,i,j,k+1))
                TaskL[indexer(s,a,b,i,j,k+1)].add_dependency(indexer(s,a,b,i,j,a-1))
                #Dep.append(((i,j,0),(i,j,k+1)))
                #Dep.append(((i,j,k+1),(i,j,amount-1)))

    #new:so do for gathering tasks:
    for i in range(steps):
        for k in range(aa-2):
            TaskL[gindexer(s,a,b,i,-1,0,aa)].add_dependency(gindexer(s,a,b,i,-1,k+1,aa))
            TaskL[gindexer(s,a,b,i,-1,k+1,aa)].add_dependency(gindexer(s,a,b,i,-1,aa-1,aa))



    #append intra steps:
    #first, generate default 1F1B order schedule:
    order = []
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

    #print(order)


    #append over scratch Dep
    for i in range(steps):
        for j in range(2*b-1):
            if i%2 == 0: # means it is computation task
                TaskL[indexer(s,a,b,i,order[i][j])].add_dependency(indexer(s,a,b,i,order[i][j+1]))
            else: #means it is communication task, we add dep btn sub small task...
                TaskL[indexer(s,a,b,i,order[i][j],a-1)].add_dependency(indexer(s,a,b,i,order[i][j+1],0))
            #Dep.append(((i,order[i][j],amount-1),(i,order[i][j+1],0)))

    #append for gathering tasks:
    for i in range(steps):
        if i%2 == 0:
            TaskL[indexer(s,a,b,i,mbatch-1)].add_dependency(gindexer(s,a,b,i,-1,0,aa))
        else:
            TaskL[indexer(s,a,b,i,mbatch-1,a-1)].add_dependency(gindexer(s,a,b,i,-1,0,aa))

        TaskL[gindexer(s,a,b,i,-1,aa-1,aa)].add_dependency(end_node_index)


    file_name = pfile
    with open(file_name, "w") as file:
        #step 0: total numbers:
        num = steps*mbatch*amount
        #file.write(str(num) + "\n")

        #1st part: PRECEDENCE RELATIONS:
        file.write("PRECEDENCE RELATIONS:"+ "\n")
        file.write("jobnr.    #modes  #successors   successors"+ "\n")
            #real work begin

        #generate precedence dict
        #pred = {}
        #for dep in Dep:
        #    fromt = dep[0]
        #    tot = dep[1]
        #    fromn = fromt[0]*mbatch*amount+fromt[1]*amount+fromt[2]
        #    ton = tot[0]*mbatch*amount+tot[1]*amount+tot[2]
        #    if fromn in pred:
        #        pred[fromn].append(ton)
        #    else:
        #        pred[fromn] = [ton]
        ####***for 70nd, i.e. last computation blk of 1st stage
        ###we add one dummy node successor...
        #if mbatch*amount-1 in pred:
        #    pred[mbatch*amount-1].append(num)
        #else:
        #    pred[mbatch*amount-1] = [num]


        ##write 1st line in manual

        #file.write("  1        1          1           2  "+"\n")
        # notice:: index should +2 for sm file format(as now we add 1st dummy task as first task now...)

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


if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("Usage: python3 s_g.py <steps> <microbatch> <subtasks_amount>")
        sys.exit(1)

    s = int(sys.argv[1])
    m = int(sys.argv[2])
    a = int(sys.argv[3])

    generatesm()


    ##abandom


    
    
    

