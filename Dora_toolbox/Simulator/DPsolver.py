from utils import TopKContainer, plan_estimator, graph_plan_estimator



def dynamic_programming_planning(L, N, M, k, s, Profilelor, alpha, SLO): # layers, devices amounts, k, s are hyper parameters...
    """
    Implements Algorithm 1: Dynamic Programming for Planning.
    """
    top_k_container = TopKContainer(k)

    F = {}  # Stores best scores
    P = {}  # Stores best plans
    #give initial value
    for p in range(0, min(L, N)):
        F[(0, 0, p)] = [float('inf')]
        P[(0, 0, p)] = [[]*s]
                

    for p in range(1, min(L, N+1, (M+1)//2)):
        for n in range(p, N + 1):
            for l in range(1, L + 1):
                # Initialize inner TopK container
                top_s_container = TopKContainer(s)
                #print((l,n,p))
                for n_prime in range(p-1,n):
                    for l_prime in range(0,l):
                        if n_prime*l_prime == 0 and n_prime+l_prime >0:
                            continue #this means we rely on a special non-esixt sub problem
                        if n_prime*l_prime>0 and p-1 ==0:
                            continue #stage 0's only afford that np and lp are both 0
                        #check the new assigning can be applied, no mem/energy constraint overflow
                        add_shard = {'layer':(L-l, L - l_prime), 'device':(N - n, N-n_prime)}
                        if Profilelor.DP_solver(add_shard['device'], add_shard['layer'], p-1) == False:
                            #print("fail")
                            continue
                        for r in range(min(s, len(P[(l_prime, n_prime, p-1)]))):
                            # Generate plan and get score. plan has s lists, in which an element is a dict...
                            #important here: we consider l and n are inversed counting

                            subplan = P[(l_prime, n_prime, p-1)][r]
                            plan = [add_shard]+ subplan
                            score, allocateplan = plan_estimator(plan,M,SLO, Profilelor, alpha)
                            # Update inner top-k
                            top_s_container.update(score, plan, allocateplan)

                # Update F and P
                F[(l, n, p)] = top_s_container.scores
                P[(l, n, p)] = top_s_container.plans

            #check: only we assign over all layers, we update the top_k_container
            #wrong check method...
            #if (len(top_s_container.plans[0])!= L):
            #    print(P)
            #    print(top_s_container.plans[0])
            #    print("something goes wrong... need to make sure we only monitor schedule with full layers")
            # Update outer container

            for v, plan,aplan in zip(top_s_container.scores, top_s_container.plans, top_s_container.allocateplans):
                top_k_container.update(v, plan, aplan)
        #print("stage amount"+str(p)+"finished.")
    #print(F(80, 1, 1))
    return top_k_container


def dynamic_programming_planning_MM(Structure, Layer_structure, N, M, k, s, Profilelor, alpha, SLO, jmode = "training"): # layers, devices amounts,M is microbatch amount. k, s are hyper parameters...
    #followed by Structure, the L's stucture will also change...
    """
    Implements Algorithm 1: Dynamic Programming for Planning.
    """
    ## Structure = [2,1] : which means, there is 2 encoder and 1 backbone
    ## now, we only consider there are two 
    ## now the Layer L is a list:
    # L = [(10,10),(30)] : tells that each partition's structure...

    #print(Layer_structure)
    top_k_container = TopKContainer(k)

    

    F = {}  # Stores best scores
    P = {}  # Stores best plans
    #now, the template:  [((phase, number),l,n,p )]

    #give initial value
    for phase in range(0,len(Structure)):
        for branch_num in range(Structure[phase]):
            L = Layer_structure[phase][branch_num]

            ## do not consider how many states...
            F[((phase, branch_num), 0)] = [float('inf')]
            P[((phase, branch_num), 0)] = [[]*s]            

            for p in range(0, min(L, N)):
                F[((phase, branch_num), 0, 0, p)] = [float('inf')]
                P[((phase, branch_num), 0, 0, p)] = [[]*s]

    for n in range(0, N+1):
        F[((-1, -1), n)] = [float('inf')]
        P[((-1, -1), n)] = [[]*s]                         

    for phase in reversed(range(0,len(Structure))):
        for branch_num in reversed(range(Structure[phase])):


            ## currently for this part, we don't consider multi encoders' assigned to single device...
            ## let's do plan searching for this phase...
            ## first, get the last state index:
            last_chunk_index = (-2,-2)
            if branch_num<Structure[phase]-1:
                last_chunk_index = (phase,branch_num+1)
            elif phase <len(Structure)-1:
                last_chunk_index = (phase+1, 0)
            else:
                last_chunk_index = (-1,-1)

            #last_phase_index = (0,0)

            #if phase >0:
            #    last_phase_index = (phase-1, Structure[phase]-1)
            #else:
            #    last_phase_index = (-1,-1)

            L = Layer_structure[phase][branch_num]

            for p in range(1, min(L, N+1, (M+1)//2)):
                for n in range(p, N + 1):
                    for l in range(1, L + 1):
                        # Initialize inner TopK container
                        top_s_container = TopKContainer(s)
                        #print((l,n,p))
                        for n_prime in range(p-1,n):
                            for l_prime in range(0,l):
                                
                                if n_prime == 0 and l_prime >0:
                                    continue #this means we rely on a special non-esixt sub problem
                                if n_prime*l_prime>0 and p-1 ==0:
                                    continue #stage 0's only afford that np and lp are both 0
                                if n_prime > 0 and l_prime == 0 and p-1 == 0: # inherently means p-1==0
                                    # it is the base statement of this phase index's
                                    # a single stage, and we need to concatinte the last phase's plans onto it...
                                    # original:Previous_Plan = P[((phase, branch_num), 0, 0, p-1)]
                                    Previous_Plan = P[(last_chunk_index, n_prime)]
                                    if Previous_Plan == [[]]:
                                        continue

                                elif n_prime == 0 and l_prime == 0 and p-1 == 0 and last_chunk_index == (-1, -1):
                                    Previous_Plan = P[(last_chunk_index, n_prime)]

                                elif n_prime == 0 and l_prime == 0 and last_chunk_index != (-1, -1):
                                    continue     
                                else:
                                    if ((phase, branch_num), l_prime, n_prime, p-1) not in P:
                                        continue
                                    Previous_Plan = P[((phase, branch_num), l_prime, n_prime, p-1)]
                                    if Previous_Plan == [[]]:
                                        continue

                                add_shard = {'phase':(phase,branch_num), 'layer':(L-l, L - l_prime), 'device':(N - n, N-n_prime), 'inver_internal_stage_idx':p-1}

                                for r in range(min(s, len(Previous_Plan))):
                                    # Generate plan and get score. plan has s lists, in which an element is a dict...
                                    #important here: we consider l and n are inversed counting

                                    subplan = Previous_Plan[r]
                                    #for all subplan, test if the device is OOM or not...
                                    if Profilelor.DP_solver(add_shard['phase'],add_shard['device'], add_shard['layer'], p-1, subplan) == False:
                                        #print("fail")
                                        continue                                    

                                    plan = [add_shard]+ subplan
                                    score, allocateplan = graph_plan_estimator((phase,branch_num),plan,M,SLO, Profilelor, alpha, jmode = jmode)
                                    # Update inner top-k
                                    top_s_container.update(score, plan, allocateplan)

                        # Update F and P
                        #print((phase, branch_num), l, n, p)
                        F[((phase, branch_num), l, n, p)] = top_s_container.scores
                        P[((phase, branch_num), l, n, p)] = top_s_container.plans

                    #check: only we assign over all layers, we update the top_k_container
                    #wrong check method...
                    #if (len(top_s_container.plans[0])!= L):
                    #    print(P)
                    #    print(top_s_container.plans[0])
                    #    print("something goes wrong... need to make sure we only monitor schedule with full layers")
                    # Update outer container
                    if phase == 0 and branch_num == 0:
                        for v, plan,aplan in zip(top_s_container.scores, top_s_container.plans, top_s_container.allocateplans):
                            top_k_container.update(v, plan, aplan)
                #print("stage amount"+str(p)+"finished.")
            ##
            ##Merge all plans regardless of the phase...
            for n in range(1, N + 1):
                F[((phase, branch_num), n)] = []
                P[((phase, branch_num), n)] = []
                for ph in range(1,min(L, n+1, (M+1)//2)):
                    
                    F[((phase, branch_num), n)] = F[((phase, branch_num), n)] + F[((phase, branch_num), L, n, ph)]
                    P[((phase, branch_num), n)] = P[((phase, branch_num), n)] + P[((phase, branch_num), L, n, ph)]


    #print(P[((0, 0), 10,1,1)])
    #print(P[((0, 1), 1)])
    #print(P[((0, 1), 2)])
    #print(P[((0, 1), 3)])
    #print(P[((0, 1), 4)])
    return top_k_container


if __name__ == "__main__":
    #test_Profilelor_DPsolver()
    p = dynamic_programming_planning_MM(Structure = [2,1], Layer_structure=[[10,15],[20]], N=4, M=10, k=3, s=1, Profilelor=None, alpha=0, SLO=0).plans
    print(p[0])