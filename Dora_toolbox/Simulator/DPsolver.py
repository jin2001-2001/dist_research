from utils import TopKContainer, plan_estimator


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
                

    for p in range(1, min(L, N, (M+1)//2)):
        for n in range(p, N + 1):
            for l in range(n, L + 1):
                # Initialize inner TopK container
                top_s_container = TopKContainer(s)
                #print((l,n,p))
                for n_prime in range(p-1,n):
                    for l_prime in range(n_prime,l):
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
