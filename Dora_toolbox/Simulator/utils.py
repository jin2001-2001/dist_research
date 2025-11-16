import heapq
import math
import itertools
from collections import defaultdict
from itertools import product

class TopKContainer:
    """
    Maintains the k smallest (score, plan) pairs.
    """
    def __init__(self, k):
        self.k = k
        self.heap = []  # This is a max-heap: (-score, plan)
        self.id = 0
    def amount(self):
        return len(self.heap)


    def update(self, score, plan, allocateplan, device_order=None):
        """
        Add a (score, plan) pair.
        """
        entry = (-score, self.id, plan, allocateplan, device_order)



        if len(self.heap) < self.k:
            self.id+=1
            heapq.heappush(self.heap, entry)
        else:
            # If the new score is smaller than the largest (worst) in heap
            if score < - self.heap[0][0]:
                # Replace the worst one
                self.id+=1
                #print(self.heap)
                #print(entry)
                heapq.heapreplace(self.heap, entry)
    
    def merge(self, container):
        for j in range(len(container.scores)):
            self.update(container.scores[j], container.plans[j], container.allocateplans[j])

    def merge_with_device_type(self, container, device_order): 
        for j in range(len(container.scores)):
            self.update(container.scores[j], container.plans[j], container.allocateplans[j], device_order)


    @property
    def pairs(self):
        """
        Returns list of (score, plan) pairs sorted by score ascending.
        """
        entries = [(-s, i, p,pp,ppp) for (s, i, p,pp,ppp) in self.heap]
        sorted_entries = sorted(entries, key=lambda x: (x[0], x[1]))
        return [(score, plan, allocateplan, device_order) for (score, _, plan, allocateplan, device_order) in sorted_entries]

    @property
    def scores(self):
        """
        Returns list of scores sorted ascending.
        """
        return [score for (score, _, _,_) in self.pairs]

    @property
    def plans(self):
        """
        Returns list of plans sorted ascending by score.
        """
        return [plan for (_, plan, _,_) in self.pairs]
    
    @property
    def allocateplans(self):
        """
        Returns list of plans sorted ascending by score.
        """
        return [allocateplan for (_, _, allocateplan,_) in self.pairs]

    @property
    def device_orders(self):
        """
        Returns list of plans sorted ascending by score.
        """
        return [device_order for (_, _, _, device_order) in self.pairs]

def plan_estimator(P, M, SLO, Profilelor, alpha):
    """
    Implements Algorithm 2: Plan Estimator.
    """


    # Initialize task lists
    #B_list = []  # Your logic to create B_list
    #M = Profilelor.M  #XXXXX ERROR
    #SLO_latency = Profilelor.slo_T #XXXXXXX ERROR
    if Profilelor.getall(P) == False:
        print("error")
    B_ft, B_bt, B_fe, B_be, T_gathering, E_gathering, BatchAllocateList = Profilelor.getall(P) 
    B_list = [B_ft, B_bt]
    #print(B_list)
    #length of B list should be S.

    S = 2 * len(P) - 1 #including the 

    # Find bottleneck step index
    d = max(range(S), key=lambda i: (B_ft[i] + B_bt[i],i)) #if the same big, choose one with hight step idx.

    # Compute phase times
    T1 = start_phase_time_est(B_ft, B_list, d)
    T2 = (M - S + d) * (B_ft[d] + B_bt[d])
    T3_list = end_phase_time_est(B_ft, B_list, d)

    T_latency = float('-inf')

    #print(B_list)
    #print(T3_list)

    for i in range(S):
        T = T1 + T2 + T3_list[i]
        if i % 2 == 0 and P[i // 2]['device'][1] > P[i // 2]['device'][0]+1: #so we know this line is mapped to a device group
            T += T_gathering[i]

        if T > T_latency:
            T_latency = T

    # Sum energy
    E_consumption = sum(
        M * (B_fe[i] + B_be[i]) + E_gathering[i]
        for i in range(S)
    )

    return abs(T_latency-SLO)+95e-6*(E_consumption**alpha), BatchAllocateList




def plan_parser(P):

    #example: L = [(0,0),(0,0),(0,1),(0,1),(1,0),(1,0),(1,0),(1,1),(1,1),(2,0),(2,1)]
    L = []
    for shard in P:
        L.append(shard['phase'])

    #print(L)
    # Step 1: Group indices by phase and index type
    phase_to_indices = defaultdict(lambda: defaultdict(list))
    for idx, (phase, branch) in enumerate(L):
        phase_to_indices[phase][branch].append(idx)

    # Step 2: Generate all possible "branch choices" per phase

    all_phases = sorted(phase_to_indices.keys())
    choices_per_phase = [list(phase_to_indices[p].keys()) for p in all_phases]

    # Step 3: For each branch combination (one per phase), form the chain
    chains = []
    for choice_tuple in product(*choices_per_phase):
        chain_indices = []
        for phase, branch in zip(all_phases, choice_tuple):
            chain_indices.extend(phase_to_indices[phase][branch])
        chains.append(sorted(chain_indices))

    return chains



def graph_plan_estimator(current_phase_index ,P, M, SLO, Profilelor, alpha):
    """
    Implements Algorithm 2: Plan Estimator.
    """
    ## OK, now we need to generate a 

    # Initialize task lists
    #B_list = []  # Your logic to create B_list
    #M = Profilelor.M  #XXXXX ERROR
    #SLO_latency = Profilelor.slo_T #XXXXXXX ERROR

    
    
    try:
        B_ft_a, B_bt_a, B_fe_a, B_be_a, T_gathering_a, E_gathering_a, BatchAllocateList_a = Profilelor.getall(P) 

    except Exception as e:
        # Code that runs *only* if an error occurs
        print("❌ Error caught:", e)
        print("profilelor getall: error")   
        return -len(P), []


    # noticed that dependency don't count communication index, i,.e. len of dependency *2 - 1 equal to len of B_f/b
    chain_list = plan_parser(P)

    T_max = -1
    BatchAllocateList = BatchAllocateList_a
    for per_chain in chain_list:
        ##construct the index list:
        actual_index_l = []
        for i in per_chain:
            if i == per_chain[-1]: # the last state of chain...
                actual_index_l = actual_index_l + [i*2]          ##for last part of the sharding, there is no commnuication...
            else:
                actual_index_l = actual_index_l + [i*2, i*2+1]   ##we need to also consider the communication part...


        B_ft = [B_ft_a[i] for i in actual_index_l]
        B_bt = [B_bt_a[i] for i in actual_index_l]
        T_gathering = [T_gathering_a[i] for i in actual_index_l]




        B_list = [B_ft, B_bt]

        #print(B_list)

        S = len(actual_index_l) 

        # Find bottleneck step index
        d = max(range(S), key=lambda i: (B_ft[i] + B_bt[i],i)) #if the same big, choose one with hight step idx.

        # Compute phase times
        T1 = start_phase_time_est(B_ft, B_list, d)
        T2 = (M - S + d) * (B_ft[d] + B_bt[d])
        T3_list = end_phase_time_est(B_ft, B_list, d)

        T_latency = float('-inf')

        for i in range(S):
            T = T1 + T2 + T3_list[i]
            if i % 2 == 0 and P[i // 2]['device'][1] > P[i // 2]['device'][0]+1: #so we know this line is mapped to a device group
                T += T_gathering[i]

            if T > T_latency:
                T_latency = T

        if T_latency>T_max:
            T_max = T_latency

    # Sum energy
    E_consumption = sum(
        M * (B_fe_a[i] + B_be_a[i]) + E_gathering_a[i]
        for i in range(S)
    )

    return abs(T_max-SLO)+95e-6*(E_consumption**alpha), BatchAllocateList






def start_phase_time_est(P, B_list, d):
    """
    Implements StartPhaseTimeEst.
    """
    B_ft, B_bt = B_list
    S = len(P)
    CritiPathT = 0

    for p in range(d, S):
        CurrPathT = 0

        for i in range(p+1): #necessary sub_path
            CurrPathT += B_ft[i]
        
        CurrPathT +=(S - (p+1)) * max(B_ft[0:p+1]) #rest for max blks 

        for i in range(d+1, p+1):
            CurrPathT += B_bt[i]

        if CurrPathT > CritiPathT:
            CritiPathT = CurrPathT

    return CritiPathT


def end_phase_time_est(P, B_list, d):
    """
    Implements EndPhaseTimeEst.
    """
    B_ft, B_bt = B_list
    S = len(P)
    CritiPathTList = []

    for s in range(S):
        CritiPathT = 0
        for p in range(max(s, d), S):
            CurrPathT = 0
            CurrPathT += ((S-s) - (p+1-s)) * max(B_bt[s:p+1]) #first get maximum blks
            for i in range(s, p+1):
                CurrPathT += B_bt[i]           
            for i in range(d + 1, p+1):
                CurrPathT += B_ft[i]
            if CurrPathT > CritiPathT:
                CritiPathT = CurrPathT
        CritiPathTList.append(CritiPathT)

    return CritiPathTList

if __name__ == "__main__":


    exampleP = [{'phase': (0, 0), 'layer': (0, 9), 'device': (0, 1)}, {'phase': (0, 0), 'layer': (9, 10), 'device': (1, 2)}, {'phase': (0, 1), 'layer': (0, 15), 'device': (2, 3)}, {'phase': (1, 0), 'layer': (0, 20), 'device': (3, 4)}]

    print(plan_parser(exampleP))