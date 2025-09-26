import heapq
import math

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

    def update(self, score, plan, allocateplan):
        """
        Add a (score, plan) pair.
        """
        entry = (-score, self.id, plan, allocateplan)

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

    @property
    def pairs(self):
        """
        Returns list of (score, plan) pairs sorted by score ascending.
        """
        entries = [(-s, i, p,pp) for (s, i, p,pp) in self.heap]
        sorted_entries = sorted(entries, key=lambda x: (x[0], x[1]))
        return [(score, plan, allocateplan) for (score, _, plan, allocateplan) in sorted_entries]

    @property
    def scores(self):
        """
        Returns list of scores sorted ascending.
        """
        return [score for (score, _, _) in self.pairs]

    @property
    def plans(self):
        """
        Returns list of plans sorted ascending by score.
        """
        return [plan for (_, plan, _) in self.pairs]
    
    @property
    def allocateplans(self):
        """
        Returns list of plans sorted ascending by score.
        """
        return [allocateplan for (_, _, allocateplan) in self.pairs]

def plan_estimator(P, M, SLO, Profilelor, alpha):
    """
    Implements Algorithm 2: Plan Estimator.
    """


    # Initialize task lists
    #B_list = []  # Your logic to create B_list
    #M = Profilelor.M  #XXXXX ERROR
    #SLO_latency = Profilelor.slo_T #XXXXXXX ERROR
    B_ft, B_bt, B_fe, B_be, T_gathering, E_gathering, BatchAllocateList = Profilelor.getall(P) 
    B_list = [B_ft, B_bt]
    #length of B list should be S.

    S = 2 * len(P) - 1 #including the 

    # Find bottleneck step index
    d = max(range(S), key=lambda i: (B_ft[i] + B_bt[i],i)) #if the same big, choose one with hight step idx.

    # Compute phase times
    T1 = start_phase_time_est(P, B_list, d)
    T2 = (M - S + d) * (B_ft[d] + B_bt[d])
    T3_list = end_phase_time_est(P, B_list, d)

    T_latency = float('-inf')

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

    return alpha*abs(T_latency-SLO) + (1-alpha) * E_consumption, BatchAllocateList

def start_phase_time_est(P, B_list, d):
    """
    Implements StartPhaseTimeEst.
    """
    B_ft, B_bt = B_list
    S = 2 * len(P) - 1
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
    S = 2 * len(P) - 1
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
    c = TopKContainer(3)
    c.update(5, "A")
    c.update(8, "B")
    c.update(2, "C")
    c.update(7, "D")
    c.update(3, "E")

    print("Heap:", c.heap)
    print("Pairs:", c.pairs)
    print("Scores:", c.scores)