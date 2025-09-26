from profile_real import Device, Profilelor
from DPsolver import dynamic_programming_planning
import utils

def test_heap_func():
    print("heap test:")
    c = utils.TopKContainer(3)
    c.update(5, "A")
    c.update(8, "B")
    c.update(2, "C")
    c.update(7, "D")
    c.update(3, "E")
    print("Heap:", c.heap)
    print("Pairs:", c.pairs)
    print("Scores:", c.scores)
    assert 1 == 1

#For Qwen3:
def generate_profiler_samples_nolimit(n=4,type_list = ["cpu100"]*5,MbatchSize=4, profilehome="../../Profile", band = 100):
    dList = []
    for i in range(len(type_list)):
        # type, tprofile_loc, eprofile_loc = 0, Mem = 0, Energy = 1000)
        dList.append(Device(type_list[i], profilehome, Mem = 48*1024))#can hold 30 layers model, big enough
    simprofile = Profilelor(dList,hiddenSize=1024, seq=256, MbatchSize = MbatchSize, Bandwidth = band)
    return simprofile, band

def test_Profilelor_DPsolver():  #a self defined examples...
    #for example, we have a 15 layer models:
    simprofile, band = generate_profiler_samples_nolimit()
    plan1 = [{'layer':(0,15), 'device':(0,3)}]
    plan2 = [{'layer':(0,5), 'device':(0,1)}, {'layer':(5,15), 'device':(1,3)}]

    sharded_batches = simprofile.DP_solver(plan1[0]['device'], plan1[0]['layer'], 0)
    print(sharded_batches) ##pass!##
    B_ft, B_bt, B_fe, B_be, T_gathering, E_gathering, BatchAllocateList = simprofile.getall(plan2)
    print(B_ft, B_bt, B_fe, B_be, T_gathering, E_gathering, BatchAllocateList)

def test_DP_solver_onlytime(ks, ss):
    ndevice = 4
    nmbatch = 4
    mbatchsize = 8
    layers = 30
    simprofile, band = generate_profiler_samples_nolimit(
        n = ndevice, MbatchSize=mbatchsize)
    score_list = []
    plan_list = []
    allo_list = []

    for i in range(1, ss+1):
        result = dynamic_programming_planning(L = layers, N= ndevice , M = nmbatch, k = ks, s = i,
                                          Profilelor = simprofile, 
                                          alpha = 1, SLO = 0)
        for j in range(len(result.scores)):
            score_list.append(result.scores[j])
            plan_list.append(result.plans[j])
            allo_list.append(result.allocateplans[j])
    return (layers,ndevice, nmbatch,mbatchsize, band), score_list, plan_list, allo_list, simprofile


if __name__ == "__main__":
    #test_Profilelor_DPsolver()
    test_DP_solver_onlytime(1,1)