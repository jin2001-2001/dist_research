from profile_real import Device, Profilelor, GraphProfilelor
from DPsolver import dynamic_programming_planning
import utils
from collections import Counter

def unique_permutations(iterable):
    """
    Yields unique permutations even when iterable has duplicate elements.
    Produces tuples; convert to list if you prefer.
    """
    counter = Counter(iterable)
    keys = list(counter)
    n = len(iterable)
    perm = []

    def backtrack():
        if len(perm) == n:
            yield tuple(perm)
            return
        for k in keys:
            if counter[k] > 0:
                counter[k] -= 1
                perm.append(k)
                yield from backtrack()
                perm.pop()
                counter[k] += 1

    yield from backtrack()



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
def generate_profiler_samples_nolimit(n=4,hidden_size = 0, seq = 256, layers = 28, type_list = ["cpu100"]*5,MbatchSize=4, profilehome="../../Profile", band = 100, mem_list = [100*1024]*5):
    dList = []
    for i in range(len(type_list)):
        # type, tprofile_loc, eprofile_loc = 0, Mem = 0, Energy = 1000)
        dList.append(Device(type_list[i], profilehome, layers, Mem = mem_list[i]))
    simprofile = Profilelor(dList,hiddenSize=hidden_size, seq=seq, total_layer= layers ,MbatchSize = MbatchSize, Bandwidth = band)
    return simprofile, band


def generate_profiler_samples_nolimit_MM(model_names=['x','x', 'x'],n=4,hidden_size = [0]*3, seq = [0]*3, layers = [0]*3, type_list = ["cpu100"]*5,MbatchSize=4, profilehome="../../Profile", band_str = None, mem_list = [100*1024]*5,map_dict={}):
    dList = []
    for i in range(len(type_list)):
        # type, tprofile_loc, eprofile_loc = 0, Mem = 0, Energy = 1000)
        pd_modellist = []
        for idx, model_name in enumerate(model_names):
            real_profile_name = str(type_list[i])+str(model_name)
            #print(idx, i, layers, mem_list)
            pd_modellist.append(Device(real_profile_name, profilehome, layers[idx], Mem = mem_list[i]))
            

        dList.append(pd_modellist)
    simprofile = GraphProfilelor(dList,hiddenSize=hidden_size, seq=seq, total_layer= layers ,MbatchSize = MbatchSize, band_str = band_str,map_dict=map_dict)
    return simprofile, 0



def test_Profilelor_DPsolver():  #a self defined examples...
    #for example, we have a 15 layer models:
    simprofile, band = generate_profiler_samples_nolimit()
    #plan1 = [{'layer':(0,26), 'device':(0,1)},{'layer':(26,28), 'device':(1,3)}]
    plan1 = [{'layer':(0,28), 'device':(0,1,2,3)}]
    plan1 = [{'layer':(0,7), 'device':(0,1)},{'layer':(7,14), 'device':(1,2)}, {'layer':(14,21), 'device':(2,3)},{'layer':(21,28), 'device':(3,4)}]
    #plan1 = [{'layer':(0,14), 'device':(0,1)},{'layer':(14,15), 'device':(1,3)},{'layer':(15,28), 'device':(3,4)}]
    #plan2 = [{'layer':(0,5), 'device':(0,1)}, {'layer':(5,15), 'device':(1,3)}]

    #sharded_batches = simprofile.DP_solver(plan1[0]['device'], plan1[0]['layer'], 0)
    simprofile, band = generate_profiler_samples_nolimit(
            n = 4,
            type_list = ["cpu60"]*0 + ["cpu60"]*0+["cpu30"]*4,  
            MbatchSize=5)
    #print(sharded_batches) ##pass!##
    B_ft, B_bt, B_fe, B_be, T_gathering, E_gathering, BatchAllocateList = simprofile.getall(plan1)
    #print(B_ft, B_bt, B_fe, B_be, T_gathering, E_gathering, BatchAllocateList)

    T = utils.plan_estimator(plan1, 20, 0, simprofile, 1)
    return T, B_ft, B_bt, T_gathering

def test_DP_solver_onlytime(ks, ss):
    ndevice = 4
    nmbatch = 20
    mbatchsize = 5
    layers = 28
    test_list = ["CPU60"]*4 + ["CPU30"]*2+["CPU100"]*0
    score_list = []
    plan_list = []
    allo_list = []
    device_order_list = []
    topK = utils.TopKContainer(ks)

    for device_order in unique_permutations(test_list):
        simprofile, band = generate_profiler_samples_nolimit(
            n = ndevice, layers = layers,
            type_list = device_order,  
            profilehome = "../../Profile_exp_0.7",
            MbatchSize=mbatchsize,
            band=200)
        #print("Communication" ,simprofile.communication_solver(10))
        #print("computation:", simprofile.DList[0].computeprofile.batchFuncforward(5), simprofile.DList[0].computeprofile.batchFuncbackward(5))

        result = dynamic_programming_planning(L = layers, N= ndevice , M = nmbatch, k = ks, s = ss,
                                          Profilelor = simprofile, 
                                          alpha = 1, SLO = 0)
        topK.merge_with_device_type(result, device_order)
        

    for j in range(len(topK.scores)):
        score_list.append(topK.scores[j])
        plan_list.append(topK.plans[j])
        allo_list.append(topK.allocateplans[j])
        device_order_list.append(topK.device_orders[j])

    return (layers,ndevice, nmbatch,mbatchsize, band), score_list, plan_list, allo_list,device_order_list, simprofile


if __name__ == "__main__":
    #test_Profilelor_DPsolver()
    print(test_Profilelor_DPsolver())


    meta, score_L, plan_L, allo_L,d_L, profile = test_DP_solver_onlytime(5,1)
    print(score_L, plan_L, allo_L)
    print(d_L)
