from profile_sim import Device, Profilelor
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

def generate_profiler_samples_nolimit(n = 4, c_level = [5, 3, 1, 2], e_level = [5, 3, 1, 2], l_limit = [15]*4, stage = 4, mbatch = 5):
    dList = []
    h=1024
    l=7*h**2
    s=4096
    for i in range(len(c_level)):
        onesecond_base_ability = l*mbatch*s*h #means the necessary ability to train 1 layer mbatch in 1 second
        oneconsumption_base_E = 1/(20*5*l)   #now the actually standard time latency is 20 seconds# currently ignore
        c = onesecond_base_ability/20*(c_level[i])  # /20 stands 20 seconds
        p = oneconsumption_base_E*(e_level[i])
        onelayermem = 1*l*1*s*h*mbatch # no consideration of buffered microbatch for pipeline method...
        dList.append(Device(c, p, onelayermem*l_limit[i], 10000000000))#can hold 30 layers model, big enough
    onesecond_base_band = h*s*mbatch  # 5 as a microbatch size...
    band = onesecond_base_band/12 # bandwidth contension mode:
    simprofile = Profilelor(dList,layerSize=l,hiddenSize=h, seq=s, MbatchSize = mbatch, Bandwidth = band)
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
    #nstage = 4
    ndevice = 6
    nmbatch = 4
    mbatchsize = 8
    layers = 30
    simprofile, band = generate_profiler_samples_nolimit(
        n = ndevice,
        c_level = [3, 2.5, 4, 2, 2], 
        e_level = [500, 400,200, 5, 5],
        l_limit = [80, 70, 40,40,20], 
        stage = 5, mbatch = mbatchsize)
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



def test_DP_solver_onlytime_prime(ks, ss):
    #nstage = 4
    ndevice = 6
    nmbatch = 7
    mbatchsize = 9
    layers = 12
    simprofile, band = generate_profiler_samples_nolimit(
        n = ndevice,
        c_level = [40, 30, 20, 10, 10,10], 
        e_level = [500, 400,200, 50, 50,50],
        l_limit = [20*6, 50*6, 15*5, 20*3, 15*2, 15*2], 
        stage = 5, mbatch = mbatchsize)
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
    test_DP_solver_onlytime(1)