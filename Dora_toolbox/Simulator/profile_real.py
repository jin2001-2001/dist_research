
import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np

Profile_Energy_checklist ={
    "9600x":8,                          #65W
    "4050":8*1.53, "4050INT8":8*1.53,             #100W
    "4060":8*1.76, "4060INT8":8*1.76,             #115W
    "Xiaomi":8*0.14, "XiaomiINT8":8*0.14,         #9.3W
    "Samsung":8*0.136, "SamsungINT8":8*0.136,                 #8.9W
    "Camera":8*0.153, "CameraINT8":8*0.153,             #10W
    "A40":8*1.53*3, "A40INT8":8*1.53*3,               # 300W
    "V100":8*1.53*2.5, "V100INT8":8*1.53*2.5,             # 250W
}



#0, 5%, 10%,... 100%
# so, int(0.7/0.05)
util_to_power_ratio = [
    0.000,
    0.000,
    0.000,
    0.000,
    0.030,
    0.071,
    0.143,
    0.262,
    0.357,
    0.500,
    0.595,
    0.679,
    0.750,
    0.810,
    0.857,
    0.905,
    0.952,
    0.976,
    0.988,
    0.994,
    1.000
]

def util_to_ratio(util):
    index = int(util//0.05)
    rest = util%0.05

    value_base = util_to_power_ratio[index]
    value_next = 1
    if index <len(util_to_power_ratio)-1:
        value_next = util_to_power_ratio[index+1]
    
    return value_base + rest*(value_next-value_base)






##profile is based on: for a fixed model...
##first, we need time/energy cost for Batch size, layer amounts...
##so, the input can be two elements: B and L
##first, input should be the name of the device, profile location, maximun batch/computation profiled...

## we use ax**2 + bx + c to simulate a time cost under batches  for a fixed number of layers
## we need extract out important things...
## 1. forward & backward time cost for base situation: 1 batches, for a fixed number of layers
## how to get more 
def syn_func(original_fun, tlist):
    def new_func(x):
        l = len(tlist)
        if x <=l:
            return original_fun(x)
        else:
            diff = tlist[-1] - tlist[-2]
            return tlist[-1] + diff*(x-l)
    return new_func

def fit_cubic_from_series(y_series):
    """
    y_series: list/array of y-values at x=1,2,3,...
    returns: p(x) callable (numpy.poly1d)
    """
    y = np.asarray(y_series, dtype=float)
    n = y.size
    if n < 2:
        raise ValueError("Need at least 2 points.")
    x = np.arange(1, n + 1, dtype=float)
    deg = min(5, n - 1)  # degrade if not enough points
    coefs = np.polyfit(x, y, deg=deg)  # highest power first
    return np.poly1d(coefs)

@dataclass
class ComputProfile:
    forward: float
    backward: float
    batch_base: int  #unused...
    layer_base: int
    batchFuncforward: object
    batchFuncbackward: object

class Device:
    def __init__(self, type, tprofile_loc, actual_layers, eprofile_loc = 0, Mem = 0, util = 1, Energy = 1000, jmode = None,omni_sim = None, linear_map = True):
        self.type = type
        self.Eability = 0
        self.Mconstraint = Mem  # unit: MB
        self.Econstraint = Energy
        self.activation_bytes_per_sample = 0
        self.activation_bytes_seq_len = 0
        self.layer_param_bytes = 0
        self.embedding_param_bytes = 0
        self.tail_param_bytes = 0
        self.total_layers = actual_layers
        self.jmode = jmode
        self.omni_sim =omni_sim
        self.util = util
        self.linear_map = linear_map
        #self.total_parameters_bytes = 0

        root = Path(tprofile_loc)
        self.root = tprofile_loc
        batch = 1
        total_profile_layers = 0
        path = root / f"{type}_bs{batch}.json"
        if not path.exists():
            print(path)
            raise FileNotFoundError("Could not find the basic config file")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)  # parses JSON -> Python objects
            forward = 0
            backward = 0
            tailforward = data["head&tail"][1]["forward_time_s"]/util
            tailbackward = data["head&tail"][1]["backward_time_s"]/util
            headforward = data["head&tail"][0]["forward_time_s"]/util
            headbackward = data["head&tail"][0]["backward_time_s"]/util
            self.activation_bytes_seq_len = data["seq_len"]
            self.activation_bytes_per_sample = data["layers"][0]["activation_bytes_per_sample"]
            self.layer_param_bytes = data["layers"][0]["param_bytes"]
            self.embedding_param_bytes = data["embed_param_bytes"]
            self.tail_param_bytes = data["tail_param_bytes"]
            #self.total_parameters_bytes = data["total_parameters_bytes"]
            total_profile_layers = len(data["layers"])
            for i in range(total_profile_layers):
                forward+=data["layers"][i]["forward_time_s"]/util
                backward+=data["layers"][i]["backward_time_s"]/util
            forward =forward/total_profile_layers
            backward=backward/total_profile_layers
        
        onebforward = forward
        onebbackward = backward
        batch = 2
        outforward = []
        outbackward = []
        headoutforward = []
        headoutbackward = []
        tailoutforward = []
        tailoutbackward = []
        while True:
            path = root / f"{type}_bs{batch}.json"
            if not path.exists():
                break  # stop once a batch file is missing
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                forward = 0
                backward = 0
                self.total_layers = total_profile_layers
                for i in range(total_profile_layers):
                    forward+=data["layers"][i]["forward_time_s"]/util
                    backward+=data["layers"][i]["backward_time_s"]/util
                forward =forward/total_profile_layers
                backward=backward/total_profile_layers
                outforward.append(forward)
                outbackward.append(backward)

                tf = data["head&tail"][1]["forward_time_s"]/util
                tb = data["head&tail"][1]["backward_time_s"]/util
                hf = data["head&tail"][0]["forward_time_s"]/util
                hb = data["head&tail"][0]["backward_time_s"]/util
                headoutforward.append(hf)
                headoutbackward.append(hb)
                tailoutforward.append(tf)
                tailoutbackward.append(tb)
            batch += 1
        
        outforward = [onebforward]+outforward
        outbackward = [onebbackward]+outbackward
        headoutforward = [headforward]+headoutforward
        headoutbackward = [headbackward]+headoutbackward
        tailoutforward = [tailforward]+tailoutforward
        tailoutbackward = [tailbackward]+tailoutbackward
        #print(outforward)
        #print(outbackward)

        a = fit_cubic_from_series(outforward)
        a = syn_func(a,outforward)
        b = fit_cubic_from_series(outbackward)
        b = syn_func(b,outbackward)
        self.computeprofile = ComputProfile(forward,backward,
                                            1,1,a,b)

        a = fit_cubic_from_series(headoutforward)
        a = syn_func(a,headoutforward )
        b = fit_cubic_from_series(headoutbackward)
        b = syn_func(b, headoutbackward)
        self.computeprofile_h = ComputProfile(headforward,headbackward,
                                            1,0,a,b)
        a = fit_cubic_from_series(tailoutforward)
        a = syn_func(a, tailoutforward)
        b = fit_cubic_from_series(tailoutbackward)
        b = syn_func(b, tailoutbackward)
        self.computeprofile_t = ComputProfile(tailforward,tailbackward,
                                            1,0,a,b)


    def util_map(self, layers):
        half = self.util/2
        percent= layers/self.total_layers
        bonus_util = percent*half/2
        final_util = half+bonus_util
        if final_util<0.2:
            final_util = 0.2
        return final_util, final_util/self.util


    def Tlatency(self, layer_slice, batch_size,seq, hidden):
        layers = layer_slice[1]-layer_slice[0]
        
        ratio = 1
        if self.omni_sim == 1:
            ratio = seq*hidden/(2048*256)

        # 20% --- 100%
        futil, the_util_r = self.util_map(layers)

        if self.linear_map == False:
            the_util_r = 1        

        Tf = self.computeprofile.batchFuncforward(batch_size)*layers/self.computeprofile.layer_base       *ratio
        Tb = self.computeprofile.batchFuncbackward(batch_size)*layers/self.computeprofile.layer_base      *ratio
        if layer_slice[0] == 0:
            Tf+=self.computeprofile_h.batchFuncforward(batch_size)
            Tb+=self.computeprofile_h.batchFuncbackward(batch_size)
        if layer_slice[1] == self.total_layers:
            Tf+=self.computeprofile_t.batchFuncforward(batch_size)
            Tb+=self.computeprofile_t.batchFuncbackward(batch_size)


        #print(batch_size,Tf,Tb)
        return Tf/the_util_r,Tb/the_util_r
    
    def Econsump(self, layer_slice, layers_size, batch_size,seq,hidden):
        T1, T2 = self.Tlatency(layer_slice, batch_size,seq,hidden)
        coff = (1/self.computeprofile.forward)**1.5
        E =(T1+T2)*coff
        return E
    def Econsumpf(self, layer_slice, layers_size, batch_size,seq,hidden):
        f_util, _ = self.util_map(layer_slice[1]-layer_slice[0])
        if self.linear_map == False:
            f_util = self.util

        T1, T2 = self.Tlatency(layer_slice, batch_size,seq,hidden)
        E =T1*util_to_ratio(f_util)*Profile_Energy_checklist[str(self.type)]
        return E
    def Econsumpb(self, layer_slice, layers_size, batch_size,seq,hidden):
        f_util, _ = self.util_map(layer_slice[1]-layer_slice[0])
        if self.linear_map == False:
            f_util = self.util
        T1, T2 = self.Tlatency(layer_slice, batch_size,seq,hidden)
        E =T2*util_to_ratio(f_util)*Profile_Energy_checklist[str(self.type)]
        return E
    def maximum_batches_available(self, layer_slice, inversestage,seq,hidden): #reversely get the maximum batches that can be assigned on this devices
        layers = layer_slice[1]-layer_slice[0]
        per_b_s = 2
        if "bert" in str(self.root):
            per_b_s = 4


        #batch_act_storage = 6*seq*hidden*2 *layers *(inversestage*2+1)   #no batch, we need calculate batches...#4 is fp16 needs 2 bytes
        if self.jmode == "training":
            batch_act_storage = 6*seq*hidden*per_b_s *layers *(inversestage*1+1) #2 is fp16 needs 2 bytes
        else:
            batch_act_storage = 6*seq*hidden*per_b_s * 1 *(1) #consider only one layer's peak...

        parameter_storage = self.layer_param_bytes * layers *3 #(1 for opt, 1 for gradient)


        if layer_slice[1] == self.total_layers:
            parameter_storage+= self.tail_param_bytes*per_b_s#f16
        if layer_slice[0] == 0:
            parameter_storage+= self.embedding_param_bytes*per_b_s#f16

        #print(layer_slice, (parameter_storage+batch_act_storage*8)/1024/1024/1024)
        max1 = (self.Mconstraint*1024*1024-parameter_storage)/(batch_act_storage)
        max2 = 1024*512
        #max2 = self.Econstraint * self.Tability/(self.Eability*(layers**2)*(layers_size**2))/(seq*hidden)
        #print(max1, max2)
        return min(max1, max2)



class Profilelor:
    """
    Maintains the k smallest (score, plan) pairs.
    """
    def __init__(self,DList,hiddenSize, seq, total_layer, MbatchSize, Bandwidth):
        #self.deviceN = damount
        self.DList = DList
        self.hiddenSize = hiddenSize
        self.seq_len = seq
        self.MbatchSize = MbatchSize
        self.bandwidth = Bandwidth
        self.total_layers = total_layer


    def communication_solver(self, layer_slice=1): #simple version
        #T = bsize*self.hiddenSize*self.seq_len/(self.bandwidth)
        T = self.DList[0].activation_bytes_per_sample * self.MbatchSize/1e6/self.bandwidth*8
        #print(T)
        return T

    def gathering_solver(self, device_slice, layer_slice): #simple version
        D_amount = device_slice[1]-device_slice[0]
        l_amount = layer_slice[1]-layer_slice[0]

        accum= l_amount*self.DList[0].layer_param_bytes

        if layer_slice[0] == 0:
            accum += self.DList[0].embedding_param_bytes
        if layer_slice[1] == self.total_layers:
            accum += self.DList[0].tail_param_bytes

        total_parameter_bytes = accum*(D_amount-1)/D_amount *2 #plus embedding estimiation

        T = total_parameter_bytes*8/1e6/(self.bandwidth/D_amount)

        return T


    def DP_solver(self, device_slice, layer_slice, inverseStage):
        #print(device_slice,layer_slice)
        layers = layer_slice[1]-layer_slice[0]
        c_list = [1/self.DList[i].computeprofile.forward for i in range(device_slice[0],device_slice[1])]
        b_list = [int(self.DList[i].maximum_batches_available(layer_slice, inverseStage,self.seq_len,self.hiddenSize)) for i in range(device_slice[0],device_slice[1])] #memeory bound but tranfer to upperbound of batches
        if sum(b_list)<self.MbatchSize:              # layers, inversestage,seq,hidden
            #print(b_list, self.MbatchSize)
            return False ##cannot reach the batch assignment...
        if len(b_list)>self.MbatchSize:
            return False
        ###  "Largest Remainder Method" to distribute the batch in average...###
        n = len(b_list)
        c_sum = sum(c_list)
        ideal = [c * (self.MbatchSize) / c_sum for c in c_list]
        floored = [max(min(int(x), b_list[i]),1) for i, x in enumerate(ideal)]
        assigned = sum(floored)
        remainder = self.MbatchSize - assigned

        fractional = [(i, ideal[i] - int(ideal[i])) for i in range(n)]
        # Sort descending by fractional part
        fractional.sort(key=lambda x: -x[1])

        if remainder<0: # that is, too many necessary 1's, so we have to discard some value from 
            while remainder<0:
                fractional.sort(key=lambda x: x[1])
                for i, frac in fractional:
                    if floored[i] >1:
                        floored[i]-=1
                        remainder+=1

        # Distribute remaining units
        for i, frac in fractional:
            if remainder == 0:
                break
            gap = b_list[i] - floored[i]
            if gap > 0:
                assign = min(gap, remainder)
                floored[i] += assign
                remainder -= assign
        return floored



    def getall(self, P):
        stepsN = 2*len(P)-1
        B_ft, B_bt, B_fe, B_be, T_gathering, E_gathering = ([] for _ in range(6))
        BatchAllocateList = []
        for s in range(stepsN):
            if s%2 == 0: # calculation step:
                stepinfo = P[s//2]
                layer_slice = stepinfo['layer']
                device_slice = stepinfo['device']
                #for each loops, use DP solver to get the answer,
                inverseStage = len(P) -1 - s//2    # it should be the inverse stage index
                batch_shard = self.DP_solver(device_slice, layer_slice, inverseStage)
                if batch_shard == False:
                    return False #failed to allocate batch for one Devices group
                BatchAllocateList.append(batch_shard)
                #calculate the time latency for the device group...
                layers = layer_slice[1]- layer_slice[0]
                devices = device_slice[1] - device_slice[0]
                latencyf = 0
                latencyb = 0
                index = 0
                f_energy = 0
                b_energy = 0

                for i in range(device_slice[0],device_slice[1]):
                    d = self.DList[i]
                    b = batch_shard[index]
                    lf,lb = d.Tlatency(layer_slice, b,self.seq_len,self.hiddenSize)
                    f_energy+=d.Econsumpf(layer_slice, self.DList[0].layer_param_bytes, b,self.seq_len,self.hiddenSize)
                    b_energy+=d.Econsumpb(layer_slice, self.DList[0].layer_param_bytes, b,self.seq_len,self.hiddenSize)
                    if lf>latencyf:
                        latencyf = lf
                    if lb>latencyb:
                        latencyb = lb
                    index+=1
                #now, we get latency of the device group

                ###jin important: we multiply a constant to estimate the backward cost
                B_ft.append(latencyf)
                B_bt.append(latencyb)
                B_fe.append(f_energy)
                B_be.append(b_energy)
                tt=0
                ee=0
                ###jin important: we gathering actually only involve transmission energy cost, it is little...
                if (devices > 1):
                    #print("a DP group")
                    tt = self.gathering_solver(device_slice, layer_slice)
                T_gathering.append(tt)
                E_gathering.append(ee)
            else: # communication step:
                T = self.communication_solver(self.MbatchSize)
                B_ft.append(T)
                B_bt.append(T)
                B_fe.append(0)
                B_be.append(0)                
                T_gathering.append(0)
                E_gathering.append(0)                   




        return B_ft, B_bt, B_fe, B_be, T_gathering, E_gathering, BatchAllocateList

###########
###########
###########
#graph version:


###helper func...
def stage_helper(phase_index, the_whole_plan):
    curphase = phase_index[0]
    maping = {}
    #print(the_whole_plan)
    for shard in the_whole_plan:
        ph_idx = shard['phase']
        if ph_idx[0]>curphase:
            if ph_idx in maping:
                maping[ph_idx]+=1
            else:
                maping[ph_idx] = 1
    # the mapping stores all phase_idx tuples' internal shard numbers...
    maping1 = {}
    for atuple in maping:
        phase = atuple[0]
        amount = maping[atuple]
        if phase in maping1:
            if amount> maping1[phase]:
                maping1[phase] = amount
        else:
            maping1[phase] = amount
    
    return sum(maping1.values())

class GraphProfilelor:
    """
    Maintains the k smallest (score, plan) pairs.
    """
    def __init__(self,DList,hiddenSize, seq, total_layer, MbatchSize, band_str, map_dict, jmode):
        #self.deviceN = damount
        self.DList = DList      #per ele should be a list of Devices, and use map_dict to get the correct devices...
        self.hiddenSize = hiddenSize # the same, is a list 
        self.seq_len = seq    # list 
        self.MbatchSize = MbatchSize # not list 
        self.band_str = band_str # not list 
        self.total_layers = total_layer # list...
        self.phase_mapping = map_dict
        self.jmode = jmode

    def dependency_list(self, plan):   ## the same one in O2_omni,,, if we need to do modification, we need change omni on as well

        L = []
        biggest_phase = 0
        for shard in plan:
            index = shard['phase']
            L.append(index)
            if shard !=plan[-1]:
                L.append(index)
            if index[0]>=biggest_phase:
                biggest_phase = index[0]
        #print(L)
        #L = [(0,0), (0,0), (0,1), (0,1), (1,0), (1,0)]

        n = len(L)
        deps = [-1] * n  # initialize all with -1

        # 1Rule 1: identical neighbors
        for i in range(n - 1):
            if L[i] == L[i + 1]:
                deps[i] = i + 1
        if   biggest_phase>0:
            # Rule 2: last (0,*) → first (1,0)
            #print(L)
            first_1_idx = next(i for i, t in enumerate(L) if t == (1,0))
            # find last index of each unique (0, y)
            seen = set()
            for i in range(n - 1, -1, -1):
                if L[i][0] == 0 and L[i][1] not in seen:
                    deps[i] = first_1_idx
                    seen.add(L[i][1])
        if biggest_phase>1:
            for j in range(biggest_phase-1):
                lower = 1+j
                higher = lower +1
                first_higher_idx= next(i for i, t in enumerate(L) if t == (higher,0))
                deps[first_higher_idx-1] = first_higher_idx
            
        #print("recodrded for dp_list gen:", plan, deps, n,L)
        return(deps)


    
    def communication_solver(self,phase_index, mode="shared", device_slice= None, next_device_slice = None ): #simple version
        index = self.phase_mapping[phase_index]

        if mode == "shared":
        #bw = self.band_str.available_bw(device_slice_from[0], device_slice_from[-1]+1)
            bw =  self.band_str.shardband
        else: 
            bw, _ = self.band_str.available_bw(device_slice[0], next_device_slice[0]) 


        per_b_s = 4
        if "bert" in str(self.DList[0][0].root):
            per_b_s = 4

        T = self.MbatchSize*self.hiddenSize[index]*per_b_s*self.seq_len[index]/(bw)/1e6*8                    # consider fp16
    


        #T = self.DList[0][index].activation_bytes_per_sample * self.MbatchSize/1e6/self.bandwidth*8
        #print(T)
        return T

    def gathering_solver(self,phase_index,mode="shared", device_slice = None, layer_slice = None): #simple version

        index = self.phase_mapping[phase_index]

        D_amount = device_slice[1]-device_slice[0]
        l_amount = layer_slice[1]-layer_slice[0]

        accum= l_amount*self.DList[0][index].layer_param_bytes

        if layer_slice[0] == 0:
            accum += self.DList[0][index].embedding_param_bytes
        if layer_slice[1] == self.total_layers[index]:
            accum += self.DList[0][index].tail_param_bytes

        total_parameter_bytes = accum*(D_amount-1)/D_amount *2 #plus embedding estimiation


        local_lan = self.band_str.available_group_lan_bw(list(range(device_slice[0], device_slice[1])))
        #local_lan = 0
        bb = self.band_str.shardband
        if bb<=50 and mode != "shared":
            bb = 0 ## goes into don't care shared BW mode
        
        if mode == "shared":
            local_lan = 0

        #print(bb, D_amount)

        per_b_s = 2
        if "bert" in str(self.DList[0][0].root):
            per_b_s = 4

        T = total_parameter_bytes*per_b_s/1e6/(bb/D_amount+local_lan)                 # consider fp16

        #print((1.7*10**9/2)*per_b_s/1e6/(bb/3+local_lan))
        return T





    def DP_solver(self, phase_index, device_slice, layer_slice, inverseStage, the_whole_plan):

        index = self.phase_mapping[phase_index]
        rest_stage = stage_helper(phase_index, the_whole_plan)

        #print(device_slice,layer_slice)
        layers = layer_slice[1]-layer_slice[0]
        #print(len(self.DList[0]), index, self.DList[4][0], device_slice)
        c_list = [1/self.DList[i][index].computeprofile.forward for i in range(device_slice[0],device_slice[1])]
        #notice here, we need to take the rest stage into calculation...
        b_list = [int(self.DList[i][index].maximum_batches_available(layer_slice, inverseStage + rest_stage,self.seq_len[index],self.hiddenSize[index])) for i in range(device_slice[0],device_slice[1])] #memeory bound but tranfer to upperbound of batches
        if sum(b_list)<self.MbatchSize:              # layers, inversestage,seq,hidden
            #print(b_list, self.MbatchSize)
            return False ##cannot reach the batch assignment...
        if len(b_list)>self.MbatchSize:
            return False
        ###  "Largest Remainder Method" to distribute the batch in average...###
        n = len(b_list)
        c_sum = sum(c_list)
        ideal = [c * (self.MbatchSize) / c_sum for c in c_list]
        floored = [max(min(int(x), b_list[i]),1) for i, x in enumerate(ideal)]
        assigned = sum(floored)
        remainder = self.MbatchSize - assigned

        fractional = [(i, ideal[i] - int(ideal[i])) for i in range(n)]
        # Sort descending by fractional part
        fractional.sort(key=lambda x: -x[1])

        if remainder<0: # that is, too many necessary 1's, so we have to discard some value from 
            while remainder<0:
                fractional.sort(key=lambda x: x[1])
                for i, frac in fractional:
                    if floored[i] >1:
                        floored[i]-=1
                        remainder+=1

        # Distribute remaining units
        for i, frac in fractional:
            if remainder == 0:
                break
            gap = b_list[i] - floored[i]
            if gap > 0:
                assign = min(gap, remainder)
                floored[i] += assign
                remainder -= assign
        return floored



    def getall(self, P, mode = "shared"):
        #print("profilor's info:", P)

        ##warning: if having multiple decoders at the end,,, we need to change stepsN to s*len(P) and consider 
        ##redundant communication step envolved that might affect drawing...
        stepsN = 2*len(P)-1
        dp_list = self.dependency_list(P)
        #print("what is the dp_list?:", dp_list, P)
        B_ft, B_bt, B_fe, B_be, T_gathering, E_gathering = ([] for _ in range(6))
        BatchAllocateList = []
        for s in range(stepsN):


            stepinfo = P[s//2]


            
            layer_slice = stepinfo['layer']
            device_slice = stepinfo['device']
            
            phase_index = stepinfo['phase']
            #for each loops, use DP solver to get the answer,
            inverseStage = stepinfo['inver_internal_stage_idx']

            if s%2 == 0: # calculation step:
                    
                batch_shard = self.DP_solver(phase_index,device_slice, layer_slice, inverseStage, P)
                if batch_shard == False:
                    return False #failed to allocate batch for one Devices group
                BatchAllocateList.append(batch_shard)
                #calculate the time latency for the device group...
                layers = layer_slice[1]- layer_slice[0]
                devices = device_slice[1] - device_slice[0]
                latencyf = 0
                latencyb = 0
                index = 0
                energyf = 0
                energyb = 0
                power_total = 0
                the_model_index = self.phase_mapping[phase_index]
                for i in range(device_slice[0],device_slice[1]):
                    d = self.DList[i][the_model_index]
                    b = batch_shard[index]
                    lf,lb = d.Tlatency(layer_slice, b,self.seq_len[the_model_index],self.hiddenSize[the_model_index])
                    energyf+=d.Econsumpf(layer_slice, self.DList[0][the_model_index].layer_param_bytes, b,self.seq_len[the_model_index],self.hiddenSize[the_model_index])
                    energyb+=d.Econsumpb(layer_slice, self.DList[0][the_model_index].layer_param_bytes, b,self.seq_len[the_model_index],self.hiddenSize[the_model_index])
                    if lf>latencyf:
                        latencyf = lf
                    if lb>latencyb:
                        latencyb = lb
                    index+=1
                    power_total+=util_to_ratio(d.util)*Profile_Energy_checklist[str(d.type)]
                #now, we get latency of the device group

                ###jin important: we multiply a constant to estimate the backward cost
                B_ft.append(latencyf)
                B_bt.append(latencyb)
                B_fe.append(energyf)
                B_be.append(energyb)
                tt=0
                ee=0
                ###jin important: we gathering actually only involve transmission energy cost, it is little...
                if (devices > 1):
                    #print("a DP group")
                    tt = self.gathering_solver(phase_index, mode, device_slice, layer_slice)
                    ee = power_total*0.001*tt
                    #print(tt)
                T_gathering.append(tt)
                E_gathering.append(ee)


            else: # communication step:
                #print(dp_list)
                if dp_list[s]%2 != 0:
                    print("####################something error on communication cal#######################")
                nextstepinfo = P[dp_list[s]//2] if dp_list[s]  != -1 else None
                next_device_slice = nextstepinfo['device'] if nextstepinfo != None else None

                #print(device_slice, next_device_slice,P, s, dp_list)
                T = self.communication_solver(phase_index, mode, device_slice, next_device_slice)

                power_total = 0
                for i in range(device_slice[0],device_slice[1]):
                    d = self.DList[i][the_model_index]
                    power_total+=util_to_ratio(d.util)*Profile_Energy_checklist[str(d.type)]                
                for i in range(next_device_slice[0],next_device_slice[1]):
                    d = self.DList[i][the_model_index]
                    power_total+=util_to_ratio(d.util)*Profile_Energy_checklist[str(d.type)]   
                
                ee = power_total*0.001*T

                B_ft.append(T)
                B_bt.append(T)
                B_fe.append(ee)
                B_be.append(ee)                
                T_gathering.append(0)
                E_gathering.append(0)                   




        return B_ft, B_bt, B_fe, B_be, T_gathering, E_gathering, BatchAllocateList
    

if __name__ == "__main__":


    exampleP = [{'phase': (0, 0), 'layer': (0, 9), 'device': (0, 1)}, {'phase': (0, 0), 'layer': (9, 10), 'device': (1, 2)}, {'phase': (0, 1), 'layer': (0, 15), 'device': (2, 3)}, {'phase': (1, 0), 'layer': (0, 20), 'device': (3, 4)}]

    print(stage_helper((1,0),exampleP))