
import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
##profile is based on: for a fixed model...
##first, we need time/energy cost for Batch size, layer amounts...
##so, the input can be two elements: B and L
##first, input should be the name of the device, profile location, maximun batch/computation profiled...

## we use ax**2 + bx + c to simulate a time cost under batches  for a fixed number of layers
## we need extract out important things...
## 1. forward & backward time cost for base situation: 1 batches, for a fixed number of layers

## how to get more 

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
    deg = min(3, n - 1)  # degrade if not enough points
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
    def __init__(self, type, tprofile_loc, eprofile_loc = 0, Mem = 0, Energy = 1000):
        self.type = type
        self.Eability = 0
        self.Mconstraint = Mem  # unit: MB
        self.Econstraint = Energy
        self.activation_bytes_per_sample = 0
        self.activation_bytes_seq_len = 0
        self.layer_param_bytes = 0
        self.embedding_param_bytes = 0
        self.tail_param_bytes = 0
        #self.total_layers = 0
        #self.total_parameters_bytes = 0

        root = Path(tprofile_loc)
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
            self.activation_bytes_seq_len = data["seq_len"]
            self.activation_bytes_per_sample = data["layers"][0]["activation_bytes_per_sample"]
            self.layer_param_bytes = data["layers"][0]["param_bytes"]
            self.embedding_param_bytes = data["embed_param_bytes"]
            self.tail_param_bytes = data["tail_param_bytes"]
            #self.total_parameters_bytes = data["total_parameters_bytes"]
            total_profile_layers = len(data["layers"])
            for i in range(total_profile_layers):
                forward+=data["layers"][i]["forward_time_s"]
                backward+=data["layers"][i]["backward_time_s"]
            forward =forward/total_profile_layers
            backward=backward/total_profile_layers
        
        onebforward = forward
        onebbackward = backward
        batch = 2
        outforward = []
        outbackward = []
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
                    forward+=data["layers"][i]["forward_time_s"]
                    backward+=data["layers"][i]["backward_time_s"]
                forward =forward/total_profile_layers
                backward=backward/total_profile_layers
                outforward.append(forward)
                outbackward.append(backward)
            batch += 1
        
        outforward = [onebforward]+outforward
        outbackward = [onebbackward]+outbackward

        a = fit_cubic_from_series(outforward)
        b = fit_cubic_from_series(outbackward)
        self.computeprofile = ComputProfile(forward,backward,1,1,a,b)






    def Tlatency(self, layers, batch_size):
        Tf = self.computeprofile.batchFuncforward(batch_size)*layers/self.computeprofile.layer_base
        Tb = self.computeprofile.batchFuncbackward(batch_size)*layers/self.computeprofile.layer_base
        return Tf,Tb
    
    def Econsump(self, layers, layers_size, batch_size,seq,hidden):
        E =0
        return E

    def maximum_batches_available(self, layers, inversestage,seq,hidden): #reversely get the maximum batches that can be assigned on this devices
        batch_act_storage = 7*seq*hidden*4 *layers *(inversestage*2+1)   #no batch, we need calculate batches...#4 is float32 needs 4 bytes
        parameter_storage = self.layer_param_bytes * layers *4 #(2 for opt, 1 for gradient)
        
        max1 = self.Mconstraint*1024*1024/(batch_act_storage+parameter_storage)
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


    def batch_checker(self, device_slice, layer_slice, inverseStage):
        total_mem = sum(self.DList[i].Mconstraint for i in range(device_slice[0], device_slice[1]))
        layers = (layer_slice[1]-layer_slice[0])
        batch_act_storage = 7*self.seq_len*self.hiddenSize*4 *layers *(inverseStage*2+1)*self.MbatchSize # this 4 is float32
        parameter_storage = self.DList[0].layer_param_bytes * layers *4*self.MbatchSize

        total_size = (batch_act_storage+parameter_storage)*(device_slice[1]-device_slice[0])/1024/1024 #M
        if total_mem<total_size:
            return False
        return True
    
    def communication_solver(self, layer_slice=1): #simple version
        #T = bsize*self.hiddenSize*self.seq_len/(self.bandwidth)
        T = self.DList[0].activation_bytes_per_sample * self.MbatchSize/1024/1024/self.bandwidth

        return T

    def gathering_solver(self, device_slice, layer_slice): #simple version
        D_amount = device_slice[1]-device_slice[0]
        l_amount = layer_slice[1]-layer_slice[0]

        accum= l_amount*self.DList[0].layer_param_bytes

        if layer_slice[0] == 0:
            accum += self.DList[0].embedding_param_bytes
        if layer_slice[1] == self.total_layers:
            accum += self.DList[0].tail_param_bytes

        total_parameter_bytes = accum*D_amount *2 #plus embedding estimiation

        T = total_parameter_bytes/1024/1024/(self.bandwidth)

        return T


    def DP_solver(self, device_slice, layer_slice, inverseStage):
        #print(device_slice,layer_slice)
        layers = layer_slice[1]-layer_slice[0]
        c_list = [1/self.DList[i].computeprofile.forward for i in range(device_slice[0],device_slice[1])]
        b_list = [int(self.DList[i].maximum_batches_available(layers, inverseStage,self.seq_len,self.hiddenSize)) for i in range(device_slice[0],device_slice[1])] #memeory bound but tranfer to upperbound of batches
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
                total_energy = 0

                for i in range(device_slice[0],device_slice[1]):
                    d = self.DList[i]
                    b = batch_shard[index]
                    lf,lb = d.Tlatency(layers, b)
                    total_energy+=d.Econsump(layers, self.DList[0].layer_param_bytes, b,self.seq_len,self.hiddenSize)
                    if lf>latencyf:
                        latencyf = lf
                    if lb>latencyb:
                        latencyb = lb
                    index+=1
                #now, we get latency of the device group

                ###jin important: we multiply a constant to estimate the backward cost
                B_ft.append(latencyf)
                B_bt.append(latencyb)
                B_fe.append(total_energy*0.4)
                B_be.append(total_energy*0.6)
                tt=0
                ee=0
                ###jin important: we gathering actually only involve transmission energy cost, it is little...
                if (devices > 1):
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

    



#def compute_integer_ai(c_list, total_C):
#    c_sum = sum(c_list)
#    ideal = [c * total_C / c_sum for c in c_list]
#    floored = [int(x) for x in ideal]
#    remainder = total_C - sum(floored)
#
#    # Get fractional parts with index
#    fractional = [(i, ideal[i] - floored[i]) for i in range(len(c_list))]
#    # Sort by descending fractional part
#    fractional.sort(key=lambda x: -x[1])
#
#    # Distribute remaining units
#    for i in range(remainder):
#        idx = fractional[i][0]
#        floored[idx] += 1
#
#    return floored
