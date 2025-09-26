
class Device:
    def __init__(self, Cability, Pability, Mconstraint, Econstraint):
        self.Tability = Cability
        self.Eability = Pability
        self.Mconstraint = Mconstraint
        self.Econstraint = Econstraint

    def Tlatency(self, layers, layers_size, batch_size,seq,hidden):
        bsize = batch_size*seq*hidden
        return layers*layers_size*bsize/self.Tability
    
    def Econsump(self, layers, layers_size, batch_size,seq,hidden):
        E =layers*layers_size*self.Tlatency(layers, layers_size, batch_size,seq,hidden)*self.Eability
        #print(E)
        return E

    def maximum_batches_available(self, layers, layers_size, inversestage,seq,hidden): #reversely get the maximum batches that can be assigned on this devices
        max1 = self.Mconstraint/(layers*layers_size*(inversestage*2+1))/(seq*hidden)
        max2 = self.Econstraint * self.Tability/(self.Eability*(layers**2)*(layers_size**2))/(seq*hidden)
        #print(max1, max2)
        return min(max1, max2)

    def if_overflow(self, layers, layers_size, batch_size):
        i = False
        if (layers*layers_size*batch_size>self.Mconstraint):
            i = True
        if (self.Econsump(layers, layers_size, batch_size)>self.Econstraint):
            i = True
        return i


class Profilelor:
    """
    Maintains the k smallest (score, plan) pairs.
    """
    def __init__(self,DList,layerSize,hiddenSize, seq, MbatchSize, Bandwidth):
        #self.deviceN = damount
        self.DList = DList
        self.layerSize = layerSize
        self.hiddenSize = hiddenSize
        self.seq_len = seq
        self.MbatchSize = MbatchSize
        self.bandwidth = Bandwidth


    def batch_checker(self, device_slice, layer_slice, inverseStage):
        total_mem = sum(self.DList[i].Mconstraint for i in range(device_slice[0], device_slice[1]))
        total_size = (layer_slice[1]-layer_slice[0])*self.layerSize*inverseStage*self.MbatchSiz
        if total_mem<total_size:
            return False
        return True
    
    def communication_solver(self, bsize, layer_slice=1): #simple version
        T = bsize*self.hiddenSize*self.seq_len/(self.bandwidth)
        return T

    def gathering_solver(self, device_slice, layer_slice): #simple version
        D_amount = device_slice[1]-device_slice[0]
        l_amount = layer_slice[1]-layer_slice[0]
        total_parameter = l_amount*self.layerSize*D_amount
        T = total_parameter/(self.bandwidth)
        return T


    def DP_solver(self, device_slice, layer_slice, inverseStage):
        #print(device_slice,layer_slice)
        layers = layer_slice[1]-layer_slice[0]
        c_list = [self.DList[i].Tability for i in range(device_slice[0],device_slice[1])]
        b_list = [int(self.DList[i].maximum_batches_available(layers, self.layerSize, inverseStage,self.seq_len,self.hiddenSize)) for i in range(device_slice[0],device_slice[1])] #memeory bound but tranfer to upperbound of batches
        if sum(b_list)<self.MbatchSize:
            #print(b_list, self.MbatchSize)
            return False ##cannot reach the batch assignment...
        if len(c_list)>self.MbatchSize:
            return False
        ###  "Largest Remainder Method" to distribute the batch in average...###
        n = len(c_list)
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
                latency = 0
                index = 0
                total_energy = 0

                for i in range(device_slice[0],device_slice[1]):
                    d = self.DList[i]
                    b = batch_shard[index]
                    l = d.Tlatency(layers, self.layerSize, b,self.seq_len,self.hiddenSize)
                    total_energy+=d.Econsump(layers, self.layerSize, b,self.seq_len,self.hiddenSize)
                    if l>latency:
                        latency = l
                    index+=1
                #now, we get latency of the device group

                ###jin important: we multiply a constant to estimate the backward cost
                B_ft.append(latency)
                B_bt.append(latency*1.75)
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
