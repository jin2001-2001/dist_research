import re
from dataclasses import dataclass
from typing import NamedTuple



class TaskStatus_prime:
    def __init__(self, index, s,b,a , Tprofile, Rprofile, UnitR,
                isgathering = False, aa = 0,
                gatheringTprofile= [20,0,20], gatheringRprofile=[20,0,20],
                method1 = None, method2 = None,
                percentage = 0.5,
                plan_list = None,
                BatchAllocateList = None,
                band_str = None,
                sd_list = None,
                jmode = None):
        self.index = index
        self.s = s
        self.b = b
        self.a = a
        self.aa = aa
        self.Tprofile = Tprofile
        self.Rprofile = Rprofile
        self.gatheringTprofile = gatheringTprofile
        self.gatheringRprofile = gatheringRprofile
        self.UnitR = UnitR
        self.isgathering = isgathering
        self.percentage = percentage
        self.substitution = []


        self.plan_list = plan_list
        self.BatchAllocateList = BatchAllocateList
        self.band_str = band_str
        self.sd_list = sd_list
        self.jmode = jmode
        """
        Initialize the status variables based on input parameters.
        """
        self.set_func(method1, method2)
        # List to store dependency nodes
        self.dependency_nodes = []
        self.precedence_nodes = []

    def initialize(self):
        return 0
    
    def set_func(self, partitioner, suballocator):
        self.partitioner = partitioner
        self.suballocator = suballocator

    def add_dependency(self, node):
        """
        Add a node to the dependency list.
        """
        self.dependency_nodes.append(node)
    def add_precedence(self, node):
        self.precedence_nodes.append(node)







class Mode(NamedTuple):
    job: int
    duration: int
    demands: list[int]


@dataclass(frozen=True)
class Instance:
    """
    Problem instance class based on PSPLIB files.

    Code taken from:
    https://alns.readthedocs.io/en/latest/examples/resource_constrained_project_scheduling_problem.html
    """

    num_jobs: int  # jobs in RCPSP are tasks in PyJobshop
    num_resources: int
    successors: list[list[int]]
    predecessors: list[list[int]]
    modes: list[Mode]
    capacities: list[int]
    renewable: list[bool]

    @classmethod
    def read_instance(cls, path: str) -> "Instance":
        """
        Reads an instance of the RCPSP from a file.
        Assumes the data is in the PSPLIB format.
        """
        with open(path) as fh:
            lines = fh.readlines()

        prec_idx = lines.index("PRECEDENCE RELATIONS:\n")
        req_idx = lines.index("REQUESTS/DURATIONS:\n")
        avail_idx = lines.index("RESOURCEAVAILABILITIES:\n")

        successors = []

        aggregate = 0
        for line in lines[prec_idx + 2 : req_idx - 1]:
            if aggregate == 0:
                #print(re.split(r"\s+", line))
                aggregate+=1
            _, _, _, _, *jobs, _ = re.split(r"\s+", line)
            successors.append([int(x) - 1 for x in jobs])

        predecessors: list[list[int]] = [[] for _ in range(len(successors))]
        for job in range(len(successors)):
            for succ in successors[job]:
                predecessors[succ].append(job)

        mode_data = [
            re.split(r"\s+", line.strip())
            for line in lines[req_idx + 3 : avail_idx - 1]
        ]

        # Prepend the job index to mode data lines if it is missing.
        for idx in range(len(mode_data)):
            if idx == 0:
                continue

            prev = mode_data[idx - 1]
            curr = mode_data[idx]

            if len(curr) < len(prev):
                curr = prev[:1] + curr
                mode_data[idx] = curr

        modes = []
        for mode in mode_data:
            job_idx, _, duration, *consumption = mode
            demands = list(map(int, consumption))
            modes.append(Mode(int(job_idx) - 1, int(duration), demands))

        _, *avail, _ = re.split(r"\s+", lines[avail_idx + 2])
        capacities = list(map(int, avail))

        renewable = [
            x == "R"
            for x in lines[avail_idx + 1].strip().split(" ")
            if x in ["R", "N"]  # R: renewable, N: non-renewable
        ]

        return Instance(
            int(job_idx),
            len(capacities),
            successors,
            predecessors,
            modes,
            capacities,
            renewable,
        )