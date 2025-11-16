import psplib_
from psplib_ import Instance
from pyjobshop import Model
import matplotlib.pyplot as plt
from pyjobshop.plot import plot_resource_usage, plot_task_gantt
from sample_generation import indexer, gindexer
from sample_generation_shift import indexer_shift, gindexer_shift

def RCPSP_sol_parser(tasks,s, a,b, step, nmicrobatch, ntasks = -1,aa = 0,if_gathering = False):
    i = 0
    if if_gathering == False:
        i = indexer(s,a,b, step, nmicrobatch, ntasks )
    else:
        i = gindexer(s,a,b,step, nmicrobatch, ntasks,aa)

    return (tasks[i].start, tasks[i].end)

def RCPSP_sol_parser_shift(tasks,s, a,b, step, nmicrobatch, ntasks = -1,aa = 0,if_gathering = False):
    i = 0
    if if_gathering == False:
        i = indexer_shift(s,a,b, step, nmicrobatch, ntasks )
    else:
        i = gindexer_shift(s,a,b,step, nmicrobatch, ntasks,aa)

    return (tasks[i].start, tasks[i].end)


def RCPSP_solver(pfile = "./scratch1.sm"):
    instance = Instance.read_instance(pfile)
    model = Model()

    # It's not necessary to define jobs, but it will add coloring to the plot.
    jobs = [model.add_job() for _ in range(instance.num_jobs)]
    tasks = [model.add_task(job=jobs[idx]) for idx in range(instance.num_jobs)]
    resources = [
        model.add_renewable(capacity=capacity)
        if renewable
        else model.add_non_renewable(capacity=capacity)
        for capacity, renewable in zip(instance.capacities, instance.renewable)
        ]


    for idx, duration, demands in instance.modes:
        model.add_mode(tasks[idx], resources, duration, demands)

    for idx in range(instance.num_jobs):
        task = tasks[idx]

        for pred in instance.predecessors[idx]:
            model.add_end_before_start(tasks[pred], task)

        for succ in instance.successors[idx]:
            model.add_end_before_start(task, tasks[succ])


    result = model.solve(time_limit=10, display=False)
    #print(result.best.tasks)
    #print(result.status)
    #print(result.objective/result.lower_bound)
    #print(result.runtime)
    return model, result


def RCPSP_plot(model, result):
    data = model.data()
    fig, axes = plt.subplots(
        data.num_resources + 1,
        figsize=(12, 16),
        gridspec_kw={"height_ratios": [6] + [1] * data.num_resources},
    )
    
    plot_task_gantt(result.best, model.data(), ax=axes[0])
    plot_resource_usage(result.best, model.data(), axes=axes[1:])
    plt.show()
if __name__ == "__main__":
    model, result = RCPSP_solver()
    RCPSP_plot(model, result)