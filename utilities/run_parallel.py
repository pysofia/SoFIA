import sys
import numpy as np

def parallel_cases_with_mpi(cases, run_func, H, verbose=True):
    """
    Run a list of "cases" in parallel, divided accross all available cores.

    Specifically, this function calls run_func for each of the case descriptions
    in cases, equally divided accross the available CPU cores.  When more than
    one core is available, the order of the cases run is not guaranteed.  For
    a single core, cases are run in order.

    Note: if the runtime for each job varies significantly, you can shuffle the
    order of the cases list to improve load balancing.

    Parameters
    ==========
    cases: list
        objects representing parameterizations defining cases to run
    run_func: function object
        function which runs a given parameterization (item in cases)
    verbose: bool
        if True, information is printed regarding which core runs which case,
        otherwise nothing is printed
    """
    if 'mpi4py' not in sys.modules: from mpi4py import MPI

    comm = MPI.COMM_WORLD

    info = f'[{comm.rank}]: ' if comm.size > 1 else ''

    for i in range(len(cases)):
        sendbuf = []
        if comm.rank == 0:
            m=np.array(range(comm.size),dtype=float)
            sendbuf = m

        v = comm.scatter(sendbuf,root=0)

        case_num = i * comm.size + comm.rank

        if case_num < len(cases):
            case = cases[case_num]
            if verbose:
                print(info + f'running case {case_num} - {case}', flush=True)
            v = run_func(case,H)

            recvbuf = comm.gather(v,root=0)
            if comm.rank==0:
                for i in range(len(recvbuf)):
                    H.append(recvbuf[i])

        else:
            break

    if comm.rank==0:
        return H
    # else:
    #     return

    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # size = comm.Get_size()

    # info = f'[{rank}]: ' if size > 1 else ''

    # if verbose:
    #     if rank == 0 and size > 1:
    #         print(f'Running {len(cases)} cases with {size} processors.')
    #         print(f'MPI Version: {MPI.Get_version()}', flush=True)
    #     comm.Barrier()

    # for i in range(len(cases)):
    #     case_num = i * size + rank

    #     if case_num < len(cases):
    #         case = cases[case_num]
    #         if verbose:
    #             print(info + f'running case {case_num} - {case}', flush=True)
    #         run_func(case)
    #     else:
    #         break