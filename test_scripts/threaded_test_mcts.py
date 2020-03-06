import os
from multiprocessing.pool import ThreadPool  # dummy is nothing but multiprocessing but wrapper around threading

DOMAIN = 'namo'


def worker_p(config):
    pidx = config[0]

    command = 'python ./test_scripts/test_mcts.py -sampling_strategy uniform -problem_idx ' + str(pidx) + ' -domain ' + DOMAIN

    print command
    os.system(command)


def worker_wrapper_multi_input(multi_args):
    return worker_p(multi_args)


def main():
    trials = range(963)

    configs = []
    for t in trials:
        configs.append([t])
    n_workers = int(10)

    print configs
    pool = ThreadPool(n_workers)
    results = pool.map(worker_wrapper_multi_input, configs)


if __name__ == '__main__':
    main()
