import os
import logging
from dask.distributed import Client, LocalCluster
from psutil import virtual_memory, cpu_count

log = logging.getLogger(os.path.basename(__file__))


def set_up_dask_client():
    log.debug('Setting up dask cluster and client...')
    n_workers = 1
    memory_limit = round(virtual_memory().total / 1e9) - max(1, round(1 * virtual_memory().total / 1e9 / 16))
    threads_per_worker = max(cpu_count() // 2, 1)
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=f'{memory_limit}GB',
    )
    # total_memory_limit = round(virtual_memory().total / 1e9) - 1 * round(virtual_memory().total / 1e9 / 16)
    # n_workers = cpu_count(False)
    # threads_per_worker = 2
    # memory_limit = round(total_memory_limit / n_workers)
    # cluster = LocalCluster(
    #     n_workers=n_workers,
    #     threads_per_worker=threads_per_worker,
    #     memory_limit=f'{memory_limit}GB',
    # )
    log.debug(f'Dask cluster initiated with: n_workers={n_workers}, threads_per_worker={threads_per_worker}, '
              f'memory_limit={memory_limit}GB. Dashboard link: {cluster.dashboard_link}')
    client = Client(cluster)
    log.debug(f'Dask client dashboard link: {client.dashboard_link}')
    return client

