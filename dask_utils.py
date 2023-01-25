import os
import logging
from dask.distributed import Client, LocalCluster
from psutil import virtual_memory, cpu_count

log = logging.getLogger(os.path.basename(__file__))


def set_up_dask_client():
    log.debug('Setting up dask cluster and client...')
    memory_limit = round(virtual_memory().total / 1e9) - 1
    threads_per_worker = max(cpu_count() // 2, 1)
    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=threads_per_worker,
        memory_limit=f'{memory_limit}GB',
    )
    log.debug(f'Dask cluster initiated with: threads_per_worker={threads_per_worker}, memory_limit={memory_limit}GB. '
              f'Dashboard link: {cluster.dashboard_link}')
    client = Client(cluster)
    log.debug(f'Dask client dashboard link: {client.dashboard_link}')
    return client

