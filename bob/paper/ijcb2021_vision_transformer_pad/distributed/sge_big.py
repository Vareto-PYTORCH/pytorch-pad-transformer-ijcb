from dask.distributed import Client

from bob.pipelines.distributed.sge import SGEMultipleQueuesCluster

from bob.pipelines.distributed.sge_queues import QUEUE_GPU


QUEUE={"default": {
        "queue": "q_short_gpu",
        "memory": "16GB",
        "io_big": False,
        "resource_spec": "",
        "max_jobs": 128,
        "resources": "",}}



# QUEUE={"default": {
#         "queue": "q_1day_mth",
#         "memory": "32GB",
#         "io_big": False,
#         "resource_spec": "",
#         "job_extra": "pe_mth 2",
#         "max_jobs": 128,
#         "resources": "",}}


cluster = SGEMultipleQueuesCluster(min_jobs=1, sge_job_spec=QUEUE)
dask_client = Client(cluster)



