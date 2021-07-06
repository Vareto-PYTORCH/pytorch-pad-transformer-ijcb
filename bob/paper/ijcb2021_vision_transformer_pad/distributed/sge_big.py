from dask.distributed import Client

from bob.pipelines.distributed.sge import SGEMultipleQueuesCluster

from bob.pipelines.distributed.sge_queues import QUEUE_GPU


QUEUE={'default': {'queue': 'q_1day',
  'memory': '32GB',
  'io_big': True,
  'resource_spec': '',
  'max_jobs': 48,
  'resources': {'q_1day': 1}}}

cluster = SGEMultipleQueuesCluster(min_jobs=1, sge_job_spec=QUEUE)
dask_client = Client(cluster)



