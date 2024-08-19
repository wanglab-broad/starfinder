#!/usr/bin/env python3
"""
Script to submit a snakemake task to the Broad UGER cluster
"""

import os
import re
import sys
import subprocess
from datetime import datetime

from snakemake.utils import read_job_properties

__author__ = "Lucas van Dijk <lvandijk@broadinstitute.org>"

# Whether we're on UGER or UGES
CLUSTER_TYPE = "UGER"

dependencies = sys.argv[1:-1]
jobscript = sys.argv[-1]

# Start figuring out the requested resources for this job
job = read_job_properties(jobscript)
cluster_conf = job.get('cluster', {})
job_resources = job.get('resources', {})
# print(job_resources)

matches = re.match(r'(\S+)/snakejob\.\S+\.(\d+)\.sh', jobscript)
if not matches:
    raise Exception("Error parsing snakemake job ID: jobscript did not match"
                    " pattern.\n\nJobscript: {}".format(jobscript))

sm_tmpdir, sm_jobid = matches.groups()
jobname = "{rule}-{jobid}".format(rule=job['rule'], jobid=sm_jobid)

# Cluster memory configuration gets precedence over memory specification in
# resources section of a rule.
# Specify with --cluster-config to snakemake command
mem_mb = cluster_conf.get('mem_mb', job_resources.get('mem_mb', None))
threads = job.get('threads', 1)
project = cluster_conf.get('project', 'broad')
queue = cluster_conf.get('queue', '')
runtime = cluster_conf.get('runtime', job_resources.get('runtime', None))

# -terse flag makes sure qsub only outputs job ID to stdout
# -r signals that jobs may be restarted in cases of *cluster* crashes
command = ['qsub', '-terse', '-N', jobname, '-r', 'y', '-cwd']

# Add project or queue
if CLUSTER_TYPE == "UGER":
    command.extend(['-P', project])
elif CLUSTER_TYPE == "UGES":
    command.extend(['-q', queue])

if "log" in job and len(job['log']) > 0:
    logfile = os.path.splitext(job['log'][0])[0] + "-qsub.log"
else:
    path = f"snakemake-logs/{job['rule']}/"
    os.makedirs(path, exist_ok=True)

    logfile = (path + datetime.now().strftime('%Y%m%d-%H%M%S') +
               f"-jobid{sm_jobid}.log")

command.extend(['-o', logfile, '-j', 'y'])

if threads > 1:
    command.extend(['-pe', 'smp', str(threads), '-binding',
                    'linear:{}'.format(threads)])

# Allow for reservation
if (mem_mb and mem_mb > (15*1024)) or threads >= 4:
    command.extend(['-R', 'y'])

# The memory we request will be per core, so we divide the requested memory by
# the number of cpus
if mem_mb:
    mem_mb = round(mem_mb / threads, 2)
    command.extend(['-l', 'h_vmem={}M'.format(mem_mb)])

if runtime:
    import datetime
    runtime_uger = datetime.timedelta(minutes=runtime)
    command.extend(['-l', 'h_rt={}'.format(str(runtime_uger))])

if dependencies:
    command.extend(['-hold_jid', ",".join(dependencies)])

command.append(jobscript)

# Pass the path of the 'jobfinished' file which will be created by snakemake
# on success. The jobscript itself will use this to determine the right exit
# code.
command.append("{}/{}.jobfinished".format(sm_tmpdir, sm_jobid))

proc = subprocess.run(command, stdout=subprocess.PIPE, check=True,
                      encoding='utf-8')
print(proc.stdout.strip())
