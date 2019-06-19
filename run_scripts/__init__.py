import os
import sys
import datetime
import torch
from sqlalchemy.ext.declarative import declarative_base
from tensorboardX import SummaryWriter
import matplotlib as mpl

verbose = True
save_to_db = True
visualize = True
show_visualization = False
tensorboard_logging = False
memory_logging = True
RESIDUAL_LEARNING = False
OUTPUT_TO_FILE = True

files = [sys.stdout]


def print_(content=''):
    for file in files:
        file.write(str(content) + '\n')
        file.flush()


try:
    is_cluster_run = bool(os.environ['CLUSTER_RUN'])
except KeyError:
    is_cluster_run = False

try:
    cluster_job_run_id = int(os.environ['CLUSTER_JOB_RUN_ID'])
except KeyError:
    cluster_job_run_id = int(datetime.datetime.today().strftime("%Y%m%d%H%M%S"))

try:
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
except KeyError:
    task_id = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if is_cluster_run:
    mpl.use('Agg')
    cluster_job_run_id_folder_name = 'C' + str(cluster_job_run_id)
else:
    cluster_job_run_id_folder_name = 'L' + str(cluster_job_run_id)

REPO_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOB_DIR = REPO_BASE_DIR + '/results/logs/' + datetime.date.today().strftime("%Y-%m-%d") + '/' + cluster_job_run_id_folder_name
if not os.path.isdir(JOB_DIR):
    os.makedirs(JOB_DIR)

if OUTPUT_TO_FILE and not is_cluster_run:
    output_file_path = JOB_DIR + '/' + 'out.txt'
    f = open(output_file_path, 'w')
    files.append(f)

if is_cluster_run:
    print_("Cluster_run_id: {}, task_id: {}".format(cluster_job_run_id, task_id))
else:
    print_("Local_run_id: {}, task_id: {}".format(cluster_job_run_id, task_id))
print_("Device: {}".format(device))


Base = declarative_base()
if tensorboard_logging:
    tensorboard_writer = SummaryWriter(JOB_DIR + '/tensorboardX')
else:
    tensorboard_writer = None
