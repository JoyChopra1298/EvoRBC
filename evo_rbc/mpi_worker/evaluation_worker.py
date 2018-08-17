from mpi4py import MPI
from evo_rbc.env.ant_env import AntEAEnv
import evo_rbc.main.utils as test_utils

logger = test_utils.getLogger()

comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()

env = AntEAEnv(max_time_steps_qd=1000)
qd_function = env.qd_steady_runner

# print("child spawned and running",rank)
genomes_matrix = visualise = None

genomes = comm.scatter(genomes_matrix,root=0)

visualise = comm.bcast(visualise,root=0)
# print("visualise",visualise,rank)

qd_evaluations = []
for genome in genomes:
	behavior,quality = env.evaluate_quality_diversity_fitness(qd_function=qd_function,primitive_genome=genome,visualise=visualise)
	logger.debug(str((rank,behavior,quality)))
	qd_evaluations.append((behavior,quality))

qd_evaluations = comm.gather(qd_evaluations,root=0)

comm.Disconnect()
