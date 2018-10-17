from mpi4py import MPI
from evo_rbc.env.ant_env import AntEAEnv
import evo_rbc.main.utils as test_utils

logger = test_utils.getLogger()

comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()


genomes_matrix = visualise = max_time_steps_qd =None

max_time_steps_qd = comm.bcast(max_time_steps_qd,root=0)
env = AntEAEnv(max_time_steps_qd=max_time_steps_qd)
qd_function = env.qd_steady_runner

# print("child spawned and running",rank)

genomes = comm.scatter(genomes_matrix,root=0)

visualise = comm.bcast(visualise,root=0)
# print("visualise",visualise,rank)
qd_evaluations = []
for i in range(len(genomes)):
	behavior,quality = env.evaluate_quality_diversity_fitness(qd_function=qd_function,primitive_genome=genomes[i],visualise=visualise)
	logger.debug(str((rank,behavior,quality)))
	qd_evaluations.append((behavior,quality))
	# print("rank",rank,"i",i,"control freq",genomes[i].parameters["control_frequency"],bin_index)

# print(rank,qd_evaluations)

qd_evaluations = comm.gather(qd_evaluations,root=0)

comm.Disconnect()