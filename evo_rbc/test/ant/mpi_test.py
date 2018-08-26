from evo_rbc.genome.ant_genome import AntGenome
from evo_rbc.env.ant_env import AntEAEnv
import evo_rbc.main.utils as test_utils
from mpi4py import MPI
import sys
import copy

logger = test_utils.getLogger()

genome = AntGenome()
visualise = False
max_time_steps_qd = 1000

genomes = []
num_processes = 3

for i in range(num_processes*2+2):
	genome_i = copy.deepcopy(genome)
	genome_i.mutate()
	genomes.append(genome_i)	

comm = MPI.COMM_SELF.Spawn(sys.executable,
                           args=['../../mpi_worker/ant_evaluation_worker.py'],
                           maxprocs=num_processes)

genomes_len = len(genomes)

genomes_matrix = [[] for i in range(num_processes)]

for i in range(genomes_len):
	genomes_matrix[i%num_processes].append(genomes[i])

max_time_steps_qd = comm.bcast(max_time_steps_qd,root=MPI.ROOT)
genome = comm.scatter(genomes_matrix,root=MPI.ROOT)
visualise = comm.bcast(visualise,root=MPI.ROOT)

qd_evaluations = None
qd_evaluations = comm.gather(qd_evaluations,root=MPI.ROOT)

for i in range(len(genomes)):
	print("--",qd_evaluations[i%num_processes][int(i/num_processes)],i%num_processes,int(i/num_processes))
	print("--",genomes[i].parameters["control_frequency"],i,i%num_processes,int(i/num_processes))

comm.Disconnect()
