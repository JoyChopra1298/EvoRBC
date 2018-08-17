from evo_rbc.genome.ant_genome import AntGenome
from evo_rbc.env.ant_env import AntEAEnv
import evo_rbc.main.utils as test_utils
from mpi4py import MPI
import sys
import copy

logger = test_utils.getLogger()

genome = AntGenome()
visualise = False

genomes = []
num_processes = 3

for i in range(num_processes*2):
	genome_i = copy.deepcopy(genome)
	genome_i.mutate()
	genomes.append(genome_i)

comm = MPI.COMM_SELF.Spawn(sys.executable,
                           args=['../mpi_worker/evaluation_worker.py'],
                           maxprocs=num_processes)

genomes_len = len(genomes)
step = int(genomes_len/num_processes)

genomes_matrix = [genomes[i*step:(i+1)*step] for i in range(0,num_processes)]

for i in range(genomes_len%num_processes):
	genomes_matrix[i].append(genomes[(num_processes*step)+i])

genome = comm.scatter(genomes_matrix,root=MPI.ROOT)
visualise = comm.bcast(visualise,root=MPI.ROOT)

qd_evaluations = None
qd_evaluations = comm.gather(qd_evaluations,root=MPI.ROOT)

print(qd_evaluations)
comm.Disconnect()
