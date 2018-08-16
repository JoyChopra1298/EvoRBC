from mpi4py import MPI
from evo_rbc.env.ant_env import AntEAEnv


comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()

env = AntEAEnv(max_time_steps_qd=1000*rank+100)

print("child spawned and running",rank)
genomes_matrix = visualise = None

genomes = comm.scatter(genomes_matrix,root=0)

visualise = comm.bcast(visualise,root=0)
print("visualise",visualise,rank)

for genome in genomes:
	print("genome parameter control frequeny",genome.parameters["control_frequency"],rank)
	print("Starting Evaluation",rank)
	behavior,quality = env.evaluate_quality_diversity_fitness(qd_function=env.qd_steady_runner,primitive_genome=genome,visualise=visualise)
	print(rank,behavior,quality)

comm.Disconnect()
