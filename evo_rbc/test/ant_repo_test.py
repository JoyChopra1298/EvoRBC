from evo_rbc.qd_solver.map_elites import MAP_Elites
from evo_rbc.genome.ant_genome import AntGenome
from evo_rbc.env.ant_env import AntEAEnv
from evo_rbc.qd_solver.container.grid import Grid
import numpy as np
from evo_rbc.test.utils import print_heading

seed = 1
population_size = 100
mutation_rate = 0.1
max_time_steps_qd=1000
max_time_steps_task=2000
visualise = False

#grid details
num_dimensions = 2
lower_limit = np.array([-2,-2])
upper_limit = np.array([2,2])
resolution = np.array([.01,.01])

ant_env = AntEAEnv(seed=seed,max_time_steps_qd=max_time_steps_qd,max_time_steps_task=max_time_steps_task)
ant_genome = AntGenome(seed=seed)
map_elites = MAP_Elites(env=ant_env,genome=ant_genome,num_dimensions=num_dimensions,lower_limit=lower_limit,upper_limit=upper_limit,
	resolution=resolution,population_size=population_size,mutation_rate=mutation_rate)

print_heading("Performance and behavior after an evaluation on environment")
behavior,quality = ant_env.evaluate_quality_diversity_fitness(ant_env.qd_steady_runner,ant_genome,visualise) 
print(behavior,quality)

container = map_elites.container
print_heading("Number of bins in grid container and check grid initialisation")
print(container.num_bins)
print(container.grid[container.get_bin(lower_limit)])

print_heading("Test grid container functions")
print("is genome high quality. should return true as bin empty --- ",container.is_high_quality(behavior,quality))
container.add_genome(genome=ant_genome,behavior=behavior,quality=quality)
print("Allocated bin ",container.get_bin(behavior))
print("Total quality ",container.total_quality)
print("Max quality ",container.max_quality)
print("Max quality bin ",container.max_quality_bin)
print("Number of genomes in the container",container.num_genomes)

