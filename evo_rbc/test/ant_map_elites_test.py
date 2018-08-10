from evo_rbc.qd_solver.map_elites import MAP_Elites
from evo_rbc.genome.ant_genome import AntGenome
from evo_rbc.env.ant_env import AntEAEnv
from evo_rbc.qd_solver.container.grid import Grid
import numpy as np
from evo_rbc.test.utils import print_heading
from evo_rbc.qd_solver.selector.uniform_random_selector import Uniform_Random_Selector
from evo_rbc.qd_solver.selector.curiosity_driven_selector import Curiosity_Driven_Selector

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
map_elites = MAP_Elites(env=ant_env,genome=ant_genome,selector=Uniform_Random_Selector(),num_dimensions=num_dimensions,
	lower_limit=lower_limit,upper_limit=upper_limit,resolution=resolution,population_size=population_size,mutation_rate=mutation_rate)

print_heading("Performance and behavior after an evaluation on environment")
behavior,quality = ant_env.evaluate_quality_diversity_fitness(ant_env.qd_steady_runner,ant_genome,visualise) 
print(behavior,quality)

container = map_elites.container
print_heading("Number of bins in grid container and check grid initialisation")
print(container.num_bins)
print(container.grid)

print_heading("Test grid container functions")
print("is genome high quality. should return true as bin empty --- ",container.is_high_quality(behavior,quality))
container.add_genome(genome=ant_genome,behavior=behavior,quality=quality)
print("Allocated bin ",container.get_bin(behavior))
print("Total quality ",container.total_quality)
print("Max quality ",container.max_quality)
print("Max quality bin ",container.max_quality_bin)
print("Number of genomes in the container",container.num_genomes)
print("Normalised total quality ",container.total_quality/container.num_genomes)

print_heading("Add 5 more random genomes")
for i in range(5):
	ant_genome.mutate()
	behavior,quality = ant_env.evaluate_quality_diversity_fitness(ant_env.qd_steady_runner,ant_genome,visualise)
	container.add_genome(genome=ant_genome,behavior=behavior,quality=quality)
print("Total quality ",container.total_quality)
print("Max quality ",container.max_quality)
print("Max quality bin ",container.max_quality_bin)
print("Number of genomes in the container",container.num_genomes)
print("Normalised total quality ",container.total_quality/container.num_genomes)

print_heading("Select samples uniformly from container")
uniform_random_selector = Uniform_Random_Selector()
sampled_population = uniform_random_selector.select(container.grid,3)
print(sampled_population)
print("Sampled genome parameters - smoothness",[genome_details["genome"].parameters["smoothness"] for bin_index,genome_details in sampled_population])

print_heading("Select samples based on curiosity scores from container")
curiosity_driven_selector = Curiosity_Driven_Selector()
sampled_population = curiosity_driven_selector.select(container.grid,3)
print(sampled_population)
print("Sampled genome parameters - smoothness",[genome_details["genome"].parameters["smoothness"] for bin_index,genome_details in sampled_population])

## update curiosity of particular elements to see bias
new_genome_details = sampled_population[0][1]
new_genome_details["curiosity"] = 1000
container.update_bin(bin_index=sampled_population[0][0],genome_details=new_genome_details)
print_heading("Test biased curiosity. shoould output first element of previous test most probably")
for i in range(5):
	sampled_population = curiosity_driven_selector.select(container.grid,1)
	print("Sampled genome parameters - smoothness",[genome_details["genome"].parameters["smoothness"] for bin_index,genome_details in sampled_population])
