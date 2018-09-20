from evo_rbc.qd_solver.map_elites import MAP_Elites
from evo_rbc.genome.prosthetic_genome import ProstheticGenome
from evo_rbc.env.prosthetic_env import ProstheticEAEnv
import numpy as np
import evo_rbc.main.utils as test_utils
from evo_rbc.qd_solver.selector.uniform_random_selector import Uniform_Random_Selector
from evo_rbc.qd_solver.selector.curiosity_driven_selector import Curiosity_Driven_Selector

seed = 1
batch_size = 5
max_time_steps_qd=300
max_time_steps_task=2000
visualise = visualize = True

joint_error_margin = 0.1


#Grid details
num_dimensions = 1
lower_limit = np.array([0.0])
upper_limit = np.array([5.0])
resolution = np.array([.01])

#initialise environment, genome and repertoire generator
prosthetic_env = ProstheticEAEnv(seed=seed,max_time_steps_qd=max_time_steps_qd,max_time_steps_task=max_time_steps_task,
								 joint_error_margin=joint_error_margin,visualize=visualize)
prosthetic_genome = ProstheticGenome(seed=seed)
map_elites = MAP_Elites(env=prosthetic_env,qd_function=prosthetic_env.qd_steady_runner,genome_constructor=ProstheticGenome,seed=seed,
	selector=Curiosity_Driven_Selector(),num_dimensions=num_dimensions,lower_limit=lower_limit,upper_limit=upper_limit,
	resolution=resolution,batch_size=batch_size)

test_utils.print_heading("Performance and behavior after an evaluation on environment")
behavior,quality = prosthetic_env.evaluate_quality_diversity_fitness(prosthetic_env.qd_steady_runner,prosthetic_genome,visualise) 
print(behavior,quality)

container = map_elites.container
test_utils.print_heading("Number of bins in grid container and check grid initialisation")
print(container.num_bins)
print(container.grid)

test_utils.print_heading("Test grid container functions")
print("is genome high quality. should return true as bin empty --- ",container.is_high_quality(behavior,quality))
container.add_genome(genome=prosthetic_genome,behavior=behavior,quality=quality)
print("Allocated bin ",container.get_bin(behavior))
print("Total quality ",container.total_quality)
print("Max quality ",container.max_quality)
print("Max quality bin ",container.max_quality_bin)
print("Number of genomes in the container",container.num_genomes)
print("Normalised total quality ",container.total_quality/container.num_genomes)

test_utils.print_heading("Add 5 more random genomes")
for i in range(5):
	prosthetic_genome.mutate()
	behavior,quality = prosthetic_env.evaluate_quality_diversity_fitness(prosthetic_env.qd_steady_runner,prosthetic_genome,visualise)
	container.add_genome(genome=prosthetic_genome,behavior=behavior,quality=quality)
print("Total quality ",container.total_quality)
print("Max quality ",container.max_quality)
print("Max quality bin ",container.max_quality_bin)
print("Number of genomes in the container",container.num_genomes)
print("Normalised total quality ",container.total_quality/container.num_genomes)

test_utils.print_heading("Select samples uniformly from container")
uniform_random_selector = Uniform_Random_Selector()
sampled_population = uniform_random_selector.select(container.grid,3)
print(sampled_population)

test_utils.print_heading("Select samples based on curiosity scores from container")
curiosity_driven_selector = Curiosity_Driven_Selector()
sampled_population = curiosity_driven_selector.select(container.grid,3)
print(sampled_population)

## update curiosity of particular elements to see bias
new_genome_details = sampled_population[0][1]
new_genome_details["curiosity"] = 1000
container.update_bin(bin_index=sampled_population[0][0],genome_details=new_genome_details)
test_utils.print_heading("Test biased curiosity. should output first element of previous test most probably")
for i in range(5):
	sampled_population = curiosity_driven_selector.select(container.grid,1)


#### generate repertoire test
map_elites.generate_repertoire(num_iterations=2,save_dir=None,save_freq=1,visualise=False)
