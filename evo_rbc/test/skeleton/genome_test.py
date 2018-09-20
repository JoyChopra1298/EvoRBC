from evo_rbc.genome.prosthetic_genome import ProstheticGenome
import evo_rbc.main.utils as test_utils

prosthetic_genome = ProstheticGenome()

control_duration = 10
num_generations = 1
muscle_index = 16

test_utils.print_heading("Parameter space")
print(prosthetic_genome.parameter_space)

test_utils.print_heading("Random genome sample")
print(prosthetic_genome.parameters)

test_utils.print_heading("Control function for muscle "+str(muscle_index)+" for "+str(control_duration)+" timesteps")
for time_step in range(control_duration):
	print(prosthetic_genome.control_function(muscle_index,time_step))

for i in range(num_generations):
	prosthetic_genome.mutate()

	test_utils.print_heading("Mutated genome for generation "+str(i+1))
	print(prosthetic_genome.parameters)

	test_utils.print_heading("Control function for muscle "+str(muscle_index)+" for "+str(control_duration)+" timesteps and generation "+str(i+1))
	for time_step in range(control_duration):
		print(prosthetic_genome.control_function(muscle_index,time_step))

test_utils.print_heading("Genome 2")
prosthetic_genome_2 = ProstheticGenome()
print(prosthetic_genome_2.parameters)

test_utils.print_heading("Child Genome")
child_genome = prosthetic_genome.crossover(prosthetic_genome_2)
print(child_genome.parameters)

test_utils.print_heading("Showing plot for muscle " + child_genome.muscle_dict[muscle_index])
child_genome.plot_control_function(num_timesteps=100,muscle_index=muscle_index)
child_genome.mutate(0.5)
child_genome.plot_control_function(num_timesteps=100,muscle_index=muscle_index)

