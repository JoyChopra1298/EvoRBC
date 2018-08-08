from evo_rbc.genome.ant_genome import AntGenome

ant_genome = AntGenome()

control_duration = 10
num_generations = 0
heading_decorator = "\n-------------------------------------------\n"
joint_index = 1

def print_heading(heading):
	print(heading_decorator+heading+heading_decorator)

print_heading("Parameter space")
print(ant_genome.parameter_space)

print_heading("Random genome sample")
print(ant_genome.parameters)

print_heading("Control function for joint "+str(joint_index)+" for "+str(control_duration)+" timesteps")
for time_step in range(control_duration):
	print(ant_genome.control_function(joint_index,time_step))

for i in range(num_generations):
	ant_genome.mutate()

	print_heading("Mutated genome for generation "+str(i+1))
	print(ant_genome.parameters)

	print_heading("Control function for joint "+str(joint_index)+" for "+str(control_duration)+" timesteps and generation "+str(i+1))
	for time_step in range(control_duration):
		print(ant_genome.control_function(joint_index,time_step))

print_heading("Genome 2")
ant_genome_2 = AntGenome()
print(ant_genome_2.parameters)

print_heading("Child Genome")
child_genome = ant_genome.crossover(ant_genome_2)
print(child_genome.parameters)



