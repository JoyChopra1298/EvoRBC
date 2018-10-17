from evo_rbc.main.prosthetic_map_elites.common import get_MAPElites

load_path = "map_elites_repertoire_330.pkl"


def play(bin_index):
	if bin_index not in map_elites.container.grid:
		print("not present")
		return
	genome = map_elites.container.grid[bin_index]["genome"]
	behavior,quality = map_elites.env.evaluate_quality_diversity_fitness(qd_function=map_elites.qd_function,
					primitive_genome=genome,visualise=True)
	print(behavior,quality,map_elites.container.get_bin(behavior))
	# print(map_elites.container.grid[bin_index]["quality"],bin_index)
	return quality

max_quality = -1000.0
max_quality_seed = 0
for seed in range(1,100):
	map_elites = get_MAPElites(seed=seed)
	map_elites.load_repertoire(load_path)
	print('Seed ------------ ',seed)
	quality = play((2,))
	if(quality > max_quality):
		max_quality = quality
		max_quality_seed = seed

print("Max qaulity seed and quality ---",max_quality_seed,max_quality)

# total_quality = 0
# for bin_index,genome_details in map_elites.container.grid.items():
# 	total_quality += 1
# print(total_quality,map_elites.container.num_genomes)
# genome_details = map_elites.container.grid[(0,)]
# map_elites.container.add_genome(genome_details["genome"],0.041,genome_details["quality"])
# # map_elites.container.update_bin((4,),genome_details)
# # play((4,))
# map_elites.container.add_genome(genome_details["genome"],0.041,genome_details["quality"])

# # map_elites.view_metrics("num_new_genomes")
