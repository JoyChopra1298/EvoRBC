from evo_rbc.main.prosthetic_map_elites.common import get_MAPElites

load_path = "map_elites_repertoire_5.pkl"

map_elites = get_MAPElites()
map_elites.load_repertoire(load_path)

def play(bin_index):
	if bin_index not in map_elites.container.grid:
		print("not present")
		return
	genome = map_elites.container.grid[bin_index]["genome"]
	behavior,quality = map_elites.env.evaluate_quality_diversity_fitness(qd_function=map_elites.qd_function,
					primitive_genome=genome,visualise=True)
	print(behavior,quality,map_elites.container.get_bin(behavior))
	print(map_elites.container.grid[bin_index]["quality"],bin_index)

behavior = map_elites.container.min_quality_bin

play((0,))


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
