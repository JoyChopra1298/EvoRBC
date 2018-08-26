from evo_rbc.main.prosthetic_map_elites.common import get_MAPElites

load_path = "map_elites_repertoire_25.pkl"

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

play((61,))
# map_elites.view_metrics("num_new_genomes")
