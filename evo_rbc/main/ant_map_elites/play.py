from evo_rbc.main.ant_map_elites.common import get_MAPElites

load_path = "output/ant_map_elites_repertoire_22.pkl"

map_elites = get_MAPElites()
map_elites.load_repertoire(load_path)

def play(bin_index):
	genome = map_elites.container.grid[bin_index]["genome"]
	behavior,quality = map_elites.env.evaluate_quality_diversity_fitness(qd_function=map_elites.qd_function,
					primitive_genome=genome,visualise=True)
	print(behavior,quality,map_elites.container.get_bin(behavior))

max_behavior = map_elites.container.max_quality_bin
play(max_behavior)
map_elites.print_metrics()
print(map_elites.metrics)