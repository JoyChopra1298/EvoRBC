from evo_rbc.main.ant_map_elites.common import get_MAPElites

load_path = "ant_map_elites_repertoire_150.pkl"

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

play((15,31))
map_elites.view_metrics("container_metrics","Number of genomes")


# map_elites.print_metrics()
# print(sum(map_elites.metrics["total_quality_increase_by_better_genomes"]))
# print(map_elites.metrics)

'''repo2
good/ok
31,31
25,15
31,16
16,28
1,28
1,1
bad
30,30
1,30
1,29
'''

##repo1
#30,30
#64,0
#23,34
#22,35
#99,50