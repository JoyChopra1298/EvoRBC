from evo_rbc.main.ant_map_elites.common import get_MAPElites
import pickle

load_path = "output_3/ant_map_elites_repertoire_2225.pkl"

map_elites = get_MAPElites()
map_elites.load_repertoire(load_path)

def play(bin_index):
	if bin_index not in map_elites.container.grid:
		print("not present")
		return
	genome = map_elites.container.grid[bin_index]["genome"]
	# obs = map_elites.env.reset()
	# print(obs)
	# expert_trajectory = []
	# expert_trajectory.append(obs)
	# for i in range(500):
	# 	action = []
	# 	for joint_index in range(map_elites.env.action_space.shape[0]):
	# 		action.append(genome.control_function(joint_index=joint_index, time_step=i))
	# 	observation, _, _, _ = map_elites.env.step(action)
	# 	expert_trajectory.append(observation)
	# with open("ant_obs_forward_expert.pkl", 'wb') as f:
	# 	pickle.dump([expert_trajectory],f)

	behavior,quality = map_elites.env.evaluate_quality_diversity_fitness(qd_function=map_elites.qd_function,
					primitive_genome=genome,visualise=True)
	print(behavior,quality,map_elites.container.get_bin(behavior))
	print(map_elites.container.grid[bin_index]["quality"],bin_index)

behavior = map_elites.container.min_quality_bin

play((31,15))
# map_elites.view_metrics("num_new_genomes")


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