import opensim as osim
from osim.http.client import Client
from osim.env import ProstheticsEnv

from evo_rbc.main.prosthetic_map_elites.common import get_MAPElites

load_path = "map_elites_repertoire_50.pkl"

map_elites = get_MAPElites()
map_elites.load_repertoire(load_path)

# Settings
remote_base = "http://grader.crowdai.org:1729"
crowdai_token = "f5969a7bb0466e0da072c72d6eb6d667"

client = Client(remote_base)

# Create environment
observation = client.env_create(crowdai_token,env_id="ProstheticsEnv")

# IMPLEMENTATION OF YOUR CONTROLLER
# my_controller = ... (for example the one trained in keras_rl)


def my_controller(observation,time_step):
	bin_index = (0,)
	genome = map_elites.container.grid[bin_index]["genome"]
	action = []
	for muscle_index in range(19):
		action.append(genome.control_function(muscle_index=muscle_index,time_step=time_step)[0])
	return action

i = 0 
total_reward = 0
time_step = 0
while True:
	time_step+=1
	[observation, reward, done, info] = client.env_step(my_controller(observation,time_step), True)
	total_reward+=reward
	print(i,total_reward)
	i+=1
	if done:
		observation = client.env_reset()
		if not observation:
			break

client.submit()