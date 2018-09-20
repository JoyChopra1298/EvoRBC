from evo_rbc.genome.prosthetic_genome import ProstheticGenome
from evo_rbc.env.prosthetic_env import ProstheticEAEnv
import evo_rbc.main.utils as test_utils

logger = test_utils.getLogger()
prosthetic_env = ProstheticEAEnv(visualize=True)
prosthetic_env.reset()
prosthetic_genome = ProstheticGenome()

total_reward = 0.0
for i in range(301):
	if(i%50==0):
		print(total_reward,i/50)
		prosthetic_genome.mutate(0.5)
		prosthetic_env.reset()
		total_reward = 0.0
	action = [0.0]*19
	for j in range(19):
		action[j] = prosthetic_genome.control_function(j,i%500)
	obs,reward,done,info = prosthetic_env.step(action)
	total_reward += reward
	prosthetic_env.render()