from evo_rbc.genome.ant_genome import AntGenome
from evo_rbc.env.ant_env import AntEAEnv
import evo_rbc.main.utils as test_utils

logger = test_utils.getLogger()
ant_env = AntEAEnv()
ant_env.reset()
ant_genome = AntGenome()
for i in range(5000):
	if(i%500==0):
		ant_genome.mutate(4)
	action = [0.0]*8
	for j in range(8):
		action[j] = ant_genome.control_function(j,i%500)
	obs = ant_env.step(action)
	ant_env.render()