import logging
import numpy as np

from evo_rbc.qd_solver.map_elites import MAP_Elites
from evo_rbc.genome.ant_genome import AntGenome
from evo_rbc.env.ant_env import AntEAEnv
from evo_rbc.qd_solver.container.grid import Grid
from evo_rbc.qd_solver.selector.curiosity_driven_selector import Curiosity_Driven_Selector


##################################
#Constants/hyperparameters
seed = 1
batch_size = 100
max_time_steps_qd=500
max_time_steps_task=2000
visualise = False

#Grid details
num_dimensions = 2
lower_limit = np.array([-0.2,-0.2])
upper_limit = np.array([0.2,0.2])
resolution = np.array([.0125,.0125])


#initialise environment, genome and repertoire generator
ant_env = AntEAEnv(seed=seed,max_time_steps_qd=max_time_steps_qd,max_time_steps_task=max_time_steps_task)
ant_genome = AntGenome(seed=seed)
map_elites = MAP_Elites(env=ant_env,qd_function=ant_env.qd_steady_runner,genome_constructor=AntGenome,seed=seed,
	selector=Curiosity_Driven_Selector(),num_dimensions=num_dimensions,lower_limit=lower_limit,upper_limit=upper_limit,
	resolution=resolution,batch_size=batch_size)

def get_MAPElites():
	return map_elites

