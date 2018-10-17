import logging
import numpy as np

from evo_rbc.qd_solver.map_elites import MAP_Elites
from evo_rbc.genome.prosthetic_genome import ProstheticGenome
from evo_rbc.env.prosthetic_env import ProstheticEAEnv
from evo_rbc.qd_solver.container.grid import Grid
from evo_rbc.qd_solver.selector.curiosity_driven_selector import Curiosity_Driven_Selector


##################################
#Constants/hyperparameters
batch_size = 100
max_time_steps_qd=300
max_time_steps_task=2000
visualise = False

joint_error_margin = 0.1


#Grid details
num_dimensions = 1
lower_limit = np.array([0.0])
upper_limit = np.array([5.0])
resolution = np.array([.01])


def get_MAPElites(seed=1):
	
	#initialise environment, genome and repertoire generator
	prosthetic_env = ProstheticEAEnv(seed=seed,max_time_steps_qd=max_time_steps_qd,max_time_steps_task=max_time_steps_task,
		joint_error_margin=joint_error_margin)
	map_elites = MAP_Elites(env=prosthetic_env,qd_function=prosthetic_env.qd_steady_runner,genome_constructor=ProstheticGenome,seed=seed,
		selector=Curiosity_Driven_Selector(),num_dimensions=num_dimensions,lower_limit=lower_limit,upper_limit=upper_limit,
		resolution=resolution,batch_size=batch_size)

	return map_elites

