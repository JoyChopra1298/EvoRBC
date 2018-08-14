from .genome import Genome
import gym.spaces as spaces
import numpy as np 
from collections import OrderedDict
from scipy.stats import truncnorm

class ProstheticGenome(Genome):
	
	num_muscles = 19
	# parameter_space = spaces.Dict({
	# 	"amplitude_space":spaces.Box(low=amplitude_low,high=amplitude_high,shape=(num_joints,1),dtype=np.float32),
	# 	"phase_space":spaces.Box(low=phase_low,high=phase_high,shape=(num_joints,1),dtype=np.float32),
	# 	"smoothness_space":spaces.Box( low=np.array([smoothness_low]), high=np.array([smoothness_high]),dtype=np.float32),
	# 	"epsilon_space":spaces.Box(low=epsilon_low,high=epsilon_high,shape=(num_joints,1),dtype=np.float32),
	# 	"control_frequency_space":spaces.Box( low=np.array([control_frequency_low]), high=np.array([control_frequency_high]),dtype=np.float32),})

	def __init__(self,parameters=None,seed=1):
		super().__init__(parameters=parameters,seed=seed)

	def control_function(self,muscle_index,time_step):
		"""Calculate control actions for a particular muscle at time step time_step"""
		raise NotImplementedError
