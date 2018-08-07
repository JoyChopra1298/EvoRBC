from .genome import Genome
import gym
import numpy as np 

class AntGenome(Genome):
	#for each joint(2 for each leg, total 4*2 = 8). control function specific
	## values taken according to EvoRBC paper, see readme for details.some values adjusted
	
	parameter_space = spaces.Dict(
		"amplitude_space":spaces.Box(-0.5,0.5,8),
		"phase_space":spaces.Box(0,1,8),
		"smoothness_space":spaces.Box( (np.pi)/4, (np.pi/4) + 2*(np.pi)),
		"epsilon_space":spaces.Box(-0.5,0.5,8))
	frequency = 1

	def __init__(self,parameters):
		super().__init__(parameters)

	def isValid():
		"""Check if the genome parameters satisfy respective constraints"""
		raise NotImplementedError
		
	def mutate():
		"""Mutate the genome according to some distribution like gaussian"""
		raise NotImplementedError

	def sample_random_genome(self):
		return self.parameter_space.sample()
	
	def control_function(parameters,joint_index,time_step):
		"""Returns the action for corresponding joint from genome parameters"""
		angle = time_step*frequency + parameters["phase"][joint_index]
		sine = np.sin(2*(np.pi)*(angle))
		tanh = (np.tanh(parameters["smoothness"])* sine) 
		return parameters["epsilon"][joint_index] + parameters["amplitude"][joint_index]* tanh

	def save_genome(self,save_path):
		"""Saves the genome in save_path"""
		raise NotImplementedError
	
	def load_genome(self,load_path):
		"""Loads the genome from load_path"""
		raise NotImplementedError
		
