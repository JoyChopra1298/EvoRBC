from .genome import Genome
import gym.spaces as spaces
import numpy as np 
from collections import OrderedDict
from scipy.stats import truncnorm

class AntGenome(Genome):
	#for each joint(2 for each leg, total 4*2 = 8). control function specific
	## values taken according to EvoRBC paper, see readme for details.some values adjusted
	amplitude_low = -0.5
	amplitude_high = 0.5
	phase_low = 0
	phase_high = 1
	smoothness_low = (np.pi)/4
	smoothness_high = (np.pi/4) + 2*(np.pi)
	epsilon_low = -0.5
	epsilon_high = 0.5
	# on exponential scale, example -1 would mean 1e-1
	control_frequency_low = -3.0 
	control_frequency_high = -1.0

	parameter_space = spaces.Dict({
		"amplitude_space":spaces.Box(low=amplitude_low,high=amplitude_high,shape=(8,1),dtype=np.float32),
		"phase_space":spaces.Box(low=phase_low,high=phase_high,shape=(8,1),dtype=np.float32),
		"smoothness_space":spaces.Box( low=np.array([smoothness_low]), high=np.array([smoothness_high]),dtype=np.float32),
		"epsilon_space":spaces.Box(low=epsilon_low,high=epsilon_high,shape=(8,1),dtype=np.float32),
		"control_frequency_space":spaces.Box( low=np.array([control_frequency_low]), high=np.array([control_frequency_high]),dtype=np.float32),})

	def __init__(self,parameters=None,seed=1):
		super().__init__(parameters)
		self.seed = seed
		## Set random number seed for all scipy and numpy operations so that experiments can be reproduced
		np.random.seed(seed=seed)

	def mutate(self,sigma=1):
		"""Mutate the genome using a truncated gaussian"""
		for key,value in self.parameters.items():
			low = self.parameter_space.spaces[key+"_space"].low
			high = self.parameter_space.spaces[key+"_space"].high
			mu = value
			## truncnorm's clipped parameters are according to a standard normal so scale accordingly. refer their documentation for more details
			standard_low = (low - mu) / sigma
			standard_high = (high - mu) / sigma
			self.parameters[key] =  np.array(truncnorm.rvs(standard_low,standard_high , loc=mu, scale=sigma),dtype=np.float32)
		
	def crossover(self,mate_genome):
		"""Cross self with mate_genome. Take parameters from either parent randomly"""
		child_parameters = OrderedDict()
		for key,value in self.parameters.items():
			bit_mask = np.random.choice(2,value.shape)
			child_parameters[key] = np.array(bit_mask*value + (1 - bit_mask)*mate_genome.parameters[key],dtype=np.float32)
		return Genome(child_parameters)	

	def sample_random_genome(self):
		ordered_dict = self.parameter_space.sample()

		# Remove trailing "_space" from keys since returning parameters for a single genome
		return OrderedDict([(key[:-6],value) for key,value in ordered_dict.items()])
	
	def control_function(self,joint_index,time_step):
		"""Returns the action for corresponding joint from genome parameters"""
		angle = time_step*np.power(10,self.parameters["control_frequency"]) + self.parameters["phase"][joint_index]
		sine = np.sin(2*(np.pi)*(angle))
		tanh = (np.tanh(self.parameters["smoothness"])* sine) 
		return self.parameters["epsilon"][joint_index] + self.parameters["amplitude"][joint_index]* tanh

	def save_genome(self,save_path):
		"""Saves the genome parameters in save_path"""
		raise NotImplementedError
	
	def load_genome(self,load_path):
		"""Loads the genome parameters from load_path"""
		raise NotImplementedError
		
