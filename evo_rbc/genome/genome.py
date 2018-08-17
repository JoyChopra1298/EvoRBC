import copy,logging
import evo_rbc.main.utils as utils
import numpy as np 
from collections import OrderedDict
from scipy.stats import truncnorm

class Genome:
	parameter_space = None

	def __init__(self,parameters=None,seed=1):
		self.logger = utils.getLogger()
		if(parameters==None):
			self.parameters = self.sample_random_genome()
		else:
			self.parameters = parameters
		self.seed = seed
		## Set random number seed for all scipy and numpy operations so that experiments can be reproduced
		np.random.seed(seed=seed)
		# self.logger.debug("Created random genome"+str(self.parameters))
	
	def mutate(self,sigma=0.01):
		"""Mutate the genome using a truncated gaussian"""
		# self.logger.debug("Mutating genome. Current parameters - "+str(self.parameters))
		for key,value in self.parameters.items():
			low = self.parameter_space.spaces[key+"_space"].low
			high = self.parameter_space.spaces[key+"_space"].high
			mu = value
			## truncnorm's clipped parameters are according to a standard normal so scale accordingly. refer their documentation for more details
			standard_low = (low - mu) / sigma
			standard_high = (high - mu) / sigma
			self.parameters[key] =  np.array(truncnorm.rvs(standard_low,standard_high , loc=mu, scale=sigma),dtype=np.float32)
		# self.logger.debug("Mutation finished. Mutated parameters "+str(self.parameters))
		
	def crossover(self,mate_genome):
		"""Cross self with mate_genome. Take parameters from either parent randomly"""
		# self.logger.debug("Crossing genomes with parameters "+str(self.parameters)+str(mate_genome.parameters))
		child_parameters = OrderedDict()
		for key,value in self.parameters.items():
			bit_mask = np.random.choice(2,value.shape)
			child_parameters[key] = np.array(bit_mask*value + (1 - bit_mask)*mate_genome.parameters[key],dtype=np.float32)
		# self.logger.debug("Child parameters "+str(child_parameters))
		return self.__class__(parameters=child_parameters,seed=self.seed)	

	def sample_random_genome(self):
		ordered_dict = self.parameter_space.sample()

		# Remove trailing "_space" from keys since returning parameters for a single genome
		return OrderedDict([(key[:-6],value) for key,value in ordered_dict.items()])

	def __deepcopy__(self, memo):
		cls = self.__class__
		result = cls.__new__(cls)
		memo[id(self)] = result
		for k, v in self.__dict__.items():
			if(k=="logger"):
				setattr(result, k, v)
				continue
			setattr(result, k, copy.deepcopy(v, memo))
		return result
		
	def __getstate__(self):
		excluded_subnames = ["logger"]
		state = {}
		for k, v in self.__dict__.items():
			if(k=="logger"):
				continue
			state[k] = v
		return state

	def __setstate__(self, state):
		for k,v in state.items():
			self.__dict__[k] = v
		self.logger = utils.getLogger()
				
