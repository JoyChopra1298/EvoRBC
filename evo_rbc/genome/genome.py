import copy

class Genome:
	parameter_space = None

	def __init__(self,logger,parameters=None):
		self.logger = logger
		if(parameters==None):
			self.parameters = self.sample_random_genome()
		else:
			self.parameters = parameters
	
	def mutate(self):
		"""Mutate the genome according to some distribution"""
		raise NotImplementedError

	def crossover(self,mate_genome):
		"""Cross self with mate_genome"""
		raise NotImplementedError

	def sample_random_genome(self):
		"""Samples a random set of parameters from the parameter space"""
		raise NotImplementedError
	
	def control_function(self,joint_index,time_step):
		"""Calculate control actions for a particular joint at time step time_step"""
		raise NotImplementedError
	
	def save_genome(self,save_path):
		"""Saves the genome in save_path"""
		raise NotImplementedError
	
	def load_genome(self,load_path):
		"""Loads the genome from load_path"""
		raise NotImplementedError

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
		

		
