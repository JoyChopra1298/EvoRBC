class Genome:
	parameter_space = None

	def __init__(self,parameters):
		self.parameters = parameters
	
	def isValid():
		"""Check if the genome parameters satisfy respective constraints"""
		raise NotImplementedError
		
	def mutate():
		"""Mutate the genome according to some distribution like gaussian"""
		raise NotImplementedError

	def sample_random_genome():
		"""Samples a random set of parameters from the parameter space"""
		raise NotImplementedError
	
	def control_function():
		"""The function whose parameters are represented by the genome"""
		raise NotImplementedError
	
	def save_genome(self,save_path):
		"""Saves the genome in save_path"""
		raise NotImplementedError
	
	def load_genome(self,load_path):
		"""Loads the genome from load_path"""
		raise NotImplementedError
		
