class Selector:
	
	def __init__(self,logger):
		self.logger = logger
		
	def select(self,container,num_samples):
		"""select a batch of genomes from container"""
		raise NotImplementedError