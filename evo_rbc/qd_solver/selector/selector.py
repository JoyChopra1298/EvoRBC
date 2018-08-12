import evo_rbc.main.utils as utils

class Selector:
	
	def __init__(self):
		self.logger = utils.getLogger()
		
	def select(self,container,num_samples):
		"""select a batch of genomes from container"""
		raise NotImplementedError

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
