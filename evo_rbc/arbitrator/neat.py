from .arbitrator import Arbitrator

class NEAT(Arbitrator):

	def __init__(self,env,repertoire,mapper):
		super().__init__(env,repertoire,mapper)
	
	def evolve(self,save_dir,num_generations,save_freq):
		pass
	
	def save(self,save_path):
		pass
	
	def load(self,load_path):
		pass
