
class Arbitrator:

	def __init__(self,env,repertoire,mapper):
		self.env = env
		self.repertoire = repertoire
		self.mapper = mapper
		self.current_generation = 1

	def evolve(self,save_dir,num_generations,save_freq):
		"""evolve the arbitrator for num_generations and save it after every save_freq generations"""
		raise NotImplementedError
	
	def save(self,save_path):
		"""saves the arbitrator in save_path"""
		raise NotImplementedError
	
	def load(self,load_path):
		"""loads the arbitrator from load_path"""
		raise NotImplementedError
