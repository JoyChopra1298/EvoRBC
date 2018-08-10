class Repertoire_Generator:

	def __init__(self,env,genome,batch_size):
		self.env = env
		self.genome = genome
		self.batch_size = batch_size
		self.current_generation = 1

	def generate_repertoire(self,save_dir,num_iterations,save_freq,visualise):
		"""
		Generate a repertoire by applying a quality diversity algorithm ran for num_iterations number of iterations
		Save the generated repo in the save_dir periodically after every save_freq iterations
		"""
		raise NotImplementedError

	def print_metrics(self):
		"""Print metrics about the generated repertoire"""
		raise NotImplementedError

	def save_repertoire(self,save_path):
		"""Saves the repertoire in save_path"""
		raise NotImplementedError

	def load_repertoire(self,load_path):
		"""Loads the repertoire from load_path"""
		raise NotImplementedError

