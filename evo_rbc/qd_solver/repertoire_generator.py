import evo_rbc.main.utils as utils

class Repertoire_Generator:

	def __init__(self,env,genome_constructor,batch_size,seed=1):
		self.env = env
		self.genome_constructor = genome_constructor
		self.batch_size = batch_size
		self.seed = seed
		self.logger = utils.getLogger()
		self.current_iteration = 1

	def generate_repertoire(self,num_iterations,save_dir,save_freq,visualise):
		"""
		Generate a repertoire by applying a quality diversity algorithm ran for num_iterations number of iterations
		Save the generated repo in the save_dir periodically after every save_freq iterations
		"""
		raise NotImplementedError
	
	def log_metrics(self):
		"""Logs metrics about the generated repertoire"""
		raise NotImplementedError

	def print_metrics(self):
		"""Print metrics about the generated repertoire"""
		raise NotImplementedError

	def save_repertoire(self,save_file_path):
		"""Saves the repertoire in save_path"""
		raise NotImplementedError

	def load_repertoire(self,load_file_path):
		"""Loads the repertoire from load_path"""
		raise NotImplementedError

