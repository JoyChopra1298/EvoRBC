from eaenv import EAenv

class Repertoire_Generator:

	def __init__(self,env,container,selector,mutation_rate):
		self.env = env
		self.contatiner = container
		self.selector = selector
		self.mutation_rate = mutation_rate
		self.current_generation = 1

	def generate_repertoire(self,save_dir,num_generations,save_freq):
		"""
		Generate a repertoire by applying a quality diversity algorithm ran for num_iterations number of iterations
		Save the generated repo in the save_dir periodically after every save_freq iterations
		"""
		raise NotImplementedError

	def save_repertoire(self,save_path):
		"""Saves the repertoire in save_path"""
		raise NotImplementedError

	def load_repertoire(self,load_path):
		"""Loads the repertoire from load_path"""
		raise NotImplementedError

