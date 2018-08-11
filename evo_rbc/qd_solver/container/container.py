import numpy as np
class Container:

	def __init__(self,logger):
		self.logger = logger
		self.num_genomes = 0

		"""metrics to compare different containers"""
		self.max_quality = -np.inf
		self.total_quality = 0

	def get_bin(self,behavior):
		"""get the bin corresponding to a particular behavior"""
		raise NotImplementedError

	def is_high_quality(self,behavior,quality):
		"""check if genome has high quality then current genome for the same behavior. also true if bin is empty"""
		raise NotImplementedError

	def add_genome(self,genome,behavior):
		"""add the genome to container"""
		raise NotImplementedError

	def update_bin(self,bin_index,genome_details):
		"""updates the entry in bin. genome details consists of a dictionary of genome parameters"""
		raise NotImplementedError

	def save_container(self,save_dir):
		raise NotImplementedError

	def load_container(self,load_path):
		raise NotImplementedError
		