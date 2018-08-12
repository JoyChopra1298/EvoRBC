import numpy as np
import evo_rbc.main.utils as utils

class Container:

	def __init__(self):
		self.logger = utils.getLogger()
		self.num_genomes = 0

		"""metrics to compare different containers"""
		self.max_quality = -np.inf
		self.min_quality = np.inf
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
		