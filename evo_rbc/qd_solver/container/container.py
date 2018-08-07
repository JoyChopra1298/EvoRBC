class Container:

	def __init__(self,env):
		self.env = env

	def add_genome(self,genome):
		"""adds the genome to container"""
		raise NotImplementedError

	def calculate_novelty(self,genome):
		"""calculate novelty for the genome"""
		raise NotImplementedError

	def update_container(self):
		"""update novelty score for all genomes in the container"""
		raise NotImplementedError

	def calculate_total_quality(self):
		"""calculate total quality of the conatiner, metric to compare different containers"""
		raise NotImplementedError

	def calculate_max_quality(self):
		"""calculate maximum quality of genome in the conatiner, metric to compare different containers"""
		raise NotImplementedError

	def evaluate_quality_diversity_fitness():
		raise NotImplementedError
