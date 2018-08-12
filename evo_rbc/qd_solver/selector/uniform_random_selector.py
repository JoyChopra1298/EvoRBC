from .selector import Selector
import numpy as np

class Uniform_Random_Selector(Selector):

	def __init__(self):
		super().__init__()

	def select(self,container,num_samples):
		"""select num_samples from container. container is dict of dicts {"bin_index":{"genome","quality",other details},...}
		print a warning if num_samples is more than container size"""
		population = [[bin_index,genome_details] for bin_index,genome_details in container.items()]
		population_size = len(population)
		if(num_samples>population_size):
			self.logger.warning("from Uniform_Random selector- number of samples queried from container exceed it's size,"+
			 "returning all genomes")
			num_samples = population_size
		selected_indices = np.random.choice(a=population_size,size=num_samples,replace=False)
		return [population[index] for index in selected_indices]