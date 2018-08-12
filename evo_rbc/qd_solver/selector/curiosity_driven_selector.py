from .selector import Selector
import random
import numpy as np

class Curiosity_Driven_Selector(Selector):
	
	def __init__(self):
		super().__init__()
	
	def select(self,container,num_samples):
		"""Do a weighted selection based on curiosity scores. container is dict of dicts {"bin_index":{"genome","curiosity",other details},...}
		print a warning if num_samples is more than container size"""
		population = []
		curiosity = []
		for bin_index,bin_value in container.items():
			population.append([bin_index,bin_value])
			curiosity.append(bin_value["curiosity"])
		curiosity = np.array(curiosity)
		
		population_size = len(population)

		if(num_samples>population_size):
			self.logger.warning("from Curiosity selector- number of samples queried from container exceed it's size,"+
			 "returning all genomes")
			num_samples = population_size
		
		#normalise so that it becomes a probbility distribution
		curiosity = curiosity/np.sum(curiosity)
		selected_indices = np.random.choice(a=population_size,size=num_samples,replace=False,p=curiosity)
		return [population[index] for index in selected_indices]
