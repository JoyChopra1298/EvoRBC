from .repertoire_generator import Repertoire_Generator
from .container.grid import Grid
import copy
import numpy as np

class MAP_Elites(Repertoire_Generator):

	def __init__(self,env,qd_function,genome_constructor,selector,num_dimensions,lower_limit,upper_limit,resolution,batch_size=100):
		self.container = Grid(num_dimensions=num_dimensions,lower_limit=lower_limit,upper_limit=upper_limit,resolution=resolution)
		self.selector = selector
		self.qd_function = qd_function
		super().__init__(env=env,genome_constructor=genome_constructor,batch_size=batch_size)
		
	def generate_repertoire(self,save_dir,num_iterations,save_freq,visualise):
		""" generate a random population initially, generating double the batch_size to increase the probability that
		 that at least batch_size elements get added"""
		for i in range(2*self.batch_size):
			random_genome = self.genome_constructor()
			behavior,quality = self.env.evaluate_quality_diversity_fitness(qd_function=self.qd_function,
				genome=random_genome,visualise=visualise)
			if(self.container.is_high_quality(behavior=behavior,quality=quality)):
				self.container.add_genome(genome=random_genome,behavior=behavior,quality=quality)
		
		## to do vary this while training based on metrics/ include crossover/ add save load
		mutation_stdev = 1
		for iteration in range(num_iterations):
			parents = self.selector.select(self.container,self.batch_size)
			"""len(parents) used instead of batch size since it is possible to not have a complete batch from the container"""
			for i in range(len(parents)):
				parent_genome = parents[i][1]["genome"]
				child_genome = copy.deepcopy(parent_genome).mutate(sigma=mutation_stdev)
				behavior,quality = self.env.evaluate_quality_diversity_fitness(qd_function=self.qd_function,
					genome=child_genome,visualise=visualise)
				if(self.container.is_high_quality(behavior=behavior,quality=quality)):
					self.container.add_genome(genome=child_genome,behavior=behavior,quality=quality)
					parents[i][1]["curiosity"] *= self.container.curiosity_multiplier
				else:
					parents[i][1]["curiosity"] /= self.container.curiosity_multiplier
					np.clip(a=parents[i][1]["curiosity"],a_min=self.container.min_curiosity,a_max=np.inf)
				self.container.update_bin(bin_index=parents[i][0],genome_details=parents[i][1])

	def print_metrics(self):
		print("Total quality ",self.container.total_quality)
		print("Max quality ",self.container.max_quality)
		print("Max quality bin ",self.container.max_quality_bin)
		print("Number of genomes in the container",self.container.num_genomes)
		print("Normalised total quality ",self.container.total_quality/self.container.num_genomes)
	
	def save_repertoire(self,save_path):
		pass

	def load_repertoire(self,load_path):
		pass