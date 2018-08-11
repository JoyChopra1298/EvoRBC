from .repertoire_generator import Repertoire_Generator
from .container.grid import Grid
import copy
import numpy as np

class MAP_Elites(Repertoire_Generator):

	def __init__(self,env,qd_function,genome_constructor,selector,num_dimensions,lower_limit,upper_limit,resolution,batch_size,logger,seed=1):
		self.container = Grid(num_dimensions=num_dimensions,lower_limit=lower_limit,upper_limit=upper_limit,resolution=resolution,logger=logger)
		self.selector = selector
		self.qd_function = qd_function
		super().__init__(env=env,genome_constructor=genome_constructor,batch_size=batch_size,seed=seed,logger=logger)
		
	def generate_repertoire(self,num_iterations,save_dir,save_freq,visualise):
		""" generate a random population initially, generating double the batch_size to increase the probability that
		 that at least batch_size elements get added"""
		self.logger.info("Initialising repertoire with random population")
		for i in range(2*self.batch_size):
			random_genome = self.genome_constructor(seed=self.seed,logger=self.logger)
			behavior,quality = self.env.evaluate_quality_diversity_fitness(qd_function=self.qd_function,
				primitive_genome=random_genome,visualise=visualise)
			if(self.container.is_high_quality(behavior=behavior,quality=quality)):
				self.container.add_genome(genome=random_genome,behavior=behavior,quality=quality)
		self.log_metrics()

		## to do vary stdev while training based on metrics/ include crossover/ add save load
		mutation_stdev = 1
		for iteration in range(num_iterations):
			self.logger.info("Iteration "+str(iteration))
			parents = self.selector.select(self.container.grid,self.batch_size)
			"""len(parents) used instead of batch size since it is possible to not have a complete batch from the container"""
			for i in range(len(parents)):
				parent_genome = parents[i][1]["genome"]
				child_genome = copy.deepcopy(parent_genome)
				child_genome.mutate(sigma=mutation_stdev)
				behavior,quality = self.env.evaluate_quality_diversity_fitness(qd_function=self.qd_function,
					primitive_genome=child_genome,visualise=visualise)
				self.logger.debug("parent_curiosity before "+str(parents[i][1]["curiosity"]))
				if(self.container.is_high_quality(behavior=behavior,quality=quality)):
					"""Note that it is important to update parent before adding child as if done in reverse order then parent might 
					replace a high quality child showing same behavior"""
					parents[i][1]["curiosity"] *= self.container.curiosity_multiplier
					self.container.update_bin(bin_index=parents[i][0],genome_details=parents[i][1])
					self.container.add_genome(genome=child_genome,behavior=behavior,quality=quality)
					self.logger.debug("Adding child")
				else:
					parents[i][1]["curiosity"] /= self.container.curiosity_multiplier
					np.clip(a=parents[i][1]["curiosity"],a_min=self.container.min_curiosity,a_max=np.inf)
					self.container.update_bin(bin_index=parents[i][0],genome_details=parents[i][1])
				self.logger.debug("parent_curiosity after "+str(parents[i][1]["curiosity"]))
			self.log_metrics()

	def log_metrics(self):
		self.logger.info("Repertoire Metrics")
		self.logger.info("Total quality "+str(self.container.total_quality))
		self.logger.info("Max quality "+str(self.container.max_quality))
		self.logger.info("Max quality bin "+str(self.container.max_quality_bin))
		self.logger.info("Number of genomes in the container "+str(self.container.num_genomes))
		self.logger.info("Normalised total quality "+str(self.container.total_quality/self.container.num_genomes))
	
	def save_repertoire(self,save_path):
		pass

	def load_repertoire(self,load_path):
		pass