from .repertoire_generator import Repertoire_Generator
from .container.grid import Grid
import copy,os
import numpy as np
import pickle

class MAP_Elites(Repertoire_Generator):

	def __init__(self,env,qd_function,genome_constructor,selector,num_dimensions,lower_limit,upper_limit,resolution,batch_size,seed=1):
		self.container = Grid(num_dimensions=num_dimensions,lower_limit=lower_limit,upper_limit=upper_limit,resolution=resolution)
		self.selector = selector
		self.qd_function = qd_function
		super().__init__(env=env,genome_constructor=genome_constructor,batch_size=batch_size,seed=seed)

		## metrics to measure growth for each iteration
		self.metrics = {"num_better_genome_found":[],
		"total_quality_increase_by_better_genomes":[],
		"num_new_genomes":[],"container_metrics":[]}
		
	def generate_repertoire(self,num_iterations,save_dir,save_freq,visualise,mutation_stdev=1):
		""" generate a random population initially, generating double the batch_size to increase the probability that
		 that at least batch_size elements get added"""
		if(self.current_iteration==1):
			self.logger.info("Initialising repertoire with random population")
			for i in range(2*self.batch_size):
				random_genome = self.genome_constructor(seed=self.seed)
				behavior,quality = self.env.evaluate_quality_diversity_fitness(qd_function=self.qd_function,
					primitive_genome=random_genome,visualise=visualise)
				if(self.container.is_high_quality(behavior=behavior,quality=quality)):
					self.container.add_genome(genome=random_genome,behavior=behavior,quality=quality)
			self.log_metrics()
			self.metrics["num_better_genome_found"].append(0)
			self.metrics["total_quality_increase_by_better_genomes"].append(0)
			self.metrics["num_new_genomes"].append(self.container.num_genomes)

		## to do vary stdev while training based on metrics/ include crossover
		for iteration in range(self.current_iteration,self.current_iteration+num_iterations):
			self.logger.info("Iteration "+str(iteration))
			parents = self.selector.select(self.container.grid,self.batch_size)
			
			## initialise metrics for current iteration
			num_better_genome_found_in_this_iteration = 0
			total_quality_increase_by_better_genomes_in_this_iteration = 0
			old_num_genomes = self.container.num_genomes

			#### paralellise this loop just like pi example
			"""len(parents) used instead of batch size since it is possible to not have a complete batch from the container"""
			for i in range(len(parents)):
				parent_genome = parents[i][1]["genome"]
				child_genome = copy.deepcopy(parent_genome)
				child_genome.mutate(sigma=mutation_stdev)
				behavior,quality = self.env.evaluate_quality_diversity_fitness(qd_function=self.qd_function,
					primitive_genome=child_genome,visualise=visualise)
				self.logger.debug("parent_curiosity before "+str(parents[i][1]["curiosity"]))
				
				bin_index = self.container.get_bin(behavior)				
				self.logger.debug("Child bin index "+str(bin_index))
				parent_bin_index = parents[i][0]
				self.logger.debug("Parent bin index "+str(parent_bin_index))

				### Check if child should be saved into the repertoire and update parent's curiosity score accordingly
				if(self.container.is_high_quality(behavior=behavior,quality=quality)):
					"""Note that it is important to update parent before adding child as if done in reverse order then parent might 
					replace a high quality child showing same behavior"""
					self.logger.debug("Adding child with behavior and quality"+str(behavior)+str(quality))
					### metrics when child replaces some genome
					if(bin_index in self.container.grid):
						old_genome_quality = self.container.grid[bin_index]["quality"]
						self.logger.debug("Old quality in same bin "+str(old_genome_quality))
						num_better_genome_found_in_this_iteration += 1
						total_quality_increase_by_better_genomes_in_this_iteration += quality - old_genome_quality

					parents[i][1]["curiosity"] *= self.container.curiosity_multiplier
					self.container.update_bin(bin_index=parents[i][0],genome_details=parents[i][1])
					self.container.add_genome(genome=child_genome,behavior=behavior,quality=quality)

				else:
					parents[i][1]["curiosity"] /= self.container.curiosity_multiplier
					parents[i][1]["curiosity"] = np.clip(a=parents[i][1]["curiosity"],a_min=self.container.min_curiosity,a_max=np.inf)
					self.container.update_bin(bin_index=parent_bin_index,genome_details=parents[i][1])
				self.logger.debug("parent_curiosity after "+str(parents[i][1]["curiosity"]))
			
			## Store metrics
			self.metrics["num_better_genome_found"].append(num_better_genome_found_in_this_iteration)
			self.metrics["total_quality_increase_by_better_genomes"].append(total_quality_increase_by_better_genomes_in_this_iteration)
			self.metrics["num_new_genomes"].append(self.container.num_genomes - old_num_genomes)
			self.metrics["container_metrics"].append(self.container.get_metrics())
			
			### Log metrics 
			self.log_metrics()
			self.logger.info("Number of better genomes found in this iteration "+str(num_better_genome_found_in_this_iteration))
			self.logger.info("Total quality increase by better genomes in this iteration "+str(total_quality_increase_by_better_genomes_in_this_iteration))
			if(num_better_genome_found_in_this_iteration):
				self.logger.info("Normalised quality increase by better genomes in this iteration "
					+str(total_quality_increase_by_better_genomes_in_this_iteration/num_better_genome_found_in_this_iteration))
			self.logger.info("Number of new genomes added in this iteration "+str(self.container.num_genomes - old_num_genomes)+"\n")
			
			## Save repertoire
			if(iteration%save_freq==0 and (save_dir is not None)):
				self.save_repertoire(save_file_path=save_dir+"ant_map_elites_repertoire_"+str(iteration)+".pkl")
				self.logger.info("Saving repertoire for iteration "+str(iteration)+"\n")
			self.current_iteration+=1

	def log_metrics(self):
		self.logger.info("Repertoire Metrics")
		for key,value in self.container.get_metrics().items():
			self.logger.info(key+" "+str(value))
		self.logger.info("")

	def print_metrics(self):
		print("\nRepertoire Metrics")
		for key,value in self.container.get_metrics().items():
			print(key+" "+str(value))
		print("")
	
	def save_repertoire(self,save_file_path):
		os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
		with open(save_file_path, 'wb') as f:
			pickle.dump({"container":self.container,"current_iteration":self.current_iteration,"metrics":self.metrics}, f)

	def load_repertoire(self,load_file_path):
		with open(load_file_path,'rb') as f:
			stored_dict = pickle.load(f)
			self.container = stored_dict["container"]
			self.current_iteration = stored_dict["current_iteration"]
			self.metrics = stored_dict["metrics"]

