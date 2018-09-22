from .repertoire_generator import Repertoire_Generator
from .container.grid import Grid
import copy,os
import numpy as np
import pickle
from mpi4py import MPI
import sys
import matplotlib.pyplot as plt

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
		
	def generate_repertoire(self,num_iterations,save_dir,save_freq,visualise,mutation_stdev=0.01,num_processes=1):
		""" generate a random population initially, generating double the batch_size to increase the probability that
		 that at least batch_size elements get added"""
		if(self.current_iteration==1):
			self.logger.info("Initialising repertoire with random population")
			random_genomes = []
			for i in range(2*self.batch_size):
				random_genomes.append(self.genome_constructor(seed=self.seed))

			qd_evaluations = self.parallel_evaluate(genomes=random_genomes,num_processes=num_processes,visualise=visualise)

			for i in range(len(random_genomes)):
				behavior,quality = qd_evaluations[i%num_processes][int(i/num_processes)]
				if(self.container.is_high_quality(behavior=behavior,quality=quality)):
					self.container.add_genome(genome=random_genomes[i],behavior=behavior,quality=quality)
			self.log_metrics()
			self.metrics["num_better_genome_found"].append(0)
			self.metrics["total_quality_increase_by_better_genomes"].append(0)
			self.metrics["num_new_genomes"].append(self.container.num_genomes)

		## to do vary stdev while training based on metrics/ include crossover
		for iteration in range(self.current_iteration,self.current_iteration+num_iterations):
			self.logger.info("Iteration "+str(iteration))
			parents = self.selector.select(self.container.grid,self.batch_size)
			
			parents_len = len(parents)
			"""len(parents) used instead of batch size since it is possible to not have a complete batch from the container"""
			children_genomes = []

			child_multiplier = 3
			for i in range(parents_len):
				for j in range(child_multiplier):
					parent_genome = parents[i][1]["genome"]
					child_genome = copy.deepcopy(parent_genome)
					child_genome.mutate(sigma=mutation_stdev*(j+1))
					children_genomes.append(child_genome)

			children_genomes_len = len(children_genomes)
			# add random genomes for rest of the batch size .
			# add twice the batch size if less than 25% batch was filled
			if(children_genomes_len < self.batch_size * 0.25):
				extra_random_genomes_len = (2 * self.batch_size) - children_genomes_len
			else:
				extra_random_genomes_len = self.batch_size - children_genomes_len

			for i in range(extra_random_genomes_len):
				children_genomes.append(self.genome_constructor(seed=self.seed))

			qd_evaluations = self.parallel_evaluate(genomes=children_genomes,num_processes=num_processes,visualise=visualise)

			## initialise metrics for current iteration
			num_better_genome_found_in_this_iteration = 0
			total_quality_increase_by_better_genomes_in_this_iteration = 0
			old_num_genomes = self.container.num_genomes

			for i in range(children_genomes_len):
				child_genome = children_genomes[i]
				parent_index = int(i/child_multiplier)
				behavior,quality = qd_evaluations[i%num_processes][int(i/num_processes)]
				self.logger.debug("parent_curiosity before "+str(parents[parent_index][1]["curiosity"]))
				
				bin_index = self.container.get_bin(behavior)				
				self.logger.debug("Child bin index "+str(bin_index))
				parent_bin_index = parents[parent_index][0]
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

					parents[parent_index][1]["curiosity"] *= self.container.curiosity_multiplier
					self.container.update_curiosity(bin_index=parent_bin_index,curiosity=parents[parent_index][1]["curiosity"])
					self.container.add_genome(genome=child_genome,behavior=behavior,quality=quality)

				else:
					parents[parent_index][1]["curiosity"] /= self.container.curiosity_multiplier
					parents[parent_index][1]["curiosity"] = np.clip(a=parents[parent_index][1]["curiosity"],a_min=self.container.min_curiosity,a_max=np.inf)
					self.container.update_curiosity(bin_index=parent_bin_index,curiosity=parents[parent_index][1]["curiosity"])
				self.logger.debug("parent_curiosity after "+str(parents[parent_index][1]["curiosity"]))

			for i in range(children_genomes_len,children_genomes_len+extra_random_genomes_len):
				behavior, quality = qd_evaluations[i % num_processes][int(i / num_processes)]
				if (self.container.is_high_quality(behavior=behavior, quality=quality)):
					bin_index = self.container.get_bin(behavior)
					if (bin_index in self.container.grid):
						old_genome_quality = self.container.grid[bin_index]["quality"]
						self.logger.debug("Old quality in same bin " + str(old_genome_quality))
						num_better_genome_found_in_this_iteration += 1
						total_quality_increase_by_better_genomes_in_this_iteration += quality - old_genome_quality
					self.container.add_genome(genome=children_genomes[i], behavior=behavior, quality=quality)

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

			self.current_iteration+=1

			## Save repertoire
			if(iteration%save_freq==0 and (save_dir is not None)):
				self.save_repertoire(save_file_path=save_dir+"map_elites_repertoire_"+str(iteration)+".pkl")
				self.logger.info("Saving repertoire for iteration "+str(iteration)+"\n")

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

	def parallel_evaluate(self,genomes,visualise,num_processes):
		"""send the genome for evaluation to any worker that is free and return the resultant behavior,quality"""
		comm = MPI.COMM_SELF.Spawn(sys.executable,
									   args=[self.env.mpi_worker_path],
									   maxprocs=num_processes)
		genomes_len = len(genomes)

		genomes_matrix = [[] for i in range(num_processes)]

		for i in range(genomes_len):
			genomes_matrix[i%num_processes].append(genomes[i])

		max_time_steps_qd = comm.bcast(self.env.max_time_steps_qd,root=MPI.ROOT)
		genome = comm.scatter(genomes_matrix,root=MPI.ROOT)
		visualise = comm.bcast(visualise,root=MPI.ROOT)
		joint_error_margin = comm.bcast(self.env.joint_error_margin,root=MPI.ROOT)

		qd_evaluations = None
		qd_evaluations = comm.gather(qd_evaluations,root=MPI.ROOT)

		comm.Disconnect()
		
		return qd_evaluations

	def view_metrics(self,metric_key,secondary_metric_key=None):
		"""Show the metrics as plot"""
		if(secondary_metric_key):
			metric_list = [self.metrics[metric_key][i][secondary_metric_key] for i in range(len(self.metrics[metric_key]))]
			ylabel = metric_key+" "+str(secondary_metric_key) 
		else:
			metric_list = self.metrics[metric_key]
			ylabel = metric_key
		plt.plot([i+1 for i in range(len(metric_list))],metric_list)
		plt.xlabel('Iteration number')
		plt.ylabel(ylabel)
		plt.show()
