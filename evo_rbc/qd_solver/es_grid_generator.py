from .repertoire_generator import Repertoire_Generator
from .container.es_grid import ESGrid
import numpy as np
import os
import pickle
import sys
from mpi4py import MPI
import copy
from scipy.stats import truncnorm

class ES_Grid_Generator(Repertoire_Generator):

	def __init__(self,env,qd_function,genome_constructor,num_dimensions,lower_limit,upper_limit,resolution,batch_size,seed=1):
		self.container = ESGrid(num_dimensions=num_dimensions,lower_limit=lower_limit,upper_limit=upper_limit,
								resolution=resolution,genome_constructor=genome_constructor)
		self.qd_function = qd_function
		super().__init__(env=env,genome_constructor=genome_constructor,batch_size=batch_size,seed=seed)


	def generate_repertoire(self,num_iterations,save_dir,save_freq,visualise,mutation_stdev=0.01,num_processes=1,
							num_samples_per_bin=1):
		"""
		populate grid with random parameters.
		"""
		for index in np.ndindex(self.container.num_bins):
			self.container.update_bin(bin_index=index,genome_details={"genome":self.genome_constructor(seed=self.seed)})

		for iteration in range(self.current_iteration,self.current_iteration+num_iterations):

			grid_sampled_genomes = []
			grid_probability_of_samples = []

			### generate samples from each bin. also add their negative i.e. antithetic sampling
			for index in np.ndindex(self.container.num_bins):
				grid_genome = self.container.grid[index]
				bin_sampled_genomes = []
				bin_probability_of_samples = []
				for i in range(num_samples_per_bin):
					sampled_genome = copy.deepcopy(grid_genome).mutate(mutation_stdev)
					epsilon = sampled_genome.parameters - grid_genome.parameters
					sampled_negative = self.genome_constructor(parameters=grid_genome.parameters - epsilon)
					bin_sampled_genomes.append((sampled_genome,sampled_negative))

					### store P( new params | old params) = truncnorm.pdf
					sampled_genome_parameters_probability = []
					for key, value in sampled_genome.parameters.items():
						low = sampled_genome.parameter_space.spaces[key + "_space"].low
						high = sampled_genome.parameter_space.spaces[key + "_space"].high
						mu = grid_genome.parameters[key]
						for i in range(low.shape[0]):
							sigma_temp = mutation_stdev * (high[i] - low[i])
							## truncnorm's clipped parameters are according to a standard normal so scale accordingly. refer their documentation for more details
							standard_low = (low[i] - mu[i]) / sigma_temp
							standard_high = (high[i] - mu[i]) / sigma_temp
							sampled_genome_parameters_probability.append(truncnorm.pdf(value,standard_low, standard_high,
																					   loc=mu[i], scale=sigma_temp))
					bin_probability_of_samples.append(sampled_genome_parameters_probability)
				grid_sampled_genomes.append(bin_sampled_genomes)
				grid_probability_of_samples.append(bin_probability_of_samples)
				
			### parallel evaluation

			self.current_iteration+=1

			## Save repertoire
			if(iteration%save_freq==0 and (save_dir is not None)):
				self.save_repertoire(save_file_path=save_dir+"es_grid_repertoire_"+str(iteration)+".pkl")
				self.logger.info("Saving repertoire for iteration "+str(iteration)+"\n")

	
	def save_repertoire(self,save_file_path):
		os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
		with open(save_file_path, 'wb') as save_file:
			pickle.dump({"container":self.container,"current_iteration":self.current_iteration}, save_file)

	def load_repertoire(self,load_file_path):
		with open(load_file_path,'rb') as load_file:
			stored_dict = pickle.load(load_file)
			self.container = stored_dict["container"]
			self.current_iteration = stored_dict["current_iteration"]

	def parallel_evaluate(self,genomes,visualise,num_processes):
		"""send the genome for evaluation to any worker that is free and return the resultant behavior,quality"""
		comm = MPI.COMM_SELF.Spawn(sys.executable, args=[self.env.mpi_worker_path], maxprocs=num_processes)
		genomes_len = len(genomes)

		genomes_matrix = [[] for _ in range(num_processes)]

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
