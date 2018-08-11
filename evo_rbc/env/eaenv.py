import gym

class EAenv(gym.Env):

	def __init__(self,logger,max_time_steps_qd=1000,max_time_steps_task=2000):
		self.max_time_steps_qd = max_time_steps_qd
		self.max_time_steps_task = max_time_steps_task
		self.logger = logger

	def evaluate_task_fitness(self,task_funtion,arbitrator_genome,visualise=False):
		"""Evaluate the genome's performance on given task
		Can choose various task functions according to parameter passed"""
		raise NotImplementedError

	def evaluate_quality_diversity_fitness(self,qd_function,primitive_genome,visualise=False):
		"""Evaluate the genome on the environment and return a 2-tuple (performance(quality),behavior(diversity))
		Can choose different fitness functions"""
		raise NotImplementedError

