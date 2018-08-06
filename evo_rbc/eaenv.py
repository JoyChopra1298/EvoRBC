import gym

class EAenv(gym.Env):

	def quality_diversity_fitness_function():
		raise NotImplementedError
		
	def task_fitness_function():
		raise NotImplementedError

	def evaluate_task_fitness():
		"""Evaluate fitness for the specific task"""
		raise NotImplementedError

	def evaluate_quality_diversity_fitness(Genome genome):
		"""Evaluate fitness for quality-diversity(QD)"""
		raise NotImplementedError
