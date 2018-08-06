import gym

class EAenv(gym.Env):

	def task_fitness_function():
		raise NotImplementedError

	def evaluate_task_fitness():
		"""Evaluate fitness for the specific task"""
		raise NotImplementedError

