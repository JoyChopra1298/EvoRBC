import gym
import evo_rbc.main.utils as utils

class EAenv(gym.Env):

	def __init__(self,max_time_steps_qd=1000,max_time_steps_task=2000,only_return_general_reward=False):
		self.max_time_steps_qd = max_time_steps_qd
		self.max_time_steps_task = max_time_steps_task

		"""describes whether to add behavior specific information to the reward"""
		self.only_return_general_reward = only_return_general_reward
		self.logger = utils.getLogger()

	def evaluate_task_fitness(self,task_funtion,arbitrator_genome,visualise=False):
		"""Evaluate the genome's performance on given task
		Can choose various task functions according to parameter passed"""
		raise NotImplementedError

	def evaluate_quality_diversity_fitness(self,qd_function,primitive_genome,visualise=False):
		"""Evaluate the genome on the environment and return a 2-tuple (performance(quality),behavior(diversity))
		Can choose different fitness functions"""
		raise NotImplementedError

	def __getstate__(self):
		excluded_subnames = ["logger"]
		state = {}
		for k, v in self.__dict__.items():
			if(k=="logger"):
				continue
			state[k] = v
		return state

	def __setstate__(self, state):
		for k,v in state.items():
			self.__dict__[k] = v
		self.logger = utils.getLogger()
		