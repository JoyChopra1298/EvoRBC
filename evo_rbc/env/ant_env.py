from gym.envs.mujoco.ant import AntEnv
from .eaenv import EAenv
import statistics as stats

class AntEAEnv(EAenv,AntEnv):

	def __init__(self,seed=1,max_time_steps_qd=1000,max_time_steps_task=2000):
		AntEnv.__init__(self)
		self.seed(seed)
		EAenv.__init__(self,max_time_steps_qd=max_time_steps_qd,max_time_steps_task=max_time_steps_task)

	def evaluate_task_fitness(self,task_funtion,arbitrator_genome,visualise=False):
		raise NotImplementedError

	def evaluate_quality_diversity_fitness(self,qd_function,primitive_genome,visualise=False):
		return qd_function(primitive_genome,visualise)

	def qd_steady_runner(self,primitive_genome,visualise=False):
		"""quality_diversity fitness function for a runner with steady velocity vector"""
		done = False
		self.reset()
		""" arrays for vx,vy and vz for main body(torso) of ant. It's mean is used to calculate behavior(diversity)
		 and variance is used to calulate performance(quality). 
		 vz is forced to be close to 0 by accumulating it's absolute value and penalising that in fitness"""
		body_velocity = {"vx":[],"vy":[],"vz":[]}
		for time_step in range(self.max_time_steps_qd):
			if(not done):
				action = []
				for joint_index in range(self.action_space.shape[0]):
					action.append(primitive_genome.control_function(joint_index=joint_index,time_step=time_step))
				observation, reward, done, info = self.step(action) 
				if(visualise):
					self.render()
				velocity_vector = self.data.get_body_xvelp("torso")
				body_velocity["vx"].append(velocity_vector[0])
				body_velocity["vy"].append(velocity_vector[1])
				body_velocity["vz"].append(abs(velocity_vector[2]))

		## since dividing behavior into bins depends on container so return a tuple (mean_vx,mean_vy) for behavior from env
		behavior = {"vx_mean":stats.mean(body_velocity["vx"]),"vy_mean":stats.mean(body_velocity["vy"])}
		performance = - (stats.stdev(body_velocity["vx"]) + stats.stdev(body_velocity["vy"]) + stats.mean(body_velocity["vz"]))
		return (behavior,performance)
