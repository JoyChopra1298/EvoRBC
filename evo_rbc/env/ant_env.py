from gym.envs.mujoco.ant import AntEnv
from .eaenv import EAenv
import statistics as stats
import numpy as np

class AntEAEnv(EAenv,AntEnv):

	def __init__(self,seed=1,max_time_steps_qd=1000,max_time_steps_task=2000,only_return_general_reward=False):
		AntEnv.__init__(self)
		self.seed(seed)
		EAenv.__init__(self,max_time_steps_qd=max_time_steps_qd,max_time_steps_task=max_time_steps_task,only_return_general_reward=only_return_general_reward)
		self.mpi_worker_path = '../../mpi_worker/ant_evaluation_worker.py'
		# self.logger.debug("Created ant environment")

	def evaluate_task_fitness(self,task_funtion,arbitrator_genome,visualise=False):
		raise NotImplementedError

	def evaluate_quality_diversity_fitness(self,qd_function,primitive_genome,visualise=False):
		return qd_function(primitive_genome,visualise)

	def qd_steady_runner(self,primitive_genome,visualise=False):
		"""quality_diversity fitness function for a runner with steady velocity vector"""
		done = False
		self.reset()
		""" arrays for vx,vy and rz for main body(torso) of ant. It's mean is used to calculate behavior(diversity)
		 and variance is used to calulate performance(quality). 
		 rz is forced to not change by penalising it's variation(stddev)"""
		torso_kinematics = {"vx":[],"vy":[],"rz":[]}
		self.logger.debug("Starting an evaluation for steady runner ant")
		for time_step in range(self.max_time_steps_qd):
			if(not done):
				action = []
				for joint_index in range(self.action_space.shape[0]):
					action.append(primitive_genome.control_function(joint_index=joint_index,time_step=time_step))
				observation, reward, done, info = self.step(action) 
				if(visualise):
					self.render()
				position_vector = self.data.get_body_xpos("torso")
				velocity_vector = self.data.get_body_xvelp("torso")
				torso_kinematics["vx"].append(velocity_vector[0])
				torso_kinematics["vy"].append(velocity_vector[1])
				torso_kinematics["rz"].append(position_vector[2])
		""" since dividing behavior into bins depends on container so just return a tuple (mean_vx,mean_vy) for behavior from env
		also they are not attributes of genome like genome.behavior, genome.performance since can have various fitness func 
		although then could use a dictionary"""
		behavior = (stats.mean(torso_kinematics["vx"]),stats.mean(torso_kinematics["vy"]))
		performance = 0.0
		if( not self.only_return_general_reward):
			"""add behavior specific information to the reward"""
			performance -= (stats.stdev(torso_kinematics["vx"]) + stats.stdev(torso_kinematics["vy"]) + stats.stdev(torso_kinematics["rz"]))
			self.logger.debug("Evaluation finished with\nbehavior "+str(behavior)+"\nperformance "+str(performance))
		return (behavior,performance)

	def reset_model(self):
		qpos = self.init_qpos
		qvel = self.init_qvel
		self.set_state(qpos, qvel)
		return self._get_obs()

	"""override since want to use torso positions too"""

	def _get_obs(self):
		return np.concatenate([
			self.sim.data.qpos.flat,
			self.sim.data.qvel.flat,
			np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
		])