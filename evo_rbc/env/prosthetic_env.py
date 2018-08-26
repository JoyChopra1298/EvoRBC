from osim.env import ProstheticsEnv
from .eaenv import EAenv
import statistics as stats
import numpy as np
import time

class ProstheticEAEnv(EAenv,ProstheticsEnv):

	def __init__(self,seed=1,max_time_steps_qd=300,max_time_steps_task=2000):
		ProstheticsEnv.__init__(self,visualize=False)
		self.seed(seed)
		EAenv.__init__(self,max_time_steps_qd=max_time_steps_qd,max_time_steps_task=max_time_steps_task)
		self.mpi_worker_path = '../../mpi_worker/prosthetic_evaluation_worker.py'
		# self.logger.debug("Created prosthetic environment")

	def evaluate_task_fitness(self,task_funtion,arbitrator_genome,visualise=False):
		raise NotImplementedError

	def evaluate_quality_diversity_fitness(self,qd_function,primitive_genome,visualise=False):
		return qd_function(primitive_genome,visualise)

	def qd_steady_runner(self,primitive_genome,visualise=False):
		"""quality_diversity fitness function for a runner with steady velocity vector"""
		done = False
		self.reset()
		
		pelvis_kinematics = {"vx":[],"vy":[],"vz":[],"rz":[]}
		
		initial_state_desc = self.get_state_desc()
		initial_position_x = initial_state_desc["body_pos"]["pelvis"][0]

		performance = 0.0

		start_time = time.time()
		self.logger.debug("Starting an evaluation for steady skeleton runner")

		for time_step in range(self.max_time_steps_qd):

			# stop if evaluation taking too much time (more than 5 minutes)
			current_time = time.time()
			if( (current_time - start_time) > 300):
				mean_velocity_x = stats.mean(pelvis_kinematics["vx"])
				behavior = mean_velocity_x
				self.logger.debug("Evaluation stopped since too much time elapsed .Behavior "+str(behavior))
				return (-1000.0,-1000.0)
			
			if(not done):
				action = []
				for muscle_index in range(self.action_space.shape[0]):
					action.append(primitive_genome.control_function(muscle_index=muscle_index,time_step=time_step))
				observation, reward, done, info = self.step(action) 
				if(visualise):
					self.render()
				state_desc = self.get_state_desc()
				pelvis_velocity_vector = state_desc["body_vel"]["pelvis"] 
				pelvis_kinematics["vx"].append(pelvis_velocity_vector[0])
				if(time_step==100):
					pelvis_position_x = state_desc["body_pos"]["pelvis"][0]
					if(pelvis_position_x < initial_position_x):
						### stop computation since going backwards
						mean_velocity_x = stats.mean(pelvis_kinematics["vx"])
						behavior = mean_velocity_x
						self.logger.debug("Evaluation stopped since going backwards behavior "+str(behavior))
						return (-1000.0,-1000.0)
				
				pelvis_position_vector_y = state_desc["body_pos"]["pelvis"][1]
				if(pelvis_position_vector_y < 0.75):
					performance -= 1

				performance += 2*state_desc["body_pos"]["head"][0]-state_desc["body_pos"]["pelvis"][0]

		mean_velocity_x = stats.mean(pelvis_kinematics["vx"])
		behavior = mean_velocity_x

		##penalise negative velocity
		if(behavior < 0):
			performance -= 50.0

		performance += len(pelvis_kinematics["vx"])*((mean_velocity_x**2) - stats.stdev(pelvis_kinematics["vx"])**2) 

		self.logger.debug("Evaluation finished with\nbehavior "+str(behavior)+"\nperformance "+str(performance)+"\n survived for timesteps "
			+str(len(pelvis_kinematics["vx"])))
		
		return (behavior,performance)		
