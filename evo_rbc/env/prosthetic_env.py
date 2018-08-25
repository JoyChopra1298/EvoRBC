from osim.env import ProstheticsEnv
from .eaenv import EAenv
import statistics as stats

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
		
		pelvis_kinematics = {"vx":[],"vy":[],"vz":[]}
		self.logger.debug("Starting an evaluation for steady skeleton runner")
		for time_step in range(self.max_time_steps_qd):
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
				# pelvis_kinematics["vy"].append(pelvis_velocity_vector[1])
				# pelvis_kinematics["vz"].append(pelvis_position_vector[2])

		mean_velocity_x = stats.mean(pelvis_kinematics["vx"])
		behavior = mean_velocity_x

		performance = len(pelvis_kinematics["vx"])*((mean_velocity_x**2) - stats.stdev(pelvis_kinematics["vx"])**2)
		self.logger.debug("Evaluation finished with\nbehavior "+str(behavior)+"\nperformance "+str(performance))
		return (behavior,performance)		
