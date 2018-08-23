from .genome import Genome
import gym.spaces as spaces
import numpy as np 

class AntGenome(Genome):
	#for each joint(2 for each leg, total 4*2 = 8). control function specific
	## values taken according to EvoRBC paper, see readme for details.some values adjusted
	amplitude_low = -0.5
	amplitude_high = 0.5
	phase_low = 0
	phase_high = 1
	smoothness_low = (np.pi)/4
	smoothness_high = (np.pi/4) + 2*(np.pi)
	epsilon_low = -0.5
	epsilon_high = 0.5
	# on exponential scale, example -1 would mean 1e-1
	control_frequency_low = -2.5 
	control_frequency_high = -1.5

	action_limits = [-1,1]
	num_joints = 8
	parameter_space = spaces.Dict({
		"amplitude_space":spaces.Box(low=amplitude_low,high=amplitude_high,shape=(num_joints,1),dtype=np.float32),
		"phase_space":spaces.Box(low=phase_low,high=phase_high,shape=(num_joints,1),dtype=np.float32),
		"smoothness_space":spaces.Box( low=np.array([smoothness_low]), high=np.array([smoothness_high]),dtype=np.float32),
		"epsilon_space":spaces.Box(low=epsilon_low,high=epsilon_high,shape=(num_joints,1),dtype=np.float32),
		"control_frequency_space":spaces.Box( low=np.array([control_frequency_low]), high=np.array([control_frequency_high]),dtype=np.float32),})

	def __init__(self,parameters=None,seed=1):
		super().__init__(parameters=parameters,seed=seed)
	
	def control_function(self,joint_index,time_step):
		"""Returns the action for corresponding joint from genome parameters"""
		angle = time_step*np.power(10,self.parameters["control_frequency"]) + self.parameters["phase"][joint_index]
		sine = np.sin(2*(np.pi)*(angle))
		tanh = (np.tanh(self.parameters["smoothness"])* sine) 
		return self.parameters["epsilon"][joint_index] + self.parameters["amplitude"][joint_index]* tanh
