from .genome import Genome
import gym.spaces as spaces
import numpy as np 


class ProstheticGenome(Genome):
	## Reference paper for muscle activaitons data - http://nmbl.stanford.edu/publications/pdf/Hamner2012.pdf
	num_muscles = 19
	muscle_dict = {0 :"abd_r",
					1 :"add_r",
					2 :"hamstrings_r",
					3 :"bifemsh_r",
					4 :"glut_max_r",
					5 :"iliopsoas_r",
					6 :"rect_fem_r",
					7 :"vasti_r",
					8 :"abd_l",
					9 :"add_l",
					10 :"hamstrings_l",
					11 :"bifemsh_l",
					12 :"glut_max_l",
					13 :"iliopsoas_l",
					14 :"rect_fem_l",
					15 :"vasti_l",
					16 :"gastroc_l",
					17 :"soleus_l",
					18 :"tib_ant_l"}
	# parameter_space = spaces.Dict({
	# 	"amplitude_space":spaces.Box(low=amplitude_low,high=amplitude_high,shape=(num_joints,1),dtype=np.float32),
	# 	"phase_space":spaces.Box(low=phase_low,high=phase_high,shape=(num_joints,1),dtype=np.float32),
	# 	"smoothness_space":spaces.Box( low=np.array([smoothness_low]), high=np.array([smoothness_high]),dtype=np.float32),
	# 	"epsilon_space":spaces.Box(low=epsilon_low,high=epsilon_high,shape=(num_joints,1),dtype=np.float32),
	# 	"control_frequency_space":spaces.Box( low=np.array([control_frequency_low]), high=np.array([control_frequency_high]),dtype=np.float32),})

	def __init__(self,parameters=None,seed=1):
		super().__init__(parameters=parameters,seed=seed)

	def control_function(self,muscle_index,time_step):
		"""Calculate control actions for a particular muscle at time step time_step"""
		muscle_name = self.muscle_dict[muscle_index]
		if(muscle_name == ""):
			
