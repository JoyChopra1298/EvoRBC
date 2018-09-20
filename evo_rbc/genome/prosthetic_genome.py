from .genome import Genome
import gym.spaces as spaces
import numpy as np 

class ProstheticGenome(Genome):
	""" 
	Reference paper for muscle activaitons data - 
	http://nmbl.stanford.edu/publications/pdf/Hamner2012.pdf
	https://sci-hub.tw/10.1113/jphysiol.2003.057174
	https://www.streifeneder.com/downloads/o.p./400w43_e_poster_gangphasen_druck.pdf
	http://jeb.biologists.org/content/216/11/2150
	"""
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

	gait_time_period_low = [50.0]
	gait_time_period_high = [150.0]

	## exponential scale ---- 2^(amp damper)
	amplitude_multiplier_low = [-1.0]  ## = 0.5 times original amplitutde
	amplitude_multiplier_high = [0.5]  ## = 1.41x change
	
	initial_phase_low = [0.0]
	initial_phase_high = [0.99]

	soleus_amplitude_low = [0.7,0.0]
	soleus_amplitude_high = [0.8,0.05]
	
	gastroc_amplitude_low = [0.7,0.12]
	gastroc_amplitude_high = [0.8,0.17]

	tib_ant_amplitude_low = [0.5,0.45,0.6]
	tib_ant_amplitude_high = [0.6,0.5,0.65]

	vasti_amplitude_low = [0.8,0.15]
	vasti_amplitude_high = [0.95,0.2]

	rect_fem_amplitude_low = [0.5,0.65]
	rect_fem_amplitude_high = [0.6,0.8]

	hamstrings_amplitude_low = [0.8,0.55]
	hamstrings_amplitude_high = [0.9,0.6]
	
	bifemsh_amplitude_low = [0.85,0.55,0.2]
	bifemsh_amplitude_high = [0.9,0.6,0.3]

	glut_max_amplitude_low = [0.3,0.6,0.27]
	glut_max_amplitude_high = [0.4,0.75,0.33]
	
	iliopsoas_amplitude_low = [0.55,0.75,0.2]
	iliopsoas_amplitude_high = [0.65,0.9,0.25]

	general_amplitude_low = [0.3,0.3,0.05]
	general_amplitude_high = [1.0,0.6,0.25]
	
	# percentage is based on gait cycle length
	soleus_wave_start_percentage_low = [0.2,0.3]
	soleus_wave_start_percentage_high = [0.25,0.4]
	soleus_wave_width_percentage_low = [0.3,0.1]
	soleus_wave_width_percentage_high = [0.35,0.2]

	gastroc_wave_start_percentage_low = [0.07,0.05]
	gastroc_wave_start_percentage_high = [0.12,0.1]
	gastroc_wave_width_percentage_low = [0.4,0.13]
	gastroc_wave_width_percentage_high = [0.5,0.18]

	tib_ant_wave_start_percentage_low = [-0.05,0.47,0.55]
	tib_ant_wave_start_percentage_high = [0.0,0.53,0.6]
	tib_ant_wave_width_percentage_low = [0.35,0.25,0.4]
	tib_ant_wave_width_percentage_high = [0.4,0.3,0.5]

	vasti_l_wave_start_percentage_low = [0.2,0.7]
	vasti_l_wave_start_percentage_high = [0.22,0.75]
	vasti_r_wave_start_percentage_low = [0.7,0.2]
	vasti_r_wave_start_percentage_high = [0.72,0.25]
	vasti_wave_width_percentage_low = [0.25,0.15]
	vasti_wave_width_percentage_high = [0.3,0.2]
	
	rect_fem_l_wave_start_percentage_low = [0.15,0.62]
	rect_fem_l_wave_start_percentage_high = [0.2,0.68]
	rect_fem_r_wave_start_percentage_low = [0.65,0.12]
	rect_fem_r_wave_start_percentage_high = [0.7,0.18]
	rect_fem_wave_width_percentage_low = [0.25,0.3]
	rect_fem_wave_width_percentage_high = [0.3,0.35]
	
	hamstrings_l_wave_start_percentage_low = [-0.1,0.05]
	hamstrings_l_wave_start_percentage_high = [-0.05,0.1]
	hamstrings_r_wave_start_percentage_low = [0.4,0.55]
	hamstrings_r_wave_start_percentage_high = [0.45,0.6]
	hamstrings_wave_width_percentage_low = [0.35,0.25]
	hamstrings_wave_width_percentage_high = [0.4,0.3]
	
	bifemsh_l_wave_start_percentage_low = [-0.05,-0.02,0.7]
	bifemsh_l_wave_start_percentage_high = [0.0,0.02,0.75]
	bifemsh_r_wave_start_percentage_low = [0.45,0.48,0.2]
	bifemsh_r_wave_start_percentage_high = [0.5,0.52,0.25]
	bifemsh_wave_width_percentage_low = [0.4,0.25,0.15]
	bifemsh_wave_width_percentage_high = [0.45,0.3,0.2]

	glut_max_l_wave_start_percentage_low = [0.0,0.1,0.8]
	glut_max_l_wave_start_percentage_high = [0.05,0.15,0.85]
	glut_max_r_wave_start_percentage_low = [0.5,0.6,0.3]
	glut_max_r_wave_start_percentage_high = [0.55,0.65,0.35]
	glut_max_wave_width_percentage_low = [0.17,0.3,0.12]
	glut_max_wave_width_percentage_high = [0.22,0.4,0.17]

	#for adductor and abductor
	general_wave_start_percentage_low = [0.0,0.0,0.0]
	general_wave_start_percentage_high = [0.5,0.5,0.6]
	general_wave_width_percentage_low = [0.25,0.2,0.1]
	general_wave_width_percentage_high = [0.5,0.5,0.25]
	
	iliopsoas_l_wave_start_percentage_low = [0.4,0.45,0.25]
	iliopsoas_l_wave_start_percentage_high = [0.5,0.5,0.3]
	iliopsoas_r_wave_start_percentage_low = [0.9,0.95,0.75]
	iliopsoas_r_wave_start_percentage_high = [0.99,0.99,0.8]
	iliopsoas_wave_width_percentage_low = [0.2,0.3,0.05]
	iliopsoas_wave_width_percentage_high = [0.25,0.35,0.1]

	action_limits = [0,1]

	def __init__(self,parameters=None,seed=1):
		self.parameter_space = spaces.Dict({		
		"gait_time_period_space":spaces.Box(low=self._nparray(self.gait_time_period_low),high=self._nparray(self.gait_time_period_high),dtype=np.float32),
		"amplitude_multiplier_space":spaces.Box(low=self._nparray(self.amplitude_multiplier_low),high=self._nparray(self.amplitude_multiplier_high),dtype=np.float32),
		"initial_phase_space":spaces.Box(low=self._nparray(self.initial_phase_low),high=self._nparray(self.initial_phase_high),dtype=np.float32),
		"soleus_amplitude_space":spaces.Box(low=self._nparray(self.soleus_amplitude_low),high=self._nparray(self.soleus_amplitude_high),dtype=np.float32),
		"soleus_wave_start_percentage_space":spaces.Box(low=self._nparray(self.soleus_wave_start_percentage_low),
			high=self._nparray(self.soleus_wave_start_percentage_high),dtype=np.float32),
		"soleus_wave_width_percentage_space":spaces.Box(low=self._nparray(self.soleus_wave_width_percentage_low),
			high=self._nparray(self.soleus_wave_width_percentage_high),dtype=np.float32),
		"gastroc_amplitude_space":spaces.Box(low=self._nparray(self.gastroc_amplitude_low),high=self._nparray(self.gastroc_amplitude_high),dtype=np.float32),
		"gastroc_wave_start_percentage_space":spaces.Box(low=self._nparray(self.gastroc_wave_start_percentage_low),
			high=self._nparray(self.gastroc_wave_start_percentage_high),dtype=np.float32),
		"gastroc_wave_width_percentage_space":spaces.Box(low=self._nparray(self.gastroc_wave_width_percentage_low),
			high=self._nparray(self.gastroc_wave_width_percentage_high),dtype=np.float32),
		"tib_ant_amplitude_space":spaces.Box(low=self._nparray(self.tib_ant_amplitude_low),high=self._nparray(self.tib_ant_amplitude_high),dtype=np.float32),
		"tib_ant_wave_start_percentage_space":spaces.Box(low=self._nparray(self.tib_ant_wave_start_percentage_low),
			high=self._nparray(self.tib_ant_wave_start_percentage_high),dtype=np.float32),
		"tib_ant_wave_width_percentage_space":spaces.Box(low=self._nparray(self.tib_ant_wave_width_percentage_low),
			high=self._nparray(self.tib_ant_wave_width_percentage_high),dtype=np.float32),
		"vasti_l_amplitude_space":spaces.Box(low=self._nparray(self.vasti_amplitude_low),high=self._nparray(self.vasti_amplitude_high),dtype=np.float32),
		"vasti_l_wave_start_percentage_space":spaces.Box(low=self._nparray(self.vasti_l_wave_start_percentage_low),
			high=self._nparray(self.vasti_l_wave_start_percentage_high),dtype=np.float32),
		"vasti_l_wave_width_percentage_space":spaces.Box(low=self._nparray(self.vasti_wave_width_percentage_low),
			high=self._nparray(self.vasti_wave_width_percentage_high),dtype=np.float32),
		"vasti_r_amplitude_space":spaces.Box(low=self._nparray(self.vasti_amplitude_low),high=self._nparray(self.vasti_amplitude_high),dtype=np.float32),
		"vasti_r_wave_start_percentage_space":spaces.Box(low=self._nparray(self.vasti_r_wave_start_percentage_low),
			high=self._nparray(self.vasti_r_wave_start_percentage_high),dtype=np.float32),
		"vasti_r_wave_width_percentage_space":spaces.Box(low=self._nparray(self.vasti_wave_width_percentage_low),
			high=self._nparray(self.vasti_wave_width_percentage_high),dtype=np.float32),
		"rect_fem_l_amplitude_space":spaces.Box(low=self._nparray(self.rect_fem_amplitude_low),high=self._nparray(self.rect_fem_amplitude_high),dtype=np.float32),
		"rect_fem_l_wave_start_percentage_space":spaces.Box(low=self._nparray(self.rect_fem_l_wave_start_percentage_low),
			high=self._nparray(self.rect_fem_l_wave_start_percentage_high),dtype=np.float32),
		"rect_fem_l_wave_width_percentage_space":spaces.Box(low=self._nparray(self.rect_fem_wave_width_percentage_low),
			high=self._nparray(self.rect_fem_wave_width_percentage_high),dtype=np.float32),
		"rect_fem_r_amplitude_space":spaces.Box(low=self._nparray(self.rect_fem_amplitude_low),high=self._nparray(self.rect_fem_amplitude_high),dtype=np.float32),
		"rect_fem_r_wave_start_percentage_space":spaces.Box(low=self._nparray(self.rect_fem_r_wave_start_percentage_low),
			high=self._nparray(self.rect_fem_r_wave_start_percentage_high),dtype=np.float32),
		"rect_fem_r_wave_width_percentage_space":spaces.Box(low=self._nparray(self.rect_fem_wave_width_percentage_low),
			high=self._nparray(self.rect_fem_wave_width_percentage_high),dtype=np.float32),
		"hamstrings_l_amplitude_space":spaces.Box(low=self._nparray(self.hamstrings_amplitude_low),high=self._nparray(self.hamstrings_amplitude_high),dtype=np.float32),
		"hamstrings_l_wave_start_percentage_space":spaces.Box(low=self._nparray(self.hamstrings_l_wave_start_percentage_low),
			high=self._nparray(self.hamstrings_l_wave_start_percentage_high),dtype=np.float32),
		"hamstrings_l_wave_width_percentage_space":spaces.Box(low=self._nparray(self.hamstrings_wave_width_percentage_low),
			high=self._nparray(self.hamstrings_wave_width_percentage_high),dtype=np.float32),
		"hamstrings_r_amplitude_space":spaces.Box(low=self._nparray(self.hamstrings_amplitude_low),high=self._nparray(self.hamstrings_amplitude_high),dtype=np.float32),
		"hamstrings_r_wave_start_percentage_space":spaces.Box(low=self._nparray(self.hamstrings_r_wave_start_percentage_low),
			high=self._nparray(self.hamstrings_r_wave_start_percentage_high),dtype=np.float32),
		"hamstrings_r_wave_width_percentage_space":spaces.Box(low=self._nparray(self.hamstrings_wave_width_percentage_low),
			high=self._nparray(self.hamstrings_wave_width_percentage_high),dtype=np.float32),
		"bifemsh_l_amplitude_space":spaces.Box(low=self._nparray(self.bifemsh_amplitude_low),high=self._nparray(self.bifemsh_amplitude_high),dtype=np.float32),
		"bifemsh_l_wave_start_percentage_space":spaces.Box(low=self._nparray(self.bifemsh_l_wave_start_percentage_low),
			high=self._nparray(self.bifemsh_l_wave_start_percentage_high),dtype=np.float32),
		"bifemsh_l_wave_width_percentage_space":spaces.Box(low=self._nparray(self.bifemsh_wave_width_percentage_low),
			high=self._nparray(self.bifemsh_wave_width_percentage_high),dtype=np.float32),
		"bifemsh_r_amplitude_space":spaces.Box(low=self._nparray(self.bifemsh_amplitude_low),high=self._nparray(self.bifemsh_amplitude_high),dtype=np.float32),
		"bifemsh_r_wave_start_percentage_space":spaces.Box(low=self._nparray(self.bifemsh_r_wave_start_percentage_low),
			high=self._nparray(self.bifemsh_r_wave_start_percentage_high),dtype=np.float32),
		"bifemsh_r_wave_width_percentage_space":spaces.Box(low=self._nparray(self.bifemsh_wave_width_percentage_low),
			high=self._nparray(self.bifemsh_wave_width_percentage_high),dtype=np.float32),
		"glut_max_l_amplitude_space":spaces.Box(low=self._nparray(self.glut_max_amplitude_low),high=self._nparray(self.glut_max_amplitude_high),dtype=np.float32),
		"glut_max_l_wave_start_percentage_space":spaces.Box(low=self._nparray(self.glut_max_l_wave_start_percentage_low),
			high=self._nparray(self.glut_max_l_wave_start_percentage_high),dtype=np.float32),
		"glut_max_l_wave_width_percentage_space":spaces.Box(low=self._nparray(self.glut_max_wave_width_percentage_low),
			high=self._nparray(self.glut_max_wave_width_percentage_high),dtype=np.float32),
		"glut_max_r_amplitude_space":spaces.Box(low=self._nparray(self.glut_max_amplitude_low),high=self._nparray(self.glut_max_amplitude_high),dtype=np.float32),
		"glut_max_r_wave_start_percentage_space":spaces.Box(low=self._nparray(self.glut_max_r_wave_start_percentage_low),
			high=self._nparray(self.glut_max_r_wave_start_percentage_high),dtype=np.float32),
		"glut_max_r_wave_width_percentage_space":spaces.Box(low=self._nparray(self.glut_max_wave_width_percentage_low),
			high=self._nparray(self.glut_max_wave_width_percentage_high),dtype=np.float32),
		"abd_r_amplitude_space":spaces.Box(low=self._nparray(self.general_amplitude_low),high=self._nparray(self.general_amplitude_high),dtype=np.float32),
		"abd_r_wave_start_percentage_space":spaces.Box(low=self._nparray(self.general_wave_start_percentage_low),
			high=self._nparray(self.general_wave_start_percentage_high),dtype=np.float32),
		"abd_r_wave_width_percentage_space":spaces.Box(low=self._nparray(self.general_wave_width_percentage_low),
			high=self._nparray(self.general_wave_width_percentage_high),dtype=np.float32),
		"abd_l_amplitude_space":spaces.Box(low=self._nparray(self.general_amplitude_low),high=self._nparray(self.general_amplitude_high),dtype=np.float32),
		"abd_l_wave_start_percentage_space":spaces.Box(low=self._nparray(self.general_wave_start_percentage_low),
			high=self._nparray(self.general_wave_start_percentage_high),dtype=np.float32),
		"abd_l_wave_width_percentage_space":spaces.Box(low=self._nparray(self.general_wave_width_percentage_low),
			high=self._nparray(self.general_wave_width_percentage_high),dtype=np.float32),
		"add_r_amplitude_space":spaces.Box(low=self._nparray(self.general_amplitude_low),high=self._nparray(self.general_amplitude_high),dtype=np.float32),
		"add_r_wave_start_percentage_space":spaces.Box(low=self._nparray(self.general_wave_start_percentage_low),
			high=self._nparray(self.general_wave_start_percentage_high),dtype=np.float32),
		"add_r_wave_width_percentage_space":spaces.Box(low=self._nparray(self.general_wave_width_percentage_low),
			high=self._nparray(self.general_wave_width_percentage_high),dtype=np.float32),
		"add_l_amplitude_space":spaces.Box(low=self._nparray(self.general_amplitude_low),high=self._nparray(self.general_amplitude_high),dtype=np.float32),
		"add_l_wave_start_percentage_space":spaces.Box(low=self._nparray(self.general_wave_start_percentage_low),
			high=self._nparray(self.general_wave_start_percentage_high),dtype=np.float32),
		"add_l_wave_width_percentage_space":spaces.Box(low=self._nparray(self.general_wave_width_percentage_low),
			high=self._nparray(self.general_wave_width_percentage_high),dtype=np.float32),
		"iliopsoas_r_amplitude_space":spaces.Box(low=self._nparray(self.iliopsoas_amplitude_low),high=self._nparray(self.iliopsoas_amplitude_high),dtype=np.float32),
		"iliopsoas_r_wave_start_percentage_space":spaces.Box(low=self._nparray(self.iliopsoas_r_wave_start_percentage_low),
			high=self._nparray(self.iliopsoas_r_wave_start_percentage_high),dtype=np.float32),
		"iliopsoas_r_wave_width_percentage_space":spaces.Box(low=self._nparray(self.iliopsoas_wave_width_percentage_low),
			high=self._nparray(self.iliopsoas_wave_width_percentage_high),dtype=np.float32),
		"iliopsoas_l_amplitude_space":spaces.Box(low=self._nparray(self.iliopsoas_amplitude_low),high=self._nparray(self.iliopsoas_amplitude_high),dtype=np.float32),
		"iliopsoas_l_wave_start_percentage_space":spaces.Box(low=self._nparray(self.iliopsoas_l_wave_start_percentage_low),
			high=self._nparray(self.iliopsoas_l_wave_start_percentage_high),dtype=np.float32),
		"iliopsoas_l_wave_width_percentage_space":spaces.Box(low=self._nparray(self.iliopsoas_wave_width_percentage_low),
			high=self._nparray(self.iliopsoas_wave_width_percentage_high),dtype=np.float32),

		})
		super().__init__(parameters=parameters,seed=seed)

	def control_function(self,muscle_index,time_step):
		"""Calculate control actions for a particular muscle at time step time_step"""
		
		muscle_name = self.muscle_dict[muscle_index]
		if(muscle_name=="soleus_l" or muscle_name=="gastroc_l" or muscle_name=="tib_ant_l"):
			return self._calculate_wave(muscle_name[:-2],time_step)
		else:
			return self._calculate_wave(muscle_name,time_step) 
			
	def _truncated_function(self,function,lower_limit,upper_limit,timestep,**kwargs):
		if(timestep < lower_limit or timestep > upper_limit):
			return 0
		else:
			return function(**kwargs)

	def gaussian(self,x,mean,sigma):
		exponent = np.square(x-mean)/2/sigma/sigma
		return  np.exp(-1*exponent)

	def _calculate_wave(self,muscle_name,time_step):
		gait_time_period = self.parameters["gait_time_period"]
		cycle_count = int(time_step/gait_time_period)
		initial_phase_steps = self.parameters["initial_phase"][0]

		wave = 0.0
		for i in range(len(self.parameters[muscle_name+"_amplitude"])):
			current_cycle_count = cycle_count
			if(self.parameters[muscle_name+"_wave_start_percentage"][i]+self.parameters[muscle_name+"_wave_width_percentage"][i] > 1.0):
				current_cycle_count = cycle_count - 1
			start_time_step_i = int((current_cycle_count + self.parameters[muscle_name+"_wave_start_percentage"][i] )*gait_time_period) + initial_phase_steps
			end_time_step_i = start_time_step_i + int(self.parameters[muscle_name+"_wave_width_percentage"][i] *gait_time_period)
			# wave_i = self.parameters[muscle_name+"_amplitude"][i]*self._truncated_function(np.sin,start_time_step_i,end_time_step_i,
			# 	np.pi*(time_step-start_time_step_i)/(end_time_step_i-start_time_step_i),time_step)
			wave_i = self.parameters[muscle_name+"_amplitude"][i]*self._truncated_function(self.gaussian,start_time_step_i,end_time_step_i,
				time_step,x=time_step,mean=(start_time_step_i+end_time_step_i)/2,sigma=(end_time_step_i-start_time_step_i)/4)

			wave += wave_i
		return wave*pow(2,self.parameters["amplitude_multiplier"]) + 0.01 * np.random.normal(0,1)