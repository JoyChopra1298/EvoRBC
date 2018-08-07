from gym.envs.mujoco.ant import AntEnv
from .eaenv import EAenv

class AntEAEnv(EAenv,AntEnv):

	def __init__(self):
		AntEnv.__init__()