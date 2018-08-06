from .repertoire_generator import Repertoire_Generator
from .container.grid import Grid
from .selector.uniform_random import Uniform_Random

class MAP_Elites(Repertoire_Generator):

	def __init__(env,mutation_rate):
		container = Grid(env)
		selector = Uniform_Random()
		super().__init__(env,container,selector,mutation_rate)

	def generate_repertoire(self,save_dir,num_generations,save_freq):
		pass

	def save_repertoire(self,save_path):
		pass

	def load_repertoire(self,load_path):
		pass