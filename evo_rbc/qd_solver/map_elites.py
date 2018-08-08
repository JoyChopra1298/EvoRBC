from .repertoire_generator import Repertoire_Generator
from .container.grid import Grid
from .selector.uniform_random import Uniform_Random

class MAP_Elites(Repertoire_Generator):

	def __init__(self,env,genome,mutation_rate=0.1,population_size=100):
		self.container = Grid(env)
		self.selector = Uniform_Random()
		super().__init__(env,genome,self.container,self.selector,mutation_rate,population_size)
		self.current_population = []
		
		## initialise population randomly
		for i in range(population_size):
			self.current_population.append(genome.sample_random_genome())

	def generate_repertoire(self,save_dir,num_generations,save_freq):
		pass

	def save_repertoire(self,save_path):
		pass

	def load_repertoire(self,load_path):
		pass