from .repertoire_generator import Repertoire_Generator
from .container.grid import Grid

class MAP_Elites(Repertoire_Generator):

	def __init__(self,env,genome,selector,num_dimensions,lower_limit,upper_limit,resolution,mutation_rate=0.1,population_size=100):
		self.container = Grid(num_dimensions=num_dimensions,lower_limit=lower_limit,upper_limit=upper_limit,resolution=resolution)
		self.selector = selector
		super().__init__(env,genome,mutation_rate,population_size)
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