from .container import Container
import numpy as np

class Grid(Container):

	def __init__(self,num_dimensions,lower_limit,upper_limit,resolution):
		"""lower_limit, upper_limit and resolution should be np arrays of shape (num_dimension) to specify them for each dimension"""
		super().__init__()
		self.num_dimensions = num_dimensions
		self.lower_limit = lower_limit
		self.upper_limit = upper_limit
		self.resolution = resolution
		self.num_bins = np.ceil((upper_limit - lower_limit)/resolution).astype(int)
		
		#initialise grid with empty bins. will store a dict in each bin {"genome":,"quality":}
		self.grid = {}
		for index in np.ndindex(tuple(self.num_bins)):
			self.grid[index] = {}

	def get_bin(self,behavior):
		"""get the bin corresponding to a particular behavior, linear mapping is used. behavior is forced to be between limits by clipping it"""
		behavior = np.clip(behavior,self.lower_limit,self.upper_limit)
		return  tuple(np.round((behavior - self.lower_limit)/(self.upper_limit-self.lower_limit)*(self.num_bins-1)).astype(int))

	def is_high_quality(self,behavior,quality):
		"""check if genome has high quality then current genome for the same behavior. also true if bin is empty
		   behavior should be a tuple with dimensions num_dimension"""
		if( (not self.grid[self.get_bin(behavior)]) or self.grid[self.get_bin(behavior)]["quality"]>quality):
			return True
		else:
			return False

	def add_genome(self,genome,behavior,quality):
		"""add the genome to container"""
		bin_index = self.get_bin(behavior)
		old_genome = self.grid[bin_index] 
		if(old_genome):
			self.total_quality += quality - old_genome["quality"]
		else:
			self.total_quality += quality
			self.num_genomes += 1

		if(quality>self.max_quality):
			self.max_quality = quality
			self.max_quality_bin = bin_index
		self.grid[bin_index] = {"genome":genome,"quality":quality}

	def save_container(self,save_dir):
		raise NotImplementedError

	def load_container(self,load_path):
		raise NotImplementedError