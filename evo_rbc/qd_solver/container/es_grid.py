from .container import Container
import numpy as np
import copy

class ESGrid(Container):

	def __init__(self,num_dimensions,lower_limit,upper_limit,resolution):
		"""lower_limit, upper_limit and resolution should be np arrays of shape (num_dimension) to specify them for each dimension"""
		super().__init__()
		self.num_dimensions = num_dimensions
		self.lower_limit = lower_limit
		self.upper_limit = upper_limit
		self.resolution = resolution
		self.num_bins = np.ceil((upper_limit - lower_limit)/resolution).astype(int)
		
		#initialise grid with empty bins. will store a dict in each bin {"genome":,...might add more metrics later}
		self.grid = {}
		self.logger.debug("Grid initialised")
		
	def get_bin(self,behavior):
		"""get the bin corresponding to a particular behavior, linear mapping is used. behavior is forced to be between limits by clipping it"""
		behavior = np.clip(behavior,self.lower_limit,self.upper_limit)
		return  tuple(np.round((behavior - self.lower_limit)/(self.upper_limit-self.lower_limit)*(self.num_bins-1)).astype(int))

	def update_bin(self,bin_index,genome_details):
		"""updates the entry in bin. genome details consists of a dictionary of genome parameters"""
		self.grid[bin_index] = copy.deepcopy(genome_details)
		self.logger.debug("Updating details for bin "+str(bin_index))
