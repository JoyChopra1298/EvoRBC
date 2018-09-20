from .container import Container
import numpy as np
import copy

class Grid(Container):

	def __init__(self,num_dimensions,lower_limit,upper_limit,resolution,min_curiosity=0.5,curiosity_multiplier=2):
		"""lower_limit, upper_limit and resolution should be np arrays of shape (num_dimension) to specify them for each dimension"""
		super().__init__()
		self.num_dimensions = num_dimensions
		self.lower_limit = lower_limit
		self.upper_limit = upper_limit
		self.resolution = resolution
		self.num_bins = np.ceil((upper_limit - lower_limit)/resolution).astype(int)
		self.min_curiosity = min_curiosity
		self.curiosity_multiplier = curiosity_multiplier
		
		#initialise grid with empty bins. will store a dict in each bin {"genome":,"quality":}
		self.grid = {}
		self.logger.debug("Grid initialised")
		
	def get_bin(self,behavior):
		"""get the bin corresponding to a particular behavior, linear mapping is used. behavior is forced to be between limits by clipping it"""
		behavior = np.clip(behavior,self.lower_limit,self.upper_limit)
		return  tuple(np.round((behavior - self.lower_limit)/(self.upper_limit-self.lower_limit)*(self.num_bins-1)).astype(int))

	def is_high_quality(self,behavior,quality):
		"""check if genome has high quality then current genome for the same behavior. also true if bin is empty
		   behavior should be a tuple with dimensions num_dimension"""
		bin_index = self.get_bin(behavior)
		if ((bin_index in self.grid) and self.grid[bin_index]["quality"]>=quality):
			return False
		else:
			return True

	def add_genome(self,genome,behavior,quality):
		"""add the genome to container"""
		bin_index = self.get_bin(behavior)
		if(bin_index in self.grid):
			if quality > self.grid[bin_index]["quality"]:
				self.total_quality += quality - self.grid[bin_index]["quality"]
				self.update_bin(bin_index, {"genome": genome, "quality": quality, "curiosity": 1.0})
				if (bin_index == self.min_quality_bin):
					self.logger.debug(
						"found better quality for current minima. updating min quality genome. bin " + str(bin_index))
					self.min_quality = quality
					self.find_min_quality_genome()
			else:
				self.logger.debug("Tried to add a low quality genone in bin index ", self.grid[bin_index])
		else:
			self.total_quality += quality
			self.num_genomes += 1
			self.update_bin(bin_index,{"genome":genome,"quality":quality,"curiosity":1.0})

		if(quality>self.max_quality):
			self.logger.debug("new max quality genome found with quality "+str(quality)
				+" and behavior "+str(behavior)+"replacing max quality"+str(self.max_quality))
			self.max_quality = quality
			self.max_quality_bin = bin_index
		if(quality<self.min_quality):
			self.logger.debug("new min quality genome found with quality "+str(quality)
				+" and behavior "+str(behavior)+"replacing min quality"+str(self.min_quality))
			self.min_quality = quality
			self.min_quality_bin = bin_index
		self.logger.debug("new genome added bin "+str(bin_index)+" -- quality "+str(quality)+" bin quality "+str(self.grid[bin_index]["quality"]))

	def update_bin(self,bin_index,genome_details):
		"""updates the entry in bin. genome details consists of a dictionary of genome parameters"""
		self.grid[bin_index] = copy.deepcopy(genome_details)
		self.logger.debug("Updating details for bin "+str(bin_index))

	def update_curiosity(self,bin_index,curiosity):
		"""updates curiosity for the genome"""
		self.grid[bin_index]["curiosity"] = curiosity
		self.logger.debug("Updating curiosity for bin "+str(bin_index)+" new curiosity "+str(curiosity))


	def find_min_quality_genome(self):
		### since this is rare so iterating whole container for now. need better data structure
		for bin_index,genome_details in self.grid.items():
			if(genome_details["quality"]<self.min_quality):
				self.min_quality = genome_details["quality"]
				self.min_quality_bin = bin_index

	def get_metrics(self):
		return {"Total Quality":self.total_quality,
		"Max Quality":self.max_quality,
		"Max Quality Bin":self.max_quality_bin,
		"Min Quality":self.min_quality,
		"Min Quality Bin":self.min_quality_bin,
		"Number of genomes":self.num_genomes,
		"Mean quality":self.total_quality/self.num_genomes}