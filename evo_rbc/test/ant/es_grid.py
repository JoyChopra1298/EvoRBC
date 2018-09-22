from evo_rbc.qd_solver.es_grid_generator import ES_Grid_Generator
from evo_rbc.genome.ant_genome import AntGenome
from evo_rbc.env.ant_env import AntEAEnv
import numpy as np
import evo_rbc.main.utils as test_utils

seed = 1
batch_size = 3
max_time_steps_qd=1000
max_time_steps_task=2000
visualise = False

#grid details
num_dimensions = 2
lower_limit = np.array([-1.0,-1.0])
upper_limit = np.array([1.0,1.0])
resolution = np.array([0.5,0.5])

#initialise environment, genome and repertoire generator
ant_env = AntEAEnv(seed=seed,max_time_steps_qd=max_time_steps_qd,max_time_steps_task=max_time_steps_task)
ant_genome = AntGenome(seed=seed)
es_grid_generator = ES_Grid_Generator(env=ant_env,qd_function=ant_env.qd_steady_runner,genome_constructor=AntGenome,
                                      num_dimensions=num_dimensions,lower_limit=lower_limit,upper_limit=upper_limit,
                                      resolution=resolution,batch_size=batch_size,seed=seed)

test_utils.print_heading("Check grid initialisation")
print(es_grid_generator.container.grid)

test_utils.print_heading("Check grid shape")
print(es_grid_generator.container.grid.shape)


num_iterations = 2
save_freq = 1
save_dir = "output/"
es_grid_generator.generate_repertoire(num_iterations=num_iterations,save_freq=save_freq,visualise=visualise,save_dir=save_dir)
test_utils.print_heading("Grid genome parameters at first index")
for index in np.ndindex(es_grid_generator.container.num_bins):
    zero_index = index
print(es_grid_generator.container.grid[zero_index]["genome"].parameters)

test_utils.print_heading("Grid after repertoire generation")
print(es_grid_generator.container.grid)