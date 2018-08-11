import logging
import numpy as np

from evo_rbc.qd_solver.map_elites import MAP_Elites
from evo_rbc.genome.ant_genome import AntGenome
from evo_rbc.env.ant_env import AntEAEnv
from evo_rbc.qd_solver.container.grid import Grid
from evo_rbc.qd_solver.selector.curiosity_driven_selector import Curiosity_Driven_Selector

##################################
#Logger initialisation

debug_logfile = "ant_map_elites_debug.log"
logfile = "ant_map_elites.log"

# define a Handler which writes INFO messages or higher to the sys.stderr
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)-8s %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger('').addHandler(console_handler)

# define a Handler which writes all messages to a debug log file
debug_file_handler = logging.FileHandler(debug_logfile)
debug_file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s',datefmt='%m-%d %H:%M')
debug_file_handler.setFormatter(formatter)
logging.getLogger('').addHandler(debug_file_handler)

# define a Handler which writes all >=info messages to a log file
file_handler = logging.FileHandler(logfile)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
file_handler.setFormatter(formatter)
logging.getLogger('').addHandler(file_handler)

logger = logging.getLogger()

##################################
#Constants/hyperparameters
seed = 1
batch_size = 100
max_time_steps_qd=1000
max_time_steps_task=2000
visualise = False

#Grid details
num_dimensions = 2
lower_limit = np.array([-0.5,-0.5])
upper_limit = np.array([0.5,0.5])
resolution = np.array([.005,.005])
##################################

#initialise environment, genome and repertoire generator
ant_env = AntEAEnv(seed=seed,max_time_steps_qd=max_time_steps_qd,max_time_steps_task=max_time_steps_task)
ant_genome = AntGenome(seed=seed)
map_elites = MAP_Elites(env=ant_env,qd_function=ant_env.qd_steady_runner,genome_constructor=AntGenome,seed=seed,selector=Curiosity_Driven_Selector(),
	num_dimensions=num_dimensions,lower_limit=lower_limit,upper_limit=upper_limit,
	resolution=resolution,batch_size=batch_size,logger=logger)
