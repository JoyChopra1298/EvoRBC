from evo_rbc.qd_solver.map_elites import MAP_Elites
from evo_rbc.genome.ant_genome import AntGenome
from evo_rbc.env.ant_env import AntEAEnv

seed = 1
population_size = 100
mutation_rate = 0.1
max_time_steps_qd=1000
max_time_steps_task=2000
visualise = False

ant_env = AntEAEnv(seed=seed,max_time_steps_qd=max_time_steps_qd,max_time_steps_task=max_time_steps_task)
ant_genome = AntGenome(seed=seed)
map_elites = MAP_Elites(ant_env,ant_genome,population_size=population_size,mutation_rate=mutation_rate)
# print(map_elites.current_population)
ant_env.evaluate_quality_diversity_fitness(ant_env.qd_steady_runner,ant_genome,visualise)