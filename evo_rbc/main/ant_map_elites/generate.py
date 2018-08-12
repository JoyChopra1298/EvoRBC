from evo_rbc.main.ant_map_elites.common import get_MAPElites

num_iterations = 3
save_freq = 2
visualise = False
save_dir = "output/"

map_elites = get_MAPElites()
map_elites.generate_repertoire(num_iterations=num_iterations,save_dir=save_dir,save_freq=save_freq,visualise=visualise)
