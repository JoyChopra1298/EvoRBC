from evo_rbc.main.ant_map_elites.common import get_MAPElites
import click

save_dir = "output/"

@click.command()
@click.option('--num_iterations', type=int, default=1000, help='number of iterations to run on repertoire')
@click.option('--save_freq',type=int,default=25,help='number of iterations after which to save/checkpoint repertoire')
@click.option('--num_processes',type=int,default=1,help='number of processes to run for evaluations to paralellise work')
@click.option('--visualise',type=bool,default=False,help='visualise each evaluation of environment')
@click.option('--load_path',type=str,default=None,help='path to load a pre-built repertoire and iterate further on it')
def main(num_iterations,save_freq,visualise,load_path,num_processes):
	map_elites = get_MAPElites()
	if(load_path is not None):
		map_elites.load_repertoire(load_path)
	map_elites.generate_repertoire(num_iterations=num_iterations,save_dir=save_dir,save_freq=save_freq,visualise=visualise,mutation_stdev=0.01,num_processes=num_processes)

if __name__ == "__main__":
	main()
