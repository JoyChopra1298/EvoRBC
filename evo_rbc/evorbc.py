from evo_rbc.qd_solver.map_elites import MAP_Elites
from evo_rbc.arbitrator.neat import NEAT
from evo_rbc.mapper.cartesian_mapper import CartesianMapper
import click

import gym
from osim.env import ProstheticsEnv

@click.command()
@click.option('--envid', type=str, default='ProstheticsEnv', help='the name of the environment in OpenAI Gym format')
# @click.option('--logdir', type=str, default=None, help='the path to where logs and policy pickles should go. If not specified, creates a folder in /tmp/')
@click.option('--num_generations', type=int, default=500, help='the number of generations to run')
@click.option('--save_freq', type=int, default=10, help='the number of generations aftere which to save')
# @click.option('--num_cpu', type=int, default=1, help='the number of CPU cores to use (using MPI)')
# @click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code')
@click.option('--load_repo_path',type=str,default=None,help='the path from where to load an existing repertoire')
@click.option('--load_arbitrator_path',type=str,default=None,help='the path from where to load an existing arbitrator')
@click.option('--save_dir',type=str,default="./",help='the directory where evolved values are saved')
def train(envid,num_generations,save_freq,load_repo_path,load_arbitrator_path,save_dir):
	if(envid=='ProstheticsEnv'):
		env = ProstheticsEnv()
	else:
		env = gym.make(envid)
	map_elites = MAP_Elites(env)
	if(load_repo_path):
		repertoire = map_elites.load_repertoire(load_repo_path)
	else:
		repertoire = map_elites.generate_repertoire(save_dir,num_generations,save_freq)

	mapper = CartesianMapper()

	arbitrator = NEAT(env,repertoire,mapper)
	
	if(load_arbitrator_path):
		arbitrator = arbitrator.load(load_arbitrator_path)
	else:
		arbitrator.evolve(save_dir,num_generations,save_freq)

	# test()

if __name__ == '__main__':
	train()


