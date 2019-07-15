import gym
import numpy as np
import cv2
import time
import neat
import pickle
from inputgenerator import inputgen
from visualize import plot_species, draw_net, plot_stats

env = gym.make('SpaceInvaders-v0')
env = gym.wrappers.Monitor(env, "recording",force=True)
print(env.observation_space)
print(env.unwrapped.get_action_meanings())
action_name = {0:'NOOP', 1:'FIRE', 2:'RIGHT', 3:'LEFT', 4:'RIGHTFIRE', 5:'LEFTFIRE'}

def eval_genomes(genomes,config):
#evalutes genomes based on their fitness scores
    for genome_id,genome in genomes:
        observation = env.reset()
        high_score = 0
        frame = 0
        my_pos_currrent = 0
        counter = 0
        #creates a network from the genome
        net = neat.nn.recurrent.RecurrentNetwork.create(genome,config)
        # done = False
        #Simulation starts
        while True:
            frame += 1
            env.render()
            #Processing observation image to get our set of inputs
            img = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
            input = inputgen(img)
            #Extracting basic info that will be used for fitness
            aliens_dead = input[-1]
            my_pos = input[0]

            ai_decision = net.activate(input)
            action = np.argmax(ai_decision)
            observation, reward, done, info = env.step(action)

            if my_pos!=my_pos_currrent:
                counter+=1
            my_pos_currrent = my_pos
            high_score += reward
            if done:
                break
        fitness = high_score + 0.01*frame + 3*aliens_dead + 0.5*counter
        print(genome_id,fitness)
        genome.fitness = fitness
        if aliens_dead==36:
            break
# Setting the configuration
config = neat.Config(neat.DefaultGenome,neat.DefaultReproduction,
                    neat.DefaultSpeciesSet,neat.DefaultStagnation,'config')

#Initializing the population class
p = neat.Population(config)
p = neat.Checkpointer.restore_checkpoint('./saved_models/neat-checkpoint-436')

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10,filename_prefix='./saved_models/neat-checkpoint-'))

#run upto 400 generation
best_bot = p.run(eval_genomes,400)
env.close()

draw_net(config, best_bot,view=False,filename='./plots/nets.svg')
plot_stats(stats, ylog=False, view=False,filename='./plots/avg_fitness.svg')
plot_species(stats, view=False,filename='./plots/speciation.svg')

# with open('best_bot.pkl', 'wb') as output:
    # pickle.dump(best_bot, output, 1)
