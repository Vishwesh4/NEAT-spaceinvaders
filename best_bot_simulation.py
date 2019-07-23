import gym
import numpy as np
import cv2
import time
import neat
from inputgenerator import inputgen
from visualize import plot_species, draw_net, plot_stats
import pickle

env = gym.make('SpaceInvaders-v0')
#To record in particular episodes
env = gym.wrappers.Monitor(env, "recording_image_best_bot",video_callable=lambda episode_id: episode_id%50==0,force=True)
#Size of the game box
print(env.observation_space)
print(env.unwrapped.get_action_meanings())
action_name = {0:'NOOP', 1:'FIRE', 2:'RIGHT', 3:'LEFT', 4:'RIGHTFIRE', 5:'LEFTFIRE'}

#parameters (priors) we got from analyzing images for faster computation
starting_pixel = 114
self_y = 192
search_x_start = 22
search_x_end = 139
rock_color = 107

observation = env.reset()
high_score = 0
frame = 0
my_pos_currrent = 0
counter = 0
#creates a feed forward network from the genome
config = neat.Config(neat.DefaultGenome,neat.DefaultReproduction,
                    neat.DefaultSpeciesSet,neat.DefaultStagnation,'config')
with open("best_bot.pkl", "rb") as f:
    best_bot = pickle.load(f)
net = neat.nn.feed_forward.FeedForwardNetwork.create(best_bot,config)
#Simulation starts
while True:
    frame += 1
    env.render()
    #Processing observation image to get our set of inputs
    img = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
    imp_values = inputgen(img)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #Extracting basic info that will be used for fitness
    aliens_dead = imp_values[-1]
    my_pos = imp_values[0]

    #For extending the bullet to the ground or rock , whichever comes first
    x,y = imp_values[-3],imp_values[-2]
    #For finding whether rock comes in closest bullets trajectory
    loc = np.where(img[:,x]==rock_color)
    if len(loc[0])==0:
        img[y:,x] = 200
    else:
        img[y:loc[0][0],x] = 200
    #Giving input as image
    ob = img[starting_pixel:self_y+2,search_x_start-5:search_x_end+5]
    h_ob,w_ob = ob.shape #184,127
    #Sizing it down for faster network decision making, big input takes more time to evaluate
    inx = int(h_ob/3)
    iny = int(w_ob/3)
    ob = cv2.resize(ob, (iny, inx))
    ob = np.reshape(ob, (inx,iny))
    imgarray = np.ndarray.flatten(ob)

    ai_decision = net.activate(imgarray)
    action = np.argmax(ai_decision)
    observation, reward, done, info = env.step(action)
    #for the AI to learn how to dodge the bullet
    if done:
        break
env.close()
