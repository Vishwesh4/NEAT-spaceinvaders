# NEAT-spaceinvaders
We use NeuroEvoultion of Augumenting Topologies or NEAT to solve the famous ATARI game SpaceInvaders with
the help of OpenAI's gym and NEAT-python

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Libraries needed

```
neat
gym
cv2
pickle
graphviz
time
numpy
matplotlib
```
## Introduction
Often its popular to use reinforcement learning to solve games. Evolutionary algorithm inspite being not
so popular in its application, sometimes even outperforms reinforcement learing. Using NEAT, I explore
through my project whether the algorithm is robust enough to solve ATARI games like SpaceInvaders. For the
game environment I have used OpenAI's [gym](https://gym.openai.com/envs/SpaceInvaders-v0/). The package
for NEAT algorithm is neat built by CodeReclaimers

## Approach
The bot or agent performs 6 actions namely:
  1. NOOP or No operation
  2. FIRE
  3. RIGHT
  4. LEFT
  5. RIGHTFIRE
  6. LEFTFIRE

Using NEAT we evovle a feed-forward neural network giving one of these 6 outputs. The input provided varies
based on these three methods

- <ins>Array input bot</ins>
  In this method we feed the network with important information extracted from the image of the current
  game state. The information fed is an array containing:-
    1. x,y coordinates of the bot
    2. x,y coordinate of all the aliens
    3. x,y coordinate of the nearing bullet in the vicinity space of the bot
    4. Aliens killed
  We noticed that even though the code ran pretty fast due to less number of inputs hence less time to
  calculate output but it still didn't perform well. It stagnated rather quickly and dies quickly. It
  performed only slightly better than a random bot.
  <b>Overall the bot was not able to beat the game</b>

- <ins>Image_v1 input bot</ins>
  In this method we feed the network directly with the image of the current game state. To make the
  algorithm run fastly, we resized the image and flattened it into a 1331 sized array. The bot was able
  to beat the game, however it was observed that the bot was consistent in killing decent number of bots
  but it seldom killed high number of enemies. In this method we observed slow yet steady growth in
  number of enemies killed with each generation.
  <b>Overall it took 847 generations to beat the game</b>

- <ins>Image_v2 input bot</ins>  
  In this method we feed the network a cropped version to the bot, as its sufficient to provide the bot
  with just the aliens in the vicinity of the bot. Also we modified the bullet and elongated it till the
  first object it strikes. This helps the bot understand the importance of dodging the bullets. We also
  were able to increase the resolution hence we preserve the speed of the algorithm. We observed that
  these changes forced the bot to learn a new stratergy to beat the game. In this method we observed
  stark growth with high stand deviations between different genomes in the generation.
  <b>Overall it took 117 generations to beat the game</b>

### Fitness Function
For any evolution based algorithm, fitness funtion is a very important part of the algorithm. In this we
use the following fitness function:-<br>
<font size="5">fitness = 0.8xhigh_score + 0.01xframe + 11xaliens_dead + 0.5xcounter + penalty</font><i></i><br>
Variables:-
  - frame: denotes how many frames the bot survived
  - aliens_dead: denotes how many aliens it killed
  - counter: denotes how much the bot moved from its previous position, the more movement will lead to higher value of counter
  - penalty: If the bot is in bullet's trajectory then we add a penalty

## Simulation
- Random bot

![Alttext](https://raw.github.com/Vishwesh4/NEAT-spaceinvaders/master/images_git/random_bot.gif)

- Input array bot<br>
We observe that the bot doesnt perform well and has not learnt anything

![Alttext](https://raw.github.com/Vishwesh4/NEAT-spaceinvaders/master/images_git/array_input_bot.gif)

- Image_v1 bot<br>
We observe that the bot is moving constantly and shooting. This is improving its survivabiltiy, in
further genrations the bot becomes better in surviving by constant moving. However its not moving based
on the bullet trajectory, hence it dies often. Sadly, I wasn't able to have a better footage which is
captured using gym's monitor class

![Alttext](https://raw.github.com/Vishwesh4/NEAT-spaceinvaders/master/images_git/image_bot_v1.gif)

- Image_v2 bot
We observe that the bot has found a really intresting stratergy. It now hides behind a rock and shoots.
This exponentially increases its survivabiltiy and hence it was able to beat the game faster.

![Alttext](https://raw.github.com/Vishwesh4/NEAT-spaceinvaders/master/images_git/image_bot_v2.gif)

## Visualization
- Image_v1 average fitness graph
We see that the progress is pretty slow and steady. The graph shows the fitness after 400th genration

![Alttext](https://raw.github.com/Vishwesh4/NEAT-spaceinvaders/master/images_git/avg_fitness_image.svg?sanitize=true)

- Image_v2 average fitness graph
We see that the progress is fast though with high standard deviation. The spike denotes finishing the
game which gives a default fitness value to cross the threshold

![Alttext](https://raw.github.com/Vishwesh4/NEAT-spaceinvaders/master/images_git/avg_fitness_image_v2.svg?sanitize=true)

- Image_v2 speciation graph

![Alttext](https://raw.github.com/Vishwesh4/NEAT-spaceinvaders/master/images_git/speciation1.svg?sanitize=true)

- What the bot sees

![Alttext](https://raw.github.com/Vishwesh4/NEAT-spaceinvaders/master/images_git/bots_vision.png)

- Network
The network is too big, so click on the line to see

![Alttext](https://raw.github.com/Vishwesh4/NEAT-spaceinvaders/master/plots_image/nets2.gv.svg?sanitize=true)

## Running the test
Before running the main code ensure that plots_image,recording_image,saved_models_image directories are
created
File description
* `1 - AI_bot_array.py` - Main code based on input array method
* `2 - AI_bot_image.py` - Main code based on image_v1 input method
* `3 - AI_bot_image_v2.py` - Main code based on image_v2 input method
* `4 - config` - contains all the important parameters for NEAT algorithm. Please follow the docs for more information
* `5 - inputgenerator.py` - code for extracting important information for the main code
* `6 - visualize.py` - code for genrating visualization of network
* `7 - neat-checkpoint-116` - checkpoint for image_v2, can start training after gen 116 by loading this file
* `8 - best_bot.pkl` - genome or network data for the best bot for image_v2
* `9 - plots_image` - directory containg all graphs for image_v2

## Reference
- [NEAT documentation](https://neat-python.readthedocs.io/en/latest/neat_overview.html)
- [NEAT paper](http://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf)
- [OpenAI's gym](https://gym.openai.com/docs/)
## Further Note
The code can be modified and used for other NEAT based problems by just changing the eval_genomes
functions which helps in modification of fitness score of all genomes in the genration
