# REACHER DDPG 
This project is an assignment from the Udacity Deep Reinforcement Learning course. [p3_continuous-control](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet) found [here](https://github.com/udacity/deep-reinforcement-learning#dependencies). The objective is to train the tennis agents to keep the ball in play for as long as possible. the environment is considered solved if the agents reach an average score of +0.5 over 100 episodes. The score is measured by taking the highest score over the two agents.

## Enviroment

##### Observation

Type: Continues(24)

The statespace represents the current state of the paddle and the ball.

##### Actions
Type: Continues(2)

The 2 actions controll the movement of the paddle

##### Reward

A reward of +0.1 is handed out for hitting the ball over the net. A penalty of -0.1 is recieved for either letting the ball fall or hitting it out of bounds.

## Installation and setup

##### 1.
For the dependancies to this project please follow the instructions from the assignment github [here](https://github.com/udacity/deep-reinforcement-learning#dependencies)

##### 2.
To install the unity environment you'll need to download the right version for your pc drag it into the project folder and unzip it, aswell as edit the path in the notebook.

Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

##### 3.
Launch Jupyter notebook in the project folder

## How to use
When running either of the notebooks make sure you have edited the PATH variable to your version of the unity Banana enviroment

To train a model run the Training notebook

To Evaluate an agents performance run the Evaluation notebook
