# SSPlayer

SSPlayer is an implementation of Deep Q-Network, as described in "Playing Atari with Deep Reinforcement Learning" by DeepMind, to play games.
The complete description of this project can be found [here](www.vishnudevarakonda.com/ML/ssplayer)

# Snake
Snake was the first game that I was successfully able to train. Below is an example of the results. <br>
![Alt text](https://vishnudevarakonda.com/res/ML/ssplayer/s9.gif),![Alt text](https://vishnudevarakonda.com/res/ML/ssplayer/s9-2.gif)

### Results
After running the training operation for about a 200 thousand states where the greed gradually reduces from 100% to 10% over the first 100 thousand states, the result of how well the network learned to play snake can be seen [here](www.vishnudevarakonda.com/ML/ssplayer).

Clearly, this is not the optimal solution for playing the game. Yet, we can see the network has learned some policy that is effective at finding the food in different locations and guiding the snake there. As discussed earlier, the Q function will find the optimal policy as the number of states approaches infinity. Therefore, a longer training period would ensure that the network will get better.
