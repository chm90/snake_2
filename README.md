# REINFORCEMENT LEARNING FOR SNAKE

![Snake](http://m.plonga.com/public/uploads/thumbs/nokia-snake-3310-html5-classic-online.jpg)

We've all played it. But did you make a computer play it?

Feel like snaking?

```python
python3 snake.py
```

Feel like learning the snaking?

```python
python3 run_snake.py
```

Feel like watching the learned snaking?

```python
python3 play_snake.py
```

Snake it don't break it.


## Why do we snake?

It might seem like a task that would be easy to solve with traditional method, why solve with learning?
It turns out that we have an extremely high number of states, and that makes it a hard learning problem, but for which deep learning is well suited.

Also, rather than using a premade environment from OpenAI Gym, adding your own custom environment is just more fun, challenging and ultimately a better learning experience.
Snake has the benefit to be easily scalable in terms of complexity by just increasing the board size. This is convenient when we're looking to evaluate RL algorithms with a large number of CPU/GPUs at our disposal.

## Requirements

1. We use a modified version of open ai baselines. Install https://github.com/chm90/baselines.
2. Tensorflow
3. Bunch of other stuff in general

### Gym ###
To add the Snake as a gym environment (rather than stand-alone Snake) see https://github.com/carlsi/gym

## Models

We have conducted experiments with 3 r.l models that we discuss briefly below.

### DQN

Deep Q learning is basically pure Q-learning, using a deep network as the Q-value estimator instead of the traditional table. This enables DQN to work in much larger state spaces such as snake. Except for using a deep net, DQN also uses several tricks to make the deep learning converge, like using experience replay and importance sampling, i.e sampling experience based on relevance. The main reason using an experience buffer is required in the dqn approach is that deep networks require updates to be uncorrelated, so we cant just apply the updates we receive from the environment right away, since this would brake that premise.

### A2C

A2C stands for Advantage Actor-Critic. By using multiple environments to provide uncorrelated updates instead of replay memory, A2C gains much in computational and conceptual complexity. A2C uses an actor critic based approach where instead of learning only the Q-value(the critic), an actor is learned as well using the policy gradient approach. This policy gradient is improved using the learned critic. 

### PPO
We use the PPO model for training because of its good trade-off between easy parameter tuning, fast convergence and simplicity. The PPO model is a policy gradient based approach. We are using it in a setting closely related to A2C, in that it is a Actor Critic based approach,where both actor and critic networks are trained. The critic is then used to learn the actor in a more efficient manner. The main innovation of PPO over A2C is that it uses a novel Policy gradient formulation, which hinders extensively large policy updates. We use a 3 layer mlp as our network, with distinct final layers for the value and policy parts.

## Results

Training the PPO model on a 5x5 snake board on a nvidia 670 GTX and 4 core intel core i7 cpu for approximately 6 hours gave a average score of 70 for the snake and a maximum score of 115. Check video bellow for an example of the snake.

<a href="http://www.youtube.com/watch?feature=player_embedded&v=TtF2Qfnxu-k
" target="_blank"><img src="http://img.youtube.com/vi/TtF2Qfnxu-k/0.jpg" 
alt="PPO snake" width="240" height="180" border="10" /></a>


![Training with A2C](https://i.imgur.com/1Sw0Ppn.png)
Figure of mean episode length (top) and reward (bottom) during a a2c training session of about 10 million steps. It shows the difficulty barrier met when the agent has grasped how to stay alive longer and starts to eat more apples increasing its score but also the number of states that it will die in.
