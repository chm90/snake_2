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

It might seem like a task that would be easy to solv with traditional nethod, why solve with learning?
It turns out that we have an extreme amount of states, and that makes it a hard learning problem, but for which deep learnign is well suited.

## Requirements

1. We use a modified version of open ai baselines. Install https://github.com/chm90/baselines.
2. Tensorflow
3. Bunch of other stuff in general

## Models

We have conducted experiments with 3 r.l models which we will discuss briefly bellow.

### DQN

Deep Q learning is basically pure Q-learning, using a deep network as the Q-value estimator isntead of the tradittional table. This enables DQN to work in much larger state spaces such as snake. Except for using a deep net, DQN allso uses several tricks to make the deep learning converge, like using experience replay and importance sampling, i.e sampling experience based on relevance. The main reason using an experience buffer is required in the dqn approach is that deep networks require updates to be uncorrelated, so we cant just apply the updates we recieve from the environment right away, since this would brake that premise.

### A2C

A2C stands for Advantage Actor-Critic. By using multiple environments to provide uncorrelated updates instead of replay memmory,A3C gains much in computational and conseptual complexity. A3C uses an actor critic based approach where instead of learning only the Q-value(the critic), an actor is learned as well using the policy gradient approach. This policy gradient is improved using the learned critic. 

### PPO
We use the PPO model for training bacause of its good tradeoff between easy parameter tuning, fast convergence and simplicity. The PPO model is a policy gradient based approach. We are using it in a setting closly related to A2C, in that it is a Actor Critic based approach,where both actor and critic networks are trained. The critic is then used to learn the actor in a more efficent manner. The main innovation of PPO over A2C is that it uses a novel Policy gradient formulation, which hinders extensively large policy updates. We use a 3 layer mlp as our network, with distinct final layers for the value and policy parts.

## Results





There are approximately as many states in a five-by-five game of snake as there are ants on Earth -- about 100 000 000 000 000 000. [1] Problematic. How to deal? Well, we reduce the state space to the eight cells surrounding the ant. I mean snake. Then we use plain Q learning. Probably will try some other function estimator in the future. That will be on-policy though (i.e. uses the same policy to estimate value as it does to choose actions) as opposed to the Q learning which simply takes the next state's best action as the value.

  [1] https://www.quora.com/How-many-ants-are-there-in-the-world
