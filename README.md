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

## Requirements

1. We use a modified version of open ai baselines. Check https://github.com/chm90/baselines.
2. Tensorflow
3. Bunch of other stuff in general

## Model

We use the PPO model for training bacause of its fast convergence. We use a 3 layer mlp as our policy.






There are approximately as many states in a five-by-five game of snake as there are ants on Earth -- about 100 000 000 000 000 000. [1] Problematic. How to deal? Well, we reduce the state space to the eight cells surrounding the ant. I mean snake. Then we use plain Q learning. Probably will try some other function estimator in the future. That will be on-policy though (i.e. uses the same policy to estimate value as it does to choose actions) as opposed to the Q learning which simply takes the next state's best action as the value.

  [1] https://www.quora.com/How-many-ants-are-there-in-the-world
