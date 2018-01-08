import termin
import time
import snake
import gym

def main(width=5, height=5):
    #g = snake.game.from_size(width, height)
    env = gym.make('Snake-v0')
    env.reset()
    g = env.game
    with termin.inputs() as inputs:
        direction = snake.right
        while not g.is_over:
            print(g)
            time.sleep(0.5)
            inp = direction
            while inp is not None:
                inp = next(inputs)
                direction = inp if inp in snake.dirs else direction
            try:
                ob,reward,done,info = env.step(snake.dirs.index(direction))
            except snake.GameOver as e:
                print('game over!', *e.args)
                print('score:', g.score)
            print(ob,reward,done,info)

if __name__ == "__main__":
    main()
