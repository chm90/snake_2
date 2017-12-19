import termin
import time

def main(width=5, height=5):
    g = game.from_size(width, height)
    with termin.inputs() as inputs:
        direction = right
        while not g.is_over:
            print(g.board)
            time.sleep(0.5)
            inp = direction
            while inp is not None:
                inp = next(inputs)
                direction = inp if inp in dirs else direction
            try:
                g.next(direction)
            except GameOver as e:
                print('game over!', *e.args)
                print('score:', g.score)

if __name__ == "__main__":
    main()
