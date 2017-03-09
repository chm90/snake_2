"terminal input"

import sys
from select import select
from os import O_NONBLOCK, read
from fcntl import F_GETFL, F_SETFL, fcntl
from termios import ECHO, ICANON, TCSADRAIN, VMIN, VTIME, tcgetattr, tcsetattr
from contextlib import contextmanager

@contextmanager
def drained(f):
    if hasattr(f, 'fileno'):
        f = f.fileno()
    fl = fcntl(f, F_GETFL)
    fcntl(f, F_SETFL, fl | O_NONBLOCK)
    tcold = tcgetattr(f)
    tcnew = tcgetattr(f)
    tcnew[3] = tcnew[3] & ~(ECHO | ICANON)
    tcnew[6][VMIN] = 0
    tcnew[6][VTIME] = 0
    tcsetattr(f, TCSADRAIN, tcnew)
    try:
        yield f
    finally:
        tcsetattr(f, TCSADRAIN, tcold)
        fcntl(f, F_SETFL, fl)

def read_input(f):
    esc, ansi = False, False
    if hasattr(f, 'fileno'):
        f = f.fileno()
    while True:
        if select([f], [], [], 0) == ([], [], []):
            yield
            continue
        for ch in read(f, 4):
            if   ch == ord(' '): yield 'space'
            elif ch == ord('q'): yield 'quit'
            elif ansi and ch == ord('A'): yield 'up'
            elif ansi and ch == ord('B'): yield 'down'
            elif ansi and ch == ord('C'): yield 'right'
            elif ansi and ch == ord('D'): yield 'left'
            elif esc and ch == ord('['): pass
            elif ch == 0x1b: pass
            else: print('invalid input', (ch, esc, ansi))
            ansi = esc and ch == ord('[')
            esc = ch == 0x1b
        inputs = read_input(f)

@contextmanager
def inputs(f=sys.stdin):
    with drained(f):
        yield read_input(f)
