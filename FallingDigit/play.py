# -*- coding: utf-8 -*-
import numpy as np
import cv2
import time
import gym


import termios, fcntl, sys, os

def get_char_keyboard():
    fd = sys.stdin.fileno()

    oldterm = termios.tcgetattr(fd)
    newattr = termios.tcgetattr(fd)
    newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
    termios.tcsetattr(fd, termios.TCSANOW, newattr)

    c = None
    try:
        c = sys.stdin.read(1)
    except IOError: 
        print('io error')
        pass

    termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)

    return c

key_to_action = {
    '1':0,
    '2':1,
    '3':2,
}


if __name__ == '__main__':
    env = gym.make('FallingDigitCIFAR_3-v1')


    s = env.reset()
    while True:
        img = env.render('rgb_array')
        cv2.imshow('game', img[:, :, ::-1])
        cv2.waitKey(1)
        
        c = get_char_keyboard()
        if c == 'n':
            env.reset()
        elif c == 'q' or c == 'x':
            exit()
        else:
            s, r, done, info = env.step(key_to_action[c])
            print('R:', r, ', Done:', done)
