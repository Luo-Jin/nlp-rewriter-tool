import curses, traceback
from curses import wrapper

import torch
import numpy as np

# dedine line size and row size
line_size = 200
row_size = 80
# current sentence position
s_pos = 0
y = 0
x = 0
# initialize sentences list
# [str,style,y,x]
s = [["i'm a tester.",0,0,0]
    ,["this is a new story, which has been downloaded for many times.",0,0,0]
    ,["that is a new story and also been downloaded very often.", 0,0,0]
     ,["Mom will take you to the park this afternoon by feet.",0,0,0]]

def main(stdscr):
    # Clear screen
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_CYAN, curses.COLOR_WHITE)
    curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
    movesentences(stdscr, 0)
    stdscr.refresh()
    # accept user's input
    while True:
        c = stdscr.getkey()
        if c == 'KEY_UP':
            stdscr.clear()
        elif c == 'KEY_RIGHT':
            movesentences(stdscr,1)
        elif c == 'KEY_LEFT':
            movesentences(stdscr,-1)
        elif c == 'q':
            break  # Exit the while loop
        elif c == curses.KEY_HOME:
            x = 0


def movesentences(stdscr,pos):
    global s_pos
    s[s_pos][1] = 1
    s_pos = s_pos + pos
    if s_pos > len(s) - 1:
        s_pos = 0
    if s_pos < 0:
        s_pos = 0
    s[s_pos][1] = 2
    stdscr.clear()
    x = 0
    y = 0
    stdscr.addstr(y, x, "Choose sentence using < and > arrow:", curses.color_pair(3))
    y = y + 2
    x = x + 4
    for i in np.arange(len(s)):
        total_size = x + len(s[i][0])
        l = [line_size] * int(total_size / line_size)
        l.append(total_size % line_size)
        str_len = len(s[i][0])
        st = 0

        for j in np.arange(len(l)):
            et = min((l[j] - x),str_len) + st
            if j == 0:
                s[i][2] = y
                s[i][3] = x
            stdscr.addstr(y, x, s[i][0][st:et],curses.color_pair(s[i][1]))
            x = et-st+x if et-st+x < line_size else 0
            if x == 0:
                y = y+1
            str_len = str_len - et + st
            st = et
    stdscr.move(s[s_pos][2],s[s_pos][3])

if __name__=='__main__':
    wrapper(main)

