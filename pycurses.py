import curses, traceback
from curses import wrapper
import spacy
import torch
import numpy as np

# dedine line size and row size
line_size = 90
row_size = 80
# current sentence position
s_pos = 0
r_pos = 0
y = 0
x = 0
# initialize sentences list
# [str,style,y,x]
paragraph = None
sentences = []

def readtxt(txt):
    nlp = spacy.load("en_core_web_sm")
    f = open('sample.txt', mode='r')
    texts = f.readlines()
    txt = []
    for i in np.arange(len(texts)):
        p = texts[i]
        p = p[0:len(p) - 1]
        doc = nlp(p)
        sents = []
        for sent in doc.sents:
            sents.append([str(sent), 0, 0, 0])
        txt.append(sents)
    f.close()
    return txt


def main():
    # read texts from file
    global paragraph, sentences
    paragraph = readtxt('sample.txt')
    for p in paragraph:
        for sent in p:
            sentences.append(sent)
    wrapper(contextwindow)



def contextwindow(stdscr):
    # Clear screen
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN, -1)
    curses.init_pair(2, curses.COLOR_WHITE, -1)
    curses.init_pair(3, curses.COLOR_RED, -1)
    # add menu bar
    stdscr.addstr(0, 5, "Choose sentence using < and > arrow:",curses.color_pair(3))
    stdscr.refresh()
    # add test box
    txt_box = curses.newwin(10,100,1,5)
    txt_box.keypad(True)
    txt_box.border()
    movesentences(txt_box, 0)
    # accept user's input
    while True:
        c = txt_box.getkey()
        if c == 'KEY_UP':
            movesentences(txt_box, 1)
        elif c == 'KEY_RIGHT':
            movesentences(txt_box,1)
        elif c == 'KEY_LEFT':
            movesentences(txt_box,-1)
        elif c == 'q':
            break  # Exit the while loop
        elif c == curses.KEY_HOME:
            x = 0


def movesentences(txt_box,pos):
    global s_pos, sentences
    sentences[s_pos][1] = 0
    s_pos = s_pos + pos
    if s_pos > len(sentences) - 1:
        s_pos = 0
    if s_pos < 0:
        s_pos = 0
    sentences[s_pos][1] = 1

    x = 4
    y = 0
    offset_x = 4
    offset_y = 1
    for para in paragraph:
        for i in np.arange(len(para)):
            total_size = x + len(para[i][0])
            l = [line_size] * int(total_size / line_size)
            l.append(total_size % line_size)
            str_len = len(para[i][0])
            st = 0

            for j in np.arange(len(l)):
                if y > 7:
                    break
                et = min((l[j] - x),str_len) + st
                if j == 0:
                    para[i][2] = y
                    para[i][3] = x
                txt_box.addstr(y+offset_y, x+offset_x,para[i][0][st:et],curses.color_pair(para[i][1]))
                x = et-st+x if et-st+x < line_size else 0
                if x == 0:
                    y = y+1
                str_len = str_len - et + st
                st = et
        y = y+1
        x = 4
    txt_box.move(sentences[s_pos][2]+offset_y,sentences[s_pos][3]+offset_x)

if __name__=='__main__':
    main()

