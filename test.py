#import stanza
#stanza.download('en',model_dir='stanza')       # This downloads the English models for the neural pipeline
# nlp = stanza.Pipeline('en',model_dir='stanza',processors='tokenize,ner') # This sets up a default neural pipeline in English
# doc = nlp("Auckland")
# print(*[f'token: {ent.text}\ttype: {ent.type}' for sent in doc.sentences for ent in sent.ents], sep='\n')


# from matplotlib  import pyplot as plt
# import torch
# l = {}
# #x = torch.tensor([[torch.from_numpy(t) for t  in torch.load('test/sig_l1_e1000_b5000_l0.5.pt')]])
# l['L1Loss, Sigmoid'] = torch.tensor([torch.from_numpy(t) for t  in torch.load('test/sig_l1_e1000_b5000_l0.5.pt')])
# l['L1Loss, noSigmoid'] = torch.tensor([torch.from_numpy(t) for t  in torch.load('test/nosig_l1_e1000_b5000_l0.5.pt')])
# l['SmoothL1Loss, Sigmoid'] = torch.tensor([torch.from_numpy(t) for t  in torch.load('test/sig_sml1_e1000_b5000_l0.5.pt')])
# l['SmoothL1Loss, noSigmoid'] = torch.tensor([torch.from_numpy(t) for t  in torch.load('test/nosig_sml1_e1000_b5000_l0.5.pt')])
# i = 0
# for k,v in l.items():
#     i = i + 1
#     v = v.view(int(v.size(0) / 80), 80)
#     ax = plt.subplot(220+i)
#     ax.set_title(k)
#     plt.plot(torch.arange(v.size(0)),torch.mean(v,dim=-1))
#     plt.xlabel('epoch')
#     plt.ylabel('loss (mean)')
# plt.subplots_adjust(wspace=0.5,hspace=0.5)
# plt.show()
import  time
import curses
import curses.panel as panel
import curses, traceback
from curses import wrapper
import spacy
import torch
import numpy as np
# line_size = 90
# row_size = 80
# s_pos = 0
# y = 0
# x = 0
# pos = 1
# nlp = spacy.load("en_core_web_sm")
# f = open('sample.txt', mode='r')
# texts = f.readlines()
# txt = []
# for i in np.arange(len(texts)):
#     p = texts[i]
#     p = p[0:len(p)-1]
#     doc = nlp(p)
#     sents = []
#     for sent in doc.sents:
#         sents.append([sent,0,0,0])
#     txt.append(sents)
# f.close()
#
# s_pos = s_pos + pos
# print(txt)
# txt1=[]
# for p in txt:
#     for s in p:
#         txt1.append(s)
# print(txt1)
# if s_pos > len(sents[:][:]) - 1:
#     s_pos = 0
# if s_pos < 0:
#     s_pos = 0
# sents[0][s_pos][1] = 1
# sents[0][s_pos][1] = 2
# offset_x = 4
# offset_y = 1
# for p in txt:
#     for i in np.arange(len(p)):
#         total_size = x + len(p[i][0])
#         l = [line_size] * int(total_size / line_size)
#         l.append(total_size % line_size)
#         str_len = len(p[i][0])
#         st = 0
#         for j in np.arange(len(l)):
#             et = min((l[j] - x), str_len) + st
#             if j == 0:
#                 p[i][2] = y
#                 p[i][3] = x
#             print("string :{},st:{},et:{},y:{},x:{},color:{}".format(p[i][0],st,et,y+offset_y,x+offset_x,p[i][1]))
#             #txt_box.addstr(y + offset_y, x + offset_x, p[i][0][st:et], curses.color_pair(p[i][1]))
#             x = et - st + x if et - st + x < line_size else 0
#             if x == 0:
#                 y = y + 1
#             str_len = str_len - et + st
#             st = et
#     y = y+1
#     x = 0

# Curses modules
import curses
from sys import argv
from curses import KEY_MOUSE
from random import randint
from time import time, sleep

# Get chocolate bar size and player names
players = {}
try:
    x = int(argv[1])
    y = int(argv[2])
    players[1] = str(argv[3])
    players[2] = str(argv[4])
except:
    print("Error: please supply a valid chocolate bar size and two valid player names.")
    # E.g. "python3 chomp.py 15 5 Player1 Player2"
    exit(1)

# Init curses
main = curses.initscr()

# Check if window is big enough
window_y,window_x = main.getmaxyx()
if window_y <= y + 2 or window_x <= x + 2:
    print("Terminal must be at least " + str(x + 2) + "x" + str(y + 2) + ".")
    curses.endwin()
    exit(1)

# Create new curses window of size (x+2)*(y+2) to accomodate border
win = curses.newwin(y + 2, x + 2, 0, 0)
win.keypad(1)
curses.noecho()
curses.curs_set(0)
win.border(0)
win.nodelay(1)
curses.mousemask(1)

# Define useful functions
# Return list with all coordinates of chocolate bar pieces
def get_chocolate(y,x):
    coords = []
    for i in range(1, y + 1): # Go over all horizontal lines (border uses y = 0)
        for p in range(1, x + 1): # Go over every piece in line (border uses x = 0)
            coords.append((i, p)) # Curses works with (y,x) coordinates system
    return coords

# Print chocolate pieces
def print_chocolate(coords,y,x): # It is easier to go over all coordinates
    for i in range(1, y + 1):    # checking if (y,x) is a piece of chocolate
        for p in range(1, x + 1):
            if (i,p) == (y, 1):  # Bottom left corner is the poisonous piece
                win.addch(i, p, "*")
            elif (i,p) in coords:
                win.addch(i, p, '#')
            else:
                win.addch(i, p, ' ')

# Eat chocolate pieces
def eat(coords, x, click_y, click_x):
    for i in range(1, click_y + 1): # Go over rows from top to chocolate piece included (+1)
        for p in range(click_x, x + 1): # Go over columns from clicked point to right edge
            if (i,p) in coords:
                coords.pop(coords.index((i, p)))
    return coords

coords = get_chocolate(y,x) # Define coordinates list

player = 1 # Define player turn

moves = 0 # Define moves

game_start = time() # Start timer

# Start curses print loop
while True:
    win.border(0) # Set window border
    key = win.getch() # Listen for keys pressed or mouse clicks
    if key == ord('q'): # When 'q' is hit quit the game
        while True: # Ask for confirmation
            key = win.getch()
            win.addstr(0, 1, "| Quit? (y/n) |")
            if key == ord('n'): # Resume game
                break
            if key == ord('y'): # Close window and exit
                curses.endwin()
                exit(0)

    if key == KEY_MOUSE: # If mouse is clicked
        _, click_x, click_y, _, _ = curses.getmouse() # Get clicked coordinates
        # Poisonous piece cannot be chosen
        if (click_y,click_x) in coords and (click_y,click_x) != (y, 1):
            moves += 1 # Increase number of moves by one
            player = abs(player - 3) # Switch between 1 and 2 --> |1-3|=2 and |2-3|=1
            coords = eat(coords, x, click_y, click_x)

    print_chocolate(coords, y, x) # Print chocolate pieces

    if len(coords) == 1: # If only 1 piece is left, then it must be the poisonous one
        break # End curses loop

game_end = time() # End timer

game_time = game_end - game_start # Calculate game time

curses.endwin() # Close window

# Print results
print("----------------")
print(str(players[abs(player - 3)]) + " WINS!")
print("Number of moves: " + str(moves))
print("Game time: " + str(round(game_time, 3)) + "s")
print("----------------")

exit(0)