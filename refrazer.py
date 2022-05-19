# -*- coding: utf-8 -*-
###############################################
#   File    : refrazer.py
#   Author  : Jin luo
#   Date    : 2022-05-17
#   Input   : None
#   Output  : None
###############################################
'''
a GUI for the rewriter tool
'''

import curses
import torch
import copy
import os
from curses import wrapper
from configparser import ConfigParser
from cbs import get_sentence
from embeddings import bert_tokenizer
from textbox import TextBox

# config object
file_path = os.path.join(os.path.abspath("."), "refrazer.ini")
config = ConfigParser()
config.read(file_path)
# text frame objects
screen = None
boxes = {}
current_box = ""
# initialize sentences list
replaced_sents = []
# model parameters
σ = 0
k = 0
batch = 1



def update_status():
    global boxes,current_box
    for k in boxes:
        boxes[k].calc_position(0)
        boxes[k].print_text()
        if k != "banner_box":
            if k == current_box:
                boxes[k].get_win().attron(curses.color_pair(1))
            else:
                boxes[k].get_win().attron(curses.A_NORMAL)
            boxes[k].get_win().box()
            boxes[k].get_win().addstr(0, 4, boxes[k].name)
            boxes[k].get_win().attroff(curses.color_pair(1))
            boxes[k].get_win().attroff(curses.A_NORMAL)
        boxes[k].get_win().refresh()

def read_text(file):
    f = open(file, mode='r')
    texts = f.readlines()
    f.close()
    return texts

def replace_sentence():
    paragraph = boxes["output_box"].get_paragraph()
    if len(paragraph) > 0:
        sent = paragraph[boxes["output_box"].current_para_id][boxes["output_box"].current_sent_id]
        txt = str(sent[0])
        replaced_sent = copy.deepcopy(boxes["main_box"].get_paragraph()[boxes["main_box"].current_para_id][boxes["main_box"].current_sent_id])
        replaced_sents.append(replaced_sent)
        boxes["main_box"].get_paragraph()[boxes["main_box"].current_para_id][boxes["main_box"].current_sent_id][0] = txt
        boxes["main_box"].get_paragraph()[boxes["main_box"].current_para_id][boxes["main_box"].current_sent_id][7] = 1

def rephrase_sentence():
    paragraph = boxes["main_box"].get_paragraph()
    tokens = get_sentence(paragraph[boxes["main_box"].current_para_id][boxes["main_box"].current_sent_id][0]
                             ,config.getfloat('MODEL_PARAM','σ')
                             ,config.getfloat('MODEL_PARAM','k')
                             ,config.getint('MODEL_PARAM','batch')
                             )
    sents = []
    for i in torch.arange(len(tokens["input_ids"])):
        sent = str(bert_tokenizer.decode(tokens["input_ids"][i],skip_special_tokens=True))
        sent = sent + "\n"
        sents.append(sent)

    boxes["output_box"].set_text(sents)
    curses.beep()

def save_changes():
    paras = boxes["main_box"].get_paragraph()
    txt = []
    for para in paras:
        sents = ""
        for sent in para:
            sents = sents + sent[0]
        sents = sents + "\n"
        txt.append(sents)
    filename =  "revised_"+config.get("FILE","input_file")
    f = open(filename,'w')
    f.writelines(txt)
    f.flush()
    f.close()

def undo_changes():
    global replaced_sents
    for sent in replaced_sents:
        if sent[6] == boxes["main_box"].current_line_id:
            boxes["main_box"].get_paragraph()[boxes["main_box"].current_para_id][boxes["main_box"].current_sent_id][0] = str(sent[0])
            boxes["main_box"].get_paragraph()[boxes["main_box"].current_para_id][boxes["main_box"].current_sent_id][7] = 0
            replaced_sents.remove(sent)
            break

def move_focus():
    global current_box
    if current_box == "main_box":
        current_box = "tip_box"
    elif current_box == "tip_box":
        current_box = "main_box"

def do_nothing():
    pass

def main(stdscr):
    global config,boxes,screen,current_box,σ,k,batch
    # read config file
    σ = config.getfloat("MODEL_PARAM",'σ')
    k = config.getfloat("MODEL_PARAM",'k')
    batch = config.get("MODEL_PARAM",'batch')
    # set tbe screen
    screen = stdscr
    # curses initialization
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_RED)
    curses.init_pair(2, curses.COLOR_CYAN, -1)
    curses.init_pair(3, curses.COLOR_BLUE, -1)
    curses.init_pair(4, curses.COLOR_GREEN, -1)
    curses.init_pair(5, curses.COLOR_MAGENTA, -1)
    curses.init_pair(6, curses.COLOR_RED, -1)
    curses.curs_set(0)
    # create banner window
    boxes["banner_box"] = TextBox(stdscr, 9
                                  , 170
                                  , 3
                                  , 15
                                  , 0
                                  , "Banner")
    boxes["banner_box"].color_pair = curses.color_pair
    boxes["banner_box"].selected_color = 0
    boxes["banner_box"].left_right_margins = 1
    boxes["banner_box"].top_bottom_margins = 1
    boxes["banner_box"].set_text(read_text(config.get('FILE', 'logo_file')))

    # create content window to display original sentences
    boxes["main_box"] = TextBox(stdscr,config.getint('CONTENT_BOX','row')
                        ,config.getint('CONTENT_BOX','col')
                        ,config.getint('CONTENT_BOX','y')
                        ,config.getint('CONTENT_BOX','x')
                        ,1
                        , "Content")
    boxes["main_box"].color_pair = curses.color_pair
    boxes["main_box"].enter = rephrase_sentence
    boxes["main_box"].alt_r = replace_sentence
    boxes["main_box"].alt_s = save_changes
    boxes["main_box"].u     = undo_changes
    boxes["main_box"].indent = 2
    boxes["main_box"].left_right_margins = 4
    boxes["main_box"].top_bottom_margins = 2
    boxes["main_box"].set_text(read_text(config.get('FILE','input_file')))

    # draw output window to display rephrased sentences
    boxes["output_box"] = TextBox(stdscr,config.getint('OUTPUT_BOX','row')
                        ,config.getint('OUTPUT_BOX','col')
                        ,config.getint('OUTPUT_BOX','y')
                        ,config.getint('OUTPUT_BOX','x')
                        ,2
                        ,"Output")
    boxes["output_box"].color_pair = curses.color_pair
    boxes["output_box"].selected_color = 2
    boxes["output_box"].left_right_margins = 4
    boxes["output_box"].top_bottom_margins = 2
    boxes["output_box"].enter = do_nothing
    boxes["output_box"].alt_r = do_nothing
    boxes["output_box"].alt_s = do_nothing
    boxes["output_box"].u     = do_nothing

    # draw tips window to display help info
    boxes["tip_box"] = TextBox(stdscr,config.getint('TIP_BOX','row')
                        ,config.getint('TIP_BOX','col')
                        ,config.getint('TIP_BOX','y')
                        ,config.getint('TIP_BOX','x')
                        ,3
                        ,"Tips")
    boxes["tip_box"].color_pair = curses.color_pair
    boxes["tip_box"].selected_color = 0
    boxes["tip_box"].left_right_margins = 2
    boxes["tip_box"].top_bottom_margins = 2
    boxes["tip_box"].enter = do_nothing
    boxes["tip_box"].alt_r = do_nothing
    boxes["tip_box"].alt_s = do_nothing
    boxes["tip_box"].u     = do_nothing
    boxes["tip_box"].set_text(read_text(config.get('FILE', 'tips_file')))

    # go to loop
    current_box = "main_box"
    update_status()
    while True:
        c = screen.getch()
        if c == 113:
            break
        elif c == 9:
            move_focus()
        elif c == 10:                   # enter key strike
            boxes[current_box].enter()
        elif c == 174:                  # alt+r, replace original sentences with new one
            boxes[current_box].alt_r()
        elif c == 159:                  # alt+s, save changes into file
            boxes[current_box].alt_s()
        elif c == 117:                  # 'u', undo changes.
            boxes[current_box].u()
        elif c == curses.KEY_UP:
            boxes[current_box].calc_position(-1)
        elif c == curses.KEY_DOWN:
            boxes[current_box].calc_position(1)
        elif c == curses.KEY_LEFT:
            boxes[current_box].calc_position(-1)
        elif c == curses.KEY_RIGHT:
            boxes[current_box].calc_position(1)
        else:
            pass
        update_status()

if __name__=='__main__':
    wrapper(main)

