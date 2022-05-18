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
from curses import wrapper
from configparser import ConfigParser
import torch
import numpy as np
import copy
import os
import generator as gen

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


class Textframe(object):
    def __init__(self,stdscr,row,col,y,x,id,name):
        self.row = row             # 20 by default
        self.col = col             # 100 by default
        self.y = y                 # frame coordinates y, 2 by default
        self.x = x                 # frame coordinates x, 5 by default
        self.left_right_margins = 1          # text margin size to the left and right side of frame
        self.top_bottom_margins = 1          # text margin size to the top and bottom of frame
        self.shift_window_size = self.row - self.top_bottom_margins * 2     # shift window size
        self.line_num = 0          # max line number
        self.current_line_id = 0   # selected sentence position
        self.current_sent_id = 0   # current sentence index
        self.current_para_id = 0   # current sentence index
        self.first_line_id = 0     # number of first line to print
        self.indent = 0            # first sentences indent
        self.paragraph_num = 0     # number of paragraphs
        self.sentence_num = 0      # number of sentences
        self.screen = stdscr       # screen window
        self.quit = 0              # indicator to quit loop
        self.__id = id             # instance id
        self.name = name         # frame name
        self.enter = None          # process enter key strike,
        self.alt_s = None          # process alt+s,
        self.alt_r = None          # process alt+r,
        self.u =None               # process 'u' strike
        self.selected_color = 1    # color of selected sentence
        self.changed_color  = 2    # color of unchanged sentence
        self.unchanged_color = 0   # color of changed sentence
        self.__paragraphs_list = []     # text paragraphs contain sentences
        self.__sentence_list = None     # current sentences
        self.__win = curses.newwin(self.row, self.col, y, x)   # windows object
        self.__win.keypad(True)



    def getWin(self):
        return self.__win

    def get_paragraph(self):
        return self.__paragraphs_list

    def setText(self,texts):
        # reset parameters
        self.__paragraphs_list.clear()
        self.line_num = 0           # max row number
        self.current_line_id = 0        # selected sentence position
        self.current_sent_id = 0  # current sentence index
        self.current_para_id = 0  # current sentence index
        self.first_line_id = 0      # number of first row to print
        self.paragraph_num = 0          # number of paragraphs
        self.sentence_num = 0          # number of sentences

        n = 0
        for i in np.arange(len(texts)):
            p = texts[i]
            p = p[0:len(p) - 1]
            doc = gen.nlp(p)
            sents = []
            j = 0
            for sent in doc.sents:
                sents.append([sent.text, 0, 0, 0, i, j,n,0,0]) # ["txt",color,y,x,para_idx,sent_idx,sent_num,changed,select]
                j = j + 1
                n = n + 1
            self.__paragraphs_list.append(sents)
        self.sentence_num = n

    # def selectSentence(self):
    #     updateStatus()
    #     while True:
    #         c = self.__win.getch()
    #         step = 0
    #         if c == curses.KEY_UP:
    #             step = -1
    #         elif c == curses.KEY_DOWN:
    #             step = 1
    #         elif c == curses.KEY_RIGHT:
    #             step = 1
    #         elif c == curses.KEY_LEFT:
    #             step = -1
    #         elif c == 159:           # alt+s, save changes to main_box
    #             self.alt_s()
    #             if self.__id == 2:
    #                 break
    #         elif c == 117:           # 'u', undo changes.
    #             self.u()
    #         elif c == 10:            # Enter key.
    #             self.enter(self.__id)
    #         elif c == 113:           # 'q' , quit loop
    #             break
    #         elif c == curses.KEY_HOME:
    #             x = 0
    #         self.calcPosition(step)
    #         updateStatus()


    def calcPosition(self,step):

        # update selected sentences position and color
        self.current_line_id = self.current_line_id + step
        if self.current_line_id > self.sentence_num - 1:
            self.current_line_id = self.sentence_num - 1
        if self.current_line_id < 0:
            self.current_line_id = 0
        x = self.indent
        y = 0
        line_size = self.col - self.left_right_margins * 2
        # calculate the sentences to display
        sents = []
        if len(self.__paragraphs_list) > 0:
            for para in self.__paragraphs_list:
                for i in np.arange(len(para)):
                    total_size = x + len(para[i][0])
                    l = [line_size] * int(total_size / line_size)
                    l.append(total_size % line_size)
                    str_len = len(para[i][0])
                    st = 0
                    para[i][8] = (1 if para[i][6] == self.current_line_id else 0)
                    para[i][1] =(self.selected_color if para[i][6] == self.current_line_id else
                                 (self.unchanged_color if para[i][7] == 0 else self.changed_color))
                    self.current_sent_id = (i if para[i][6] == self.current_line_id else self.current_sent_id)
                    self.current_para_id = (self.__paragraphs_list.index(para) if para[i][6] == self.current_line_id else self.current_para_id)

                    for j in np.arange(len(l)):
                        et = min((l[j] - x), str_len) + st
                        if j == 0:
                            para[i][2] = y
                            para[i][3] = x
                        sents.append([y, x, para[i][0][st:et], para[i][1],para[i][8]])
                        x = et - st + x if et - st + x < line_size else 0
                        if x == 0:
                            y = y + 1
                        str_len = str_len - et + st
                        st = et
                y = y + 1
                x = self.indent
        self.__sentence_list = sents

    def printText(self):
        self.__win.clear()
        if self.__sentence_list != None and len(self.__sentence_list) > 0 :
            self.line_num = np.max([sent[0] for sent in self.__sentence_list])
            select_sent_max_y = max([sent[0] if sent[4] == 1 else 0 for sent in self.__sentence_list])
            select_sent_min_y = 0
            for sent in self.__sentence_list:
                if sent[4] == 1:
                    select_sent_min_y = sent[0]
                    break
            if select_sent_max_y - self.first_line_id + self.top_bottom_margins * 2 > self.shift_window_size :
                self.first_line_id = select_sent_max_y - self.shift_window_size + self.top_bottom_margins * 2

            if select_sent_min_y < self.first_line_id:
                self.first_line_id = select_sent_min_y

            for sent in self.__sentence_list:
                if self.first_line_id <= sent[0] and sent[0] <= \
                        self.first_line_id + self.shift_window_size - self.top_bottom_margins * 2:
                    self.__win.addstr(sent[0] + self.top_bottom_margins - self.first_line_id
                                              , sent[1] + self.left_right_margins
                                              , str(sent[2])
                                              ,curses.color_pair(sent[3]))

        self.__win.refresh()

def updateStatus():
    global boxes,current_box
    for k in boxes:
        boxes[k].calcPosition(0)
        boxes[k].printText()
        if k != "banner_box":
            if k == current_box:
                boxes[k].getWin().attron(curses.color_pair(1))
            else:
                boxes[k].getWin().attron(curses.A_NORMAL)
            boxes[k].getWin().box()
            boxes[k].getWin().addstr(0, 4, boxes[k].name)
            boxes[k].getWin().attroff(curses.color_pair(1))
            boxes[k].getWin().attroff(curses.A_NORMAL)
        boxes[k].getWin().refresh()


def readTXT(file):
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
    tokens = gen.getSentence(paragraph[boxes["main_box"].current_para_id][boxes["main_box"].current_sent_id][0]
                             ,config.getfloat('MODEL_PARAM','σ')
                             ,config.getfloat('MODEL_PARAM','k')
                             ,config.getint('MODEL_PARAM','batch')
                             )
    sents = []
    for i in torch.arange(len(tokens["input_ids"])):
        sent = str(gen.tokenizer.decode(tokens["input_ids"][i],skip_special_tokens=True))
        sent = sent + "\n"
        sents.append(sent)

    boxes["output_box"].setText(sents)
    curses.beep()
    #updateStatus()


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
    boxes["banner_box"] = Textframe(stdscr, 9
                                  , 170
                                  , 3
                                  , 15
                                  , 0
                                  , "Banner")
    boxes["banner_box"].selected_color = 0
    boxes["banner_box"].left_right_margins = 1
    boxes["banner_box"].top_bottom_margins = 1
    boxes["banner_box"].setText(readTXT(config.get('FILE', 'logo_file')))

    # create content window to display original sentences
    boxes["main_box"] = Textframe(stdscr,config.getint('CONTENT_BOX','row')
                        ,config.getint('CONTENT_BOX','col')
                        ,config.getint('CONTENT_BOX','y')
                        ,config.getint('CONTENT_BOX','x')
                        ,1
                        , "Content")
    boxes["main_box"].enter = rephrase_sentence
    boxes["main_box"].alt_r = replace_sentence
    boxes["main_box"].alt_s = save_changes
    boxes["main_box"].u     = undo_changes
    boxes["main_box"].indent = 2
    boxes["main_box"].left_right_margins = 4
    boxes["main_box"].top_bottom_margins = 2
    boxes["main_box"].setText(readTXT(config.get('FILE','input_file')))

    # draw output window to display rephrased sentences
    boxes["output_box"] = Textframe(stdscr,config.getint('OUTPUT_BOX','row')
                        ,config.getint('OUTPUT_BOX','col')
                        ,config.getint('OUTPUT_BOX','y')
                        ,config.getint('OUTPUT_BOX','x')
                        ,2
                        ,"Output")
    boxes["output_box"].selected_color = 2
    boxes["output_box"].left_right_margins = 4
    boxes["output_box"].top_bottom_margins = 2
    boxes["output_box"].enter = do_nothing
    boxes["output_box"].alt_r = do_nothing
    boxes["output_box"].alt_s = do_nothing
    boxes["output_box"].u     = do_nothing


    # draw tips window to display help info
    boxes["tip_box"] = Textframe(stdscr,config.getint('TIP_BOX','row')
                        ,config.getint('TIP_BOX','col')
                        ,config.getint('TIP_BOX','y')
                        ,config.getint('TIP_BOX','x')
                        ,3
                        ,"Tips")
    boxes["tip_box"].selected_color = 0
    boxes["tip_box"].left_right_margins = 2
    boxes["tip_box"].top_bottom_margins = 2
    boxes["tip_box"].enter = do_nothing
    boxes["tip_box"].alt_r = do_nothing
    boxes["tip_box"].alt_s = do_nothing
    boxes["tip_box"].u     = do_nothing
    boxes["tip_box"].setText(readTXT(config.get('FILE', 'tips_file')))


    # go to loop
    current_box = "main_box"
    updateStatus()
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
            boxes[current_box].calcPosition(-1)
        elif c == curses.KEY_DOWN:
            boxes[current_box].calcPosition(1)
        elif c == curses.KEY_LEFT:
            boxes[current_box].calcPosition(-1)
        elif c == curses.KEY_RIGHT:
            boxes[current_box].calcPosition(1)
        else:
            pass

        updateStatus()


if __name__=='__main__':
    wrapper(main)

