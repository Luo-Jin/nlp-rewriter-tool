###############################################
#   File    : rewriter.py
#   Author  : Jin luo
#   Date    : 2022-04-29
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
file_path = os.path.join(os.path.abspath("."), "rewriter.ini")
config = ConfigParser()
config.read(file_path)
# text frame objects
screen = None
txt_box = None
revised_box = None
# initialize sentences list
replaced_sents = []
# model parameters
σ = 0
k = 0
batch = 1

class Textframe(object):
    def __init__(self,stdscr,line,col,y,x,id):
        self.frame_line = line     # 20 by default
        self.frame_col = col       # 100 by default
        self.frame_y = y           # frame coordinates y, 2 by default
        self.frame_x = x           # frame coordinates x, 5 by default
        self.text_line = 90        # line size
        self.row_num = 0           # max row number
        self.select_idx = 0        # selected sentence position
        self.current_sent_idx = 0  # current sentence index
        self.current_para_idx = 0  # current sentence index
        self.current_sentences = None # current sentences
        self.firstrow_idx = 0      # number of first row to print
        self.offset_x = 4          # text margin size to the left and left side of frame
        self.offset_y = 2          # text margin size to the top and bottom of frame
        self.indent = 4            # first sentences indent
        self.window_size = self.frame_line - self.offset_y * 2     # shift window size
        self.para_num = 0          # number of paragraphs
        self.sent_num = 0          # number of sentences
        self.screen = stdscr       # father window
        self.quit = 0             # indicator to quit loop
        self.__win = curses.newwin(self.frame_line, self.frame_col, y, x)   # windows object
        self.__win.keypad(True)
        self.__paragraphs = []     # text paragraphs contain sentences
        self.__sentences = []      # text sentences
        self.__id = id             # instance id
        self.enterKey = None       # process enter key strike,
        self.alt_s = None          # process alt+s,
        self.u =None               # process 'u' strike
        self.select_sent_y = 0

    def getWin(self):
        return self.__win

    def getParagragh(self):
        return self.__paragraphs

    def setText(self,texts):
        # reset parameters
        self.__paragraphs.clear()
        self.row_num = 0           # max row number
        self.select_idx = 0        # selected sentence position
        self.current_sent_idx = 0  # current sentence index
        self.current_para_idx = 0  # current sentence index
        self.firstrow_idx = 0      # number of first row to print
        self.para_num = 0          # number of paragraphs
        self.sent_num = 0          # number of sentences

        n = 0
        for i in np.arange(len(texts)):
            p = texts[i]
            p = p[0:len(p) - 1]
            doc = gen.nlp(p)
            sents = []
            j = 0
            for sent in doc.sents:
                sents.append([sent.text, 0, 0, 0, i, j,n,0]) # ["txt",color,y,x,para_idx,sent_idx,sent_num,changed]
                j = j + 1
                n = n + 1
            self.__paragraphs.append(sents)
        self.sent_num = n

    def selectSentence(self):
        while True:
            c = self.__win.getch()
            if c == curses.KEY_UP:
                self.calcPosition(-1)
            elif c == curses.KEY_DOWN:
                self.calcPosition(1)
            elif c == curses.KEY_RIGHT:
                self.calcPosition(1)
            elif c == curses.KEY_LEFT:
                self.calcPosition(-1)
            elif c == 159:           # alt+s, save changes to txt_box
                self.alt_s()
                if self.__id == 1:
                    break
            elif c == 117:           # 'u', undo changes.
                self.u()
            elif c == 10:            # Enter key.
                self.enterKey(self.__id)
            elif c == 113:           # 'q' , quit loop
                break
            elif c == curses.KEY_HOME:
                x = 0
            self.calcPosition(0)
            self.printText()

    def calcPosition(self,step):

        # update selected sentences position and color
        self.select_idx = self.select_idx + step
        if self.select_idx > self.sent_num - 1:
            self.select_idx = self.sent_num - 1
        if self.select_idx < 0:
            self.select_idx = 0
        x = self.indent
        y = 0
        # calculate the sentences to display
        sents = []
        for para in self.__paragraphs:
            for i in np.arange(len(para)):
                total_size = x + len(para[i][0])
                l = [self.text_line] * int(total_size / self.text_line)
                l.append(total_size % self.text_line)
                str_len = len(para[i][0])
                st = 0
                para[i][1] =(1 if para[i][6] == self.select_idx else (0 if para[i][7] == 0 else 2))
                self.current_sent_idx = (i if para[i][6] == self.select_idx else self.current_sent_idx)
                self.current_para_idx = (self.__paragraphs.index(para) if para[i][6] == self.select_idx else self.current_para_idx)

                for j in np.arange(len(l)):
                    et = min((l[j] - x), str_len) + st
                    if j == 0:
                        para[i][2] = y
                        para[i][3] = x
                    sents.append([y, x, para[i][0][st:et], para[i][1]])
                    x = et - st + x if et - st + x < self.text_line else 0
                    if x == 0:
                        y = y + 1
                    str_len = str_len - et + st
                    st = et
            y = y + 1
            x = self.indent
        self.current_sentences = sents

    def printText(self):
        self.__win.clear()
        self.row_num = np.max([sent[0] for sent in self.current_sentences])
        select_sent_max_y = max([sent[0] if sent[3] == 1 else 0 for sent in self.current_sentences])
        select_sent_min_y = 0
        for sent in self.current_sentences:
            if sent[3] == 1:
                select_sent_min_y = sent[0]
                break

        if select_sent_max_y - self.firstrow_idx + 1 > self.window_size :
            self.firstrow_idx = select_sent_max_y - self.window_size +1

        if select_sent_min_y < self.firstrow_idx:
            self.firstrow_idx = select_sent_min_y

        for sent in self.current_sentences:
            if self.firstrow_idx <= sent[0] and sent[0] <= self.firstrow_idx + self.window_size -1:
                self.__win.addstr(sent[0] + self.offset_y - self.firstrow_idx
                                          , sent[1] + self.offset_x
                                          , str(sent[2])
                                          ,curses.color_pair(sent[3]))

        self.updateStatus()
        self.__win.refresh()

    def updateStatus(self):
        y   = txt_box.frame_y -1
        y1  = revised_box.frame_y - 1
        x   = txt_box.frame_x
        x1  = revised_box.frame_x

        if self.__id == 0:
            color1 = 3
            color2 = 0
            border_color1 = curses.A_STANDOUT
            border_color2 = curses.A_NORMAL

        else:
            color1 = 0
            color2 = 3
            border_color1 = curses.A_NORMAL
            border_color2 = curses.A_STANDOUT
        screen.clear()
        screen.addstr(y , x,
                           "{}-{} of total {} lines. sentence {} in paragraph {} is selected"
                           .format(txt_box.firstrow_idx, np.min([txt_box.firstrow_idx + txt_box.window_size, txt_box.row_num])
                                   , txt_box.row_num, txt_box.current_sent_idx, txt_box.current_para_idx)
                           , curses.color_pair(color1))

        screen.addstr(y1,x1
                           , "{} new sentence(s) will be generated here with σ={} and k={}".format(batch,σ, k)
                           , curses.color_pair(color2))

        screen.refresh()
        txt_box.getWin().attron(border_color1)
        txt_box.getWin().box()
        txt_box.getWin().attroff(border_color1)
        txt_box.getWin().refresh()
        revised_box.getWin().attron(border_color2)
        revised_box.getWin().box()
        revised_box.getWin().attroff(border_color2)
        revised_box.getWin().refresh()

def readTXT(file):
    f = open(file, mode='r')
    texts = f.readlines()
    f.close()
    return texts

def conformChange():
    sent = revised_box.getParagragh()[revised_box.current_para_idx][revised_box.current_sent_idx]
    txt = str(sent[0])
    replaced_sent = copy.deepcopy(txt_box.getParagragh()[txt_box.current_para_idx][txt_box.current_sent_idx])
    replaced_sents.append(replaced_sent)
    txt_box.getParagragh()[txt_box.current_para_idx][txt_box.current_sent_idx][0] = txt
    txt_box.getParagragh()[txt_box.current_para_idx][txt_box.current_sent_idx][7] = 1
    #revised_box.quit = 1

def refineSentence(id):
    paragraph = txt_box.getParagragh()
    tokens = gen.getSentence(paragraph[txt_box.current_para_idx][txt_box.current_sent_idx][0], σ, k)
    sents = []
    for i in torch.arange(len(tokens["input_ids"])):
        sent = str(gen.tokenizer.decode(tokens["input_ids"][i],skip_special_tokens=True))
        sent = sent + "\n"
        sents.append(sent)

    revised_box.setText(sents)
    revised_box.calcPosition(0)
    revised_box.printText()
    curses.beep()
    if id == 0:
        revised_box.selectSentence()

def saveChange():
    paras = txt_box.getParagragh()
    txt = []
    for para in paras:
        sents = ""
        for sent in para:
            sents = sents + sent[0]
        sents = sents + "\n"
        txt.append(sents)
    filename =  "revised_"+config.get("TEXT","input_file")
    f = open(filename,'w')
    f.writelines(txt)
    f.flush()
    f.close()

def undoChange():
    for sent in replaced_sents:
        if sent[6] == txt_box.select_idx:
            txt_box.getParagragh()[txt_box.current_para_idx][txt_box.current_sent_idx][0] = str(sent[0])
            txt_box.getParagragh()[txt_box.current_para_idx][txt_box.current_sent_idx][7] = 0
            replaced_sents.remove(sent)
            break

def main(stdscr):
    global config,screen,txt_box,revised_box,σ,k,batch
    # read config file
    σ = config.getfloat("MODEL_PARAM",'σ')
    k = config.getfloat("MODEL_PARAM",'k')
    batch = config.get("MODEL_PARAM",'batch')
    # set tbe screen
    screen = stdscr
    # curses initialization
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_RED)
    curses.init_pair(2, curses.COLOR_CYAN, -1)
    curses.init_pair(3, curses.COLOR_RED, -1)
    curses.init_pair(4, curses.COLOR_BLUE, -1)
    curses.curs_set(0)
    # create text frame to display original sentences
    txt_box = Textframe(stdscr,config.getint('GUI','txtbox_line')
                        ,config.getint('GUI','txtbox_col')
                        ,config.getint('GUI','txtbox_y')
                        ,config.getint('GUI','txtbox_x')
                        ,config.getint('GUI','txtbox_id'))
    txt_box.enterKey = refineSentence
    txt_box.alt_s    = saveChange
    txt_box.u        = undoChange
    # create text frame to display revised sentences width=200,height=10,y=26,x=5
    revised_box = Textframe(stdscr,config.getint('GUI','revisedbox_line')
                        ,config.getint('GUI','revisedbox_col')
                        ,config.getint('GUI','revisedbox_y')
                        ,config.getint('GUI','revisedbox_x')
                        ,config.getint('GUI','revisedbox_id'))
    revised_box.indent = 0
    revised_box.enterKey = refineSentence
    revised_box.alt_s = conformChange
    txt_box.setText(readTXT(config.get('TEXT','input_file')))
    txt_box.calcPosition(0)
    txt_box.printText()
    # revised_box.register(fun2)
    txt_box.selectSentence()

if __name__=='__main__':
    wrapper(main)

