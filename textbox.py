# -*- coding: utf-8 -*-
###############################################
#   File    : Textframe.py
#   Author  : Jin luo
#   Date    : 2022-05-18
#   Input   : None
#   Output  : None
###############################################


import numpy as np
from embeddings import spacy_model

class TextBox(object):
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
        self.__win = stdscr.subwin(self.row, self.col, y, x)   # windows object
        self.__win.keypad(True)
        self.color_pair = None

    def get_win(self):
        return self.__win

    def get_paragraph(self):
        return self.__paragraphs_list

    def set_text(self,texts):
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
            doc = spacy_model(p)
            sents = []
            j = 0
            for sent in doc.sents:
                # ["txt",color,y,x,para_idx,sent_idx,sent_num,changed,select]
                sents.append([sent.text, 0, 0, 0, i, j,n,0,0])
                j = j + 1
                n = n + 1
            self.__paragraphs_list.append(sents)
        self.sentence_num = n

    def calc_position(self,step):
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

    def print_text(self):
        self.__win.clear()
        if self.__sentence_list != None and len(self.__sentence_list) > 0:
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
                                              ,self.color_pair(sent[3]))
        self.__win.refresh()
