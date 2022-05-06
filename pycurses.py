import curses, traceback
from curses import wrapper
import spacy
import torch
import numpy as np
import rewriter as rw

# text frame objects
screen = None
txt_box1 = None
revised_box = None
nlp = spacy.load("en_core_web_sm")
# text frame line size and row size and position
frame_width = 100
frame_height = 20
frame_y = 2
frame_x = 5
txt_width = 90
txt_rows = 80
# current sentence position
s_pos = 0
r_pos = 0
offset_x = 4
offset_y = 1
max_line = 0
window_size = 18
# initialize sentences list
paragraph = None
sentences = []

# model parameters
σ = 0.97
k = 0.1



class Textframe(object):
    def __init__(self,width=200,height=20,y=2,x=5):
        self.frame_width = width   # 200 by default
        self.frame_height = height # 20 by default
        self.frame_y = y           # frame coordinates y, 2 by default
        self.frame_x = x           # frame coordinates x, 5 by default
        self.line_size = 90        # line size
        self.row_num = 80          # max row number
        self.select_idx = 0        # selected sentence position
        self.firstrow_idx = 0      # number of first row to print
        self.offset_x = 4          # text margin size to the left and right side of frame
        self.offset_y = 1          # text margin size to the top and bottom of frame
        self.window_size = 18      # shift window size
        self.__win = curses.newwin(frame_height, frame_width, frame_y, frame_x)   # windows object
        self.__win.keypad(True)
        self.__paragraphs = []     # text paragraphs contain sentences
        self.__sentences = []      # text sentences


    def getWin(self):
        return self.__win

    def drawBorder(self,color):
        self.__win.attron(color)
        self.__win.border(0, 0, 0, 0, 0, 0, 0, 0)
        self.__win.attron(curses.A_NORMAL)
        self.__win.refresh()

    def setText(self,texts):
        for i in np.arange(len(texts)):
            p = texts[i]
            p = p[0:len(p) - 1]
            doc = nlp(p)
            sents = []
            j = 0
            for sent in doc.sents:
                sents.append([sent.text, 0, 0, 0, i, j])
                self.__sentences.append([sent.text, 0, 0, 0, i, j])
                j = j + 1
            self.__paragraphs.append(sents)

    def selectSentence(self):
        while True:
            c = self.__win.getch()
            if c == curses.KEY_UP:
                self.firstrow_idx = np.max([(self.firstrow_idx - 1), 0])
                self.printText(0)
            elif c == curses.KEY_DOWN:
                self.firstrow_idx = np.min([(self.firstrow_idx + 1), np.max([self.row_num - self.window_size, 0])])
                self.printText(0)
            elif c == curses.KEY_RIGHT:
                self.printText(1)
            elif c == curses.KEY_LEFT:
                self.printText(-1)
            elif c in [10, 13, curses.KEY_ENTER]:
                revisesentences()
            elif c == 113:
                break  # Exit the while loop
            elif c == curses.KEY_HOME:
                x = 0



    def printText(self,step):
        # update selected sentences position and color
        self.__sentences[self.select_idx][1] = 0
        self.select_idx = self.select_idx + step
        if self.select_idx > len(self.__sentences) - 1:
            self.select_idx = 0
        if self.select_idx < 0:
            self.select_idx = 0
        self.__sentences[self.select_idx][1] = 1
        x = 4
        y = 0
        sents = []
        for para in self.__paragraphs:
            for i in np.arange(len(para)):
                total_size = x + len(para[i][0])
                l = [self.line_size] * int(total_size / self.line_size)
                l.append(total_size % self.line_size)
                str_len = len(para[i][0])
                st = 0

                for j in np.arange(len(l)):
                    et = min((l[j] - x), str_len) + st
                    if j == 0:
                        para[i][2] = y
                        para[i][3] = x
                    sents.append([y, x, para[i][0][st:et], para[i][1]])
                    x = et - st + x if et - st + x < txt_width else 0
                    if x == 0:
                        y = y + 1
                    str_len = str_len - et + st
                    st = et
            y = y + 1
            x = 4

        self.__win.clear()
        #updateframe(0)
        for i in np.arange(len(sents)):
            if self.firstrow_idx <= sents[i][0] and sents[i][0] <= self.firstrow_idx + self.window_size:
                self.__win.addstr(sents[i][0] - self.firstrow_idx + self.offset_y, sents[i][1] + self.offset_x, sents[i][2],
                               curses.color_pair(sents[i][3]))
            self.row_num = sents[i][0]
        self.drawBorder(curses.A_BOLD)
        self.__win.refresh()



'''
readtxt() function 
1. read txt from a txt file.
2. chop the txt into sentences in paragraphs
3. return a list of paragraphs and sentenece like:
                    [
                     [["para1.sent1",y,x,color]["para1.sent2",y,x,color]]
                    ,[["para2.sent1",y,x,color]]
                    ,[["para3.sent1",y,x,color],[para3.sent2",y,x,color],[para3.sent3",y,x,color]]
                    ]
                    
'''
def readtxt(txt):
    paragraph = []
    nlp = spacy.load("en_core_web_sm")
    f = open('sample.txt', mode='r')
    texts = f.readlines()
    # for i in np.arange(len(texts)):
    #     p = texts[i]
    #     p = p[0:len(p) - 1]
    #     doc = nlp(p)
    #     sents = []
    #     j = 0
    #     for sent in doc.sents:
    #         sents.append([sent.text, 0, 0, 0,i,j])
    #         j = j + 1
    #     paragraph.append(sents)
    # return paragraph
    f.close()
    return texts

'''
the main() function 
'''
def main():
    # read texts from file
    global paragraph, sentences
    paragraph = readtxt('sample.txt')
    for p in paragraph:
        for sent in p:
            sentences.append(sent)
    wrapper(drawframe)


'''
txtframe() function defines the frame that can display
the sentences and the functions so that user can move their 
choice among sentences by "UP,DDWN,RIGHT and LEFT" function key
'''
def drawframe(stdscr):
    global r_pos,max_line,window_size,screen,txt_box,revised_box,txt_box1
    # Clear screen
    screen = stdscr
    # initialize
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN, -1)
    curses.init_pair(2, curses.COLOR_WHITE, -1)
    curses.init_pair(3, curses.COLOR_RED, -1)
    curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_GREEN)
    curses.curs_set(0)
    # select sentence
    txt_box = Textframe()
    txt_box.setText(readtxt('sample.txt'))
    txt_box.printText(0)
    # txt_box = curses.newwin(frame_height, frame_width, frame_y, frame_x)
    # txt_box.keypad(True)
    # revised_box = curses.newwin(10, frame_width, frame_y + 24, frame_x)
    # revised_box.keypad(True)

    # movesentences(0)
    txt_box.selectSentence()


'''
revisesentences(win,pos)
'''
def revisesentences():
    revised_box.clear()
    updateframe(1)
    txt = sentences[s_pos][0]
    tokens = rw.rewriter(txt,σ,k)
    sents = []
    color = 0
    for i in torch.arange(len(tokens["input_ids"])):
        sents.append([i+offset_y,offset_x,rw.tokenizer.decode(tokens["input_ids"][i], skip_special_tokens=True)])
        revised_box.addstr(sents[i][0],sents[i][1],sents[i][2],curses.color_pair(color))
    r_pos = 0
    # accept user's input
    while True:
        c = revised_box.getch()
        if c == curses.KEY_UP:
            r_pos = np.max([r_pos - 1,0])
            for i in np.arange(len(sents)):
                color = 1 if i == r_pos else 0
                revised_box.addstr(sents[i][0], sents[i][1], sents[i][2],curses.color_pair(color))
        elif c == curses.KEY_DOWN:
            r_pos = np.min([r_pos + 1,len(sents)-1])
            for i in np.arange(len(sents)):
                color = 1 if i == r_pos else 0
                revised_box.addstr(sents[i][0], sents[i][1], sents[i][2],curses.color_pair(color))
        elif c in [10,13,curses.KEY_ENTER]:
            revised_box.addstr(0, 0, str(c))
        elif c == 113:
            updateframe(0)
            break  # Exit the while loop
        elif c == curses.KEY_HOME:
            x = 0

    revised_box.refresh()
'''
movesentences(win,pos) function 
1. draw the border of the txt frame.
2. iterate the sentences list and 
    calculate the coordinates (y,x) of each portion 
    based on the space constraints.
3. print the sentences in the txt frame.
'''
def movesentences(pos):
    global s_pos, sentences,r_pos,max_line,window_size
    sentences[s_pos][1] = 0
    s_pos = s_pos + pos
    if s_pos > len(sentences) - 1:
        s_pos = 0
    if s_pos < 0:
        s_pos = 0
    sentences[s_pos][1] = 1

    x = 4
    y = 0
    sents = []
    for para in paragraph:
        for i in np.arange(len(para)):
            total_size = x + len(para[i][0])
            l = [txt_width] * int(total_size / txt_width)
            l.append(total_size % txt_width)
            str_len = len(para[i][0])
            st = 0

            for j in np.arange(len(l)):
                et = min((l[j] - x),str_len) + st
                if j == 0:
                    para[i][2] = y
                    para[i][3] = x
                sents.append([y,x,para[i][0][st:et],para[i][1]])
                x = et-st+x if et-st+x < txt_width else 0
                if x == 0:
                    y = y+1
                str_len = str_len - et + st
                st = et
        y = y+1
        x = 4

    txt_box.clear()
    updateframe(0)
    for i in np.arange(len(sents)):
        if r_pos <= sents[i][0] and sents[i][0] <=r_pos+window_size:
            txt_box.addstr(sents[i][0]-r_pos+offset_y,sents[i][1]+offset_x,sents[i][2],curses.color_pair(sents[i][3]))
        max_line = sents[i][0]
    txt_box.refresh()



def updateframe(win):
    if win == 0:
        color1 = 4
        color2 = 0
        border_color1 = curses.A_BOLD
        border_color2 = curses.A_NORMAL
    else:
        color1 = 0
        color2 = 4
        border_color1 = curses.A_NORMAL
        border_color2 = curses.A_BOLD

    screen.addstr(frame_y-1,frame_x,"{}-{} of total {} lines. sentence {} in paragraph {} is selected"
                  .format(r_pos,np.min([r_pos+window_size,max_line]),max_line,sentences[s_pos][5],win)
                  ,curses.color_pair(color1))
    screen.addstr(frame_y + 23, frame_x
                  ,"Three new sentences will be generated: σ={} and k={}".format(σ,k)
                  ,curses.color_pair(color2))
    screen.refresh()

    txt_box.attron(border_color1)
    txt_box.border(0,0,0,0,0,0,0,0)
    txt_box.attron(curses.A_NORMAL)
    txt_box.refresh()
    revised_box.attron(curses.A_NORMAL)
    revised_box.border(0,0,0,0,0,0,0,0)
    revised_box.attron(curses.A_NORMAL)
    revised_box.refresh()




if __name__=='__main__':
    main()

