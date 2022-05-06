import curses, traceback
from curses import wrapper
import spacy
import torch
import numpy as np

# text fame line size and row size and position
frame_width = 100
frame_height = 20
frame_y = 2
frame_x = 5
txt_width = 90
txt_rows = 80
# current sentence position
s_pos = 0
r_pos = 0
max_line = 0
window_size = 18
# initialize sentences list
paragraph = None
sentences = []

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
    for i in np.arange(len(texts)):
        p = texts[i]
        p = p[0:len(p) - 1]
        doc = nlp(p)
        sents = []
        j = 0
        for sent in doc.sents:
            sents.append([sent.text, 0, 0, 0,i,j])
            j = j + 1
        paragraph.append(sents)
    return paragraph
    f.close()

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
    global r_pos,max_line,window_size
    # Clear screen
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN, -1)
    curses.init_pair(2, curses.COLOR_WHITE, -1)
    curses.init_pair(3, curses.COLOR_RED, -1)
    curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_GREEN)
    curses.curs_set(0)
    # add test box
    txt_box = curses.newwin(frame_height,frame_width,frame_y,frame_x)
    txt_box.keypad(True)
    movesentences(txt_box,stdscr, 0)
    # add text box for revised sentences
    revised_box = curses.newwin(10,frame_width,frame_y+21,frame_x)
    revised_box.keypad(True)

    # accept user's input
    while True:
        c = txt_box.getkey()
        if c == 'KEY_UP':
            r_pos = np.max([(r_pos -1),0])
            movesentences(txt_box,stdscr, 0)
        elif c == 'KEY_DOWN':
            r_pos = np.min([(r_pos + 1),np.max([max_line-window_size,0])])
            movesentences(txt_box,stdscr, 0)
        elif c == 'KEY_RIGHT':
            movesentences(txt_box,stdscr,1)
            revisesentences(revised_box, 0)
        elif c == 'KEY_LEFT':
            movesentences(txt_box,stdscr,-1)
            revisesentences(revised_box, 0)
        elif c == 'q':
            break  # Exit the while loop
        elif c == curses.KEY_HOME:
            x = 0
'''
revisesentences(win,pos)
'''
def revisesentences(revised_box,pos):
    revised_box.box()
    revised_box.refresh()
'''
movesentences(win,pos) function 
1. draw the border of the txt frame.
2. iterate the sentences list and 
    calculate the coordinates (y,x) of each portion 
    based on the space constraints.
3. print the sentences in the txt frame.
'''
def movesentences(txt_box,stdscr,pos):
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
    offset_x = 4
    offset_y = 1
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
    # add status bar

    stdscr.addstr(1,5,"{}-{} of total {} lines. sentence {} in paragraph {} is selected"
                  .format(r_pos,np.min([r_pos+window_size,max_line]),max_line,sentences[s_pos][5],sentences[s_pos][4])
                  ,curses.color_pair(4))
    stdscr.refresh()
    # print sentences
    txt_box.clear()
    txt_box.border()
    for i in np.arange(len(sents)):
        if r_pos <= sents[i][0] and sents[i][0] <=r_pos+window_size:
            txt_box.addstr(sents[i][0]-r_pos+offset_y,sents[i][1]+offset_x,sents[i][2],curses.color_pair(sents[i][3]))
        max_line = sents[i][0]


if __name__=='__main__':
    main()

