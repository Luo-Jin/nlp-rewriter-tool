# import torch
#
# import stanza
# from transformers import BertTokenizer
# import torchtext.vocab as vocab
# import re
# cache_dir = 'train/GloVe6B5429'
# glove = vocab.GloVe(name='840B', dim=300, cache=cache_dir)
# tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
# # stanza.download('en',model_dir='stanza')       # This downloads the English models for the neural pipeline
# txt  = "Twenty state or territorial democratic parties intend to apply to hold early presidential nominating contests in 2024, " \
#        "a DNC official told CNN Saturday, as the party reevaluates its process of selecting nominees."
# tokens = tokenizer(txt, return_tensors="pt",return_token_type_ids=False,return_attention_mask=False,return_special_tokens_mask=False)
# en_nlp = stanza.Pipeline('en',model_dir='stanza',processors='tokenize,ner') # This sets up a default neural pipeline in English
# #doc = en_nlp()
#
# # for t in tokens["input_ids"][0]:
# #     w = tokenizer.ids_to_tokens[t.item()]
# #     doc=en_nlp(w)
# #     print(doc.entities)
# doc = en_nlp(txt)
# words = doc.to_dict()[0]
# # ids = {'input_ids':torch.tensor([[tokenizer.convert_tokens_to_ids(w['text']) for w in words]])}
# # ids['input_ids'] = torch.cat((torch.tensor([[101]]),ids['input_ids'],torch.tensor([[102]])),dim=-1)
# # print(tokenizer.convert_ids_to_tokens(tokens['input_ids'][0]))
# mask_pos = []
# for i in range(len(tokens["input_ids"][0])):
#         id = tokens["input_ids"][0][i]
#         w = tokenizer.ids_to_tokens[id.item()]
#         re.fullmatch('##[0-9]*', w)  # determine if it is a number
#         doc = en_nlp(w)  # determine if it is an entity
#         if  len(doc.entities) == 0:
#                mask_pos.append(i)
# print(tokenizer.convert_tokens_to_ids("I"))
# print(tokenizer.convert_tokens_to_ids("i"))
#
#
#
# # print(ids)
# # print(tokenizer.decode(ids['input_ids'][0]))
# # print(tokens)
# # print(tokenizer.decode(tokens['input_ids'][0]))
# # print([w['text'] for w in words])
# # print(tokenizer.convert_tokens_to_ids('cnn'.lower()))
import curses
import argparse
import sys
from curses import wrapper

class Window:
    def __init__(self, n_rows, n_cols, row=0, col=0):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.row = row
        self.col = col

    @property
    def bottom(self):
        return self.row + self.n_rows - 1

    def up(self, cursor):
        if cursor.row == self.row - 1 and self.row > 0:
            self.row -= 1

    def down(self, buffer, cursor):
        if cursor.row == self.bottom + 1 and self.bottom < buffer.bottom:
            self.row += 1

    def translate(self, cursor):
        return cursor.row - self.row, cursor.col - self.col

    def horizontal_scroll(self, cursor, left_margin=5, right_margin=2):
        n_pages = cursor.col // (self.n_cols - right_margin)
        self.col = max(n_pages * self.n_cols - right_margin - left_margin, 0)

class Cursor:
    def __init__(self, row=0, col=0, col_hint=None):
        self.row = row
        self._col = col
        self._col_hint = col if col_hint is None else col_hint

    @property
    def col(self):
        return self._col

    @col.setter
    def col(self, col):
        self._col = col
        self._col_hint = col

    def up(self, buffer):
        if self.row > 0:
            self.row -= 1
            self._clamp_col(buffer)

    def down(self, buffer):
        if self.row < buffer.bottom:
            self.row += 1
            self._clamp_col(buffer)

    def _clamp_col(self, buffer):
        self._col = min(self._col_hint, len(buffer[self.row]))

    def left(self, buffer):
        if self.col > 0:
            self.col -= 1
        elif self.row > 0:
            self.row -= 1
            self.col = len(buffer[self.row])

    def right(self, buffer):
        if self.col < len(buffer[self.row]):
            self.col += 1
        elif self.row < buffer.bottom:
            self.row += 1
            self.col = 0



class Buffer:
    def __init__(self, lines):
        self.lines = lines

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        return self.lines[index]

    @property
    def bottom(self):
        return len(self) - 1

    def insert(self, cursor, string):
        row, col = cursor.row, cursor.col
        current = self.lines.pop(row)
        new = current[:col] + string + current[col:]
        self.lines.insert(row, new)

    def split(self, cursor):
        row, col = cursor.row, cursor.col
        current = self.lines.pop(row)
        self.lines.insert(row, current[:col])
        self.lines.insert(row + 1, current[col:])

    def delete(self, cursor):
        row, col = cursor.row, cursor.col
        if (row, col) < (self.bottom, len(self[row])):
            current = self.lines.pop(row)
            if col < len(self[row]):
                new = current[:col] + current[col + 1:]
                self.lines.insert(row, new)
            else:
                next = self.lines.pop(row)
                new = current + next
                self.lines.insert(row, new)

def right(window, buffer, cursor):
    cursor.right(buffer)
    window.down(buffer, cursor)
    window.horizontal_scroll(cursor)

def left(window, buffer, cursor):
    cursor.left(buffer)
    window.up(cursor)
    window.horizontal_scroll(cursor)

def main(stdscr):
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()
    window = Window(20, 90)
    cursor = Cursor()

    with open(args.filename) as f:
        buffer = Buffer(f.read().splitlines())

    while True:
        stdscr.erase()
        for row, line in enumerate(buffer[window.row:window.row + window.n_rows]):
            if row == cursor.row - window.row and window.col > 0:
                line = "«" + line[window.col + 1:]
            if len(line) > window.n_cols:
                line = line[:window.n_cols - 1] + "»"
            stdscr.addstr(row, 0, line)
        stdscr.move(*window.translate(cursor))


        k = stdscr.getkey()
        if k == "q":
            sys.exit(0)
        elif k == "KEY_UP":
            cursor.up(buffer)
            window.up(cursor)
            window.horizontal_scroll(cursor)
        elif k == "KEY_DOWN":
            cursor.down(buffer)
            window.down(buffer, cursor)
            window.horizontal_scroll(cursor)
        elif k == "KEY_LEFT":
            cursor.left(buffer)
            window.up(cursor)
            window.horizontal_scroll(cursor)
        elif k == "KEY_RIGHT":
            right(window, buffer, cursor)
        elif k == "\n":
            buffer.split(cursor)
            right(window, buffer, cursor)
        elif k in ("KEY_DELETE", "\x04"):
            buffer.delete(cursor)
        elif k in ("KEY_BACKSPACE", "\x7f"):
            if (cursor.row, cursor.col) > (0, 0):
                left(window, buffer, cursor)
                buffer.delete(cursor)
        else:
            buffer.insert(cursor, k)
            for _ in k:
                right(window, buffer, cursor)

if __name__ == '__main__':
    wrapper(main)