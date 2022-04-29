
import time
from threading import Thread
import sys

class ProgressThread(Thread):
    def __init__(self):
        super(ProgressThread, self).__init__()
        self.is_stop = False
        self.cursor_index = 0
        self.cursor_str = '|/-\\'
        self.now = None
        self.info = ""
    def set_progress_info(self, info):
        self.info = info
    def get_progress_text(self):
        cursor = self.cursor_str[self.cursor_index]
        self.cursor_index = self.cursor_index + 1
        if self.cursor_index == len(self.cursor_str):
            self.cursor_index = 0
            time_second = str(int(time.time() - self.now))
            progress_text = self.info + " " + time_second + "s " + cursor
        return progress_text

    def stop_progress(self):
        self.is_stop = True
        time.sleep(0.6)
    def run(self):
        self.now = time.time()
        while True:
            if not self.is_stop:
                progress_text = self.get_progress_text()
                sys.stdout.write(progress_text)
                sys.stdout.flush()
                time.sleep(0.4)
                sys.stdout.write('\r')
            else:
                return

class Progress:
    def __init__(self):
        self.current_thread = None
        pass

    def start_progress(self, progress_info):
        if self.current_thread is not None:
            self.current_thread.stop_progress()
            self.current_thread = ProgressThread()
            self.current_thread.daemon = True
            self.current_thread.set_progress_info(progress_info)
            self.current_thread.start()

    def stop_progress(self):
        if self.current_thread is not None:
            self.current_thread.stop_progress()
            self.current_thread = None

    def show_progress(self, info):
        self.current_thread.set_progress_info(info)


if __name__ == '__main__':
    progress = Progress()
    progress.start_progress("开始上传文件")
    for i in range(10):
        time.sleep(0.5)
    progress.show_progress("文件上传中")
    progress.stop_progress()
