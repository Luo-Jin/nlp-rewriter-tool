from tkinter import *

window = Tk()
label = Label(
    text="Hello, Tkinter",
    foreground="white",  # Set the text color to white
    background="black",  # Set the background color to black
    width=10,
    height=10
)
button = Button(
    text="Click me!",
    width=25,
    height=5,
    bg="blue",
    fg="red",
)
#label.pack()
label = Label(text="Name")
entry = Entry()

label.pack()
button.pack()
entry.pack()




