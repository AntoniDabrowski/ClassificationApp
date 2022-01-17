from tkinter import *
from time import sleep

if __name__ == '__main__':
    root = Tk()
    root.geometry('700x300')
    list_of_algorithms = ["K-nearest neighbors", "Naive Bayes Classifier", "Decision Tree", "Random Forest"]
    Label(root, text="Algorithms:").grid(row=0, sticky=W)

    vars = {}
    for i, name in enumerate(list_of_algorithms):
        vars[name] = IntVar()
        Checkbutton(root, text=name, variable=vars[name]).grid(row=i+1, sticky=W)

    Button(root, text='Start', command=lambda: print([vars[key].get() for key in vars.keys()])).grid(row=len(vars)+1,
                                                                                                    sticky=W, pady=4)

    mainloop()
