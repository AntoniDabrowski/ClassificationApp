from tkinter import *
from tkinter.ttk import Progressbar
from tkinter.filedialog import askopenfilename, askdirectory
from utils.classification import classification


def import_csv_data(name):
    if name == 'train':
        global train_dataset_path
        path = askopenfilename()
        train_dataset_path.set(path)
    elif name == 'test':
        global test_dataset_path
        path = askopenfilename()
        test_dataset_path.set(path)
    elif name == 'preds':
        global prediction_path
        path = askdirectory()
        prediction_path.set(path)


def training(vars, train_path, test_path, pred_path, root, pb, txt, mode, target):
    classification(train_path.get(), test_path.get(), pred_path.get(), {key: val.get() for key, val in vars.items()},
                   root, pb, txt, mode, target)


""" 
    Main file for my application. It is responsible for communication with user
"""

# I am creating a tkinter frame
root = Tk()
root.geometry('650x300')
list_of_algorithms = ["K-nearest neighbors", "Decision Tree", "Random Forest", "Naive Bayes Classifier"]

train_dataset_path = StringVar()
train_dataset_path.set("Location of training set")
test_dataset_path = StringVar()
test_dataset_path.set("Should contain the same columns as train set, except one with targets")
prediction_path = StringVar()
prediction_path.set("Location of destination folder")

target = StringVar()
target.set("number / name")

# Entry for train set
Label(root, text="Choose a training dataset:").grid(row=0, column=2, sticky=W)
Button(root, text='Browse Data Set', command=lambda: import_csv_data('train')).grid(row=0, column=3, sticky=E)
entry = Entry(root, textvariable=train_dataset_path).grid(row=1, column=2, columnspan=2, sticky='we')

# Entry for test set
Label(root, text="Choose a test dataset:").grid(row=2, column=2, sticky=W)
Button(root, text='Browse Data Set', command=lambda: import_csv_data('test')).grid(row=2, column=3, sticky=E)
entry = Entry(root, textvariable=test_dataset_path).grid(row=3, column=2, columnspan=2, sticky='we')

# Entry for destination file
Label(root, text="Choose a place to save predictions:").grid(row=4, column=2, sticky=W)
Button(root, text='Browse', command=lambda: import_csv_data('preds')).grid(row=4, column=3, sticky=E)
entry = Entry(root, textvariable=prediction_path).grid(row=5, column=2, columnspan=2, sticky='we')

# Main communicat
text = Text(root, height=5)
text.grid(row=max(7, len(list_of_algorithms) + 1), column=0, columnspan=4, pady=5, sticky="ne")
text.config(state="normal")
t = """This is ClassificationApp. Add your train and test sets in csv format. 
       Choose algorithms to work on them and click start."""
text.insert(END, t)
text.config(state="disabled")

# Initialization of a progerss bar
pb = Progressbar(
    root,
    orient=HORIZONTAL,
    length=480,
    mode='determinate'
)
pb.grid(row=8, column=0, columnspan=3, pady=0, sticky=W)

pb_label = Label(
    root,
    text='0%'
)
pb_label.grid(row=8, column=3, columnspan=1, pady=0, sticky=E)

Label(root, text="Algorithms:").grid(row=0, column=0, sticky=W)
Label(root, text="Target column:").grid(row=0, column=1)
entry = Entry(root, textvariable=target).grid(row=1, column=1, sticky='E')

# Options Fast / Best performance
""" 
    Difference between those two modes is in how many options are checked. In general I can't
    tell which hyperparameters are the best for algorithm. I have to check various combinations
    to see which behave the best. Moreover in performance mode cross validation stage is done
    more carefully - with more folds. This gives higher confidence level in choosing a classifier.
"""
var = IntVar()
R1 = Radiobutton(root, text="Fast", variable=var, value=0)
R1.grid(row=2, column=1, pady=0, sticky='w')

R2 = Radiobutton(root, text="Best performance", variable=var, value=1)
R2.grid(row=3, column=1, pady=0, sticky='w')

# Checkboxes for each algorithm
vars = {}
for i, name in enumerate(list_of_algorithms):
    vars[name] = IntVar()
    Checkbutton(root, text=name, variable=vars[name]).grid(row=i + 1, column=0, columnspan=2, sticky="W")

start = Button(root, text='Start',
               command=lambda: training(vars, train_dataset_path, test_dataset_path, prediction_path, root, pb,
                                        pb_label, var.get(), target.get()))
start.place(rely=1.0, relx=0.5, x=0, y=0, anchor=S)

close = Button(root, text='Close', command=root.destroy)
close.place(rely=1.0, relx=1.0, x=0, y=0, anchor=SE)

mainloop()
