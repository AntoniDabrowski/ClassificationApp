from tkinter import *
from tkinter.ttk import Progressbar
from tkinter.filedialog import askopenfilename, askdirectory
from Main.classification import classification


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


def training(vars, train_path, test_path, pred_path, root, pb, txt, mode):
    classification(train_path.get(), test_path.get(), pred_path.get(), {key: val.get() for key, val in vars.items()},
                   root, pb, txt, mode)


if __name__ == '__main__':
    root = Tk()
    root.geometry('650x300')
    list_of_algorithms = ["K-nearest neighbors", "Naive Bayes Classifier", "Decision Tree", "Random Forest"]

    train_dataset_path = StringVar()
    train_dataset_path.set("First column should contain target")
    test_dataset_path = StringVar()
    test_dataset_path.set("Should contain the same columns as train set, except one with targets")
    prediction_path = StringVar()

    Label(root, text="Choose a training dataset:").grid(row=0, column=2, sticky=W)
    Button(root, text='Browse Data Set', command=lambda: import_csv_data('train')).grid(row=0, column=3, sticky=E)
    entry = Entry(root, textvariable=train_dataset_path).grid(row=1, column=2, columnspan=2, sticky='we')

    Label(root, text="Choose a test dataset:").grid(row=2, column=2, sticky=W)
    Button(root, text='Browse Data Set', command=lambda: import_csv_data('test')).grid(row=2, column=3, sticky=E)
    entry = Entry(root, textvariable=test_dataset_path).grid(row=3, column=2, columnspan=2, sticky='we')

    Label(root, text="Choose a place to save predictions:").grid(row=4, column=2, sticky=W)
    Button(root, text='Browse', command=lambda: import_csv_data('preds')).grid(row=4, column=3, sticky=E)
    entry = Entry(root, textvariable=prediction_path).grid(row=5, column=2, columnspan=2, sticky='we')

    text = Text(root, height=5)
    text.grid(row=max(7, len(list_of_algorithms) + 1), column=0, columnspan=4, pady=5, sticky="ne")
    text.config(state="normal")
    t = """This is ClassificationApp. Add your train and test sets in csv format. 
           Choose algorithms to work on them and click start."""
    text.insert(END, t)
    text.config(state="disabled")

    pb = Progressbar(
        root,
        orient=HORIZONTAL,
        length=400,
        mode='determinate'
    )
    pb.grid(row=8, column=0, columnspan=3, pady=0)

    pb_label = Label(
        root,
        text='0%'
    )
    pb_label.grid(row=8, column=3, columnspan=1, pady=0)

    Label(root, text="Algorithms:").grid(row=0, sticky=W)

    var = IntVar()
    R1 = Radiobutton(root, text="Fast", variable=var, value=0)
    R1.grid(row=5, column=0, columnspan=1, pady=0)

    R2 = Radiobutton(root, text="Best performance", variable=var, value=1)
    R2.grid(row=5, column=1, columnspan=1, pady=0)

    vars = {}
    for i, name in enumerate(list_of_algorithms):
        vars[name] = IntVar()
        Checkbutton(root, text=name, variable=vars[name]).grid(row=i + 1, column=0, columnspan=2, sticky="W")

    start = Button(root, text='Start',
                   command=lambda: training(vars, train_dataset_path, test_dataset_path, prediction_path, root, pb,
                                            pb_label, var.get()))
    start.place(rely=1.0, relx=0.5, x=0, y=0, anchor=S)

    close = Button(root, text='Close', command=root.destroy)
    close.place(rely=1.0, relx=1.0, x=0, y=0, anchor=SE)

    mainloop()
