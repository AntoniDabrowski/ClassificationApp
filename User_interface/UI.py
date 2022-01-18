from tkinter import *
from tkinter.filedialog import askopenfilename
from time import sleep

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
        path = askopenfilename()
        prediction_path.set(path)

def training(vars,train_path,test_path,pred_path):
    for key, val in vars.items():
        print(key,val.get())
    print(train_path.get())
    print(test_path.get())
    print(pred_path.get())

if __name__ == '__main__':
    root = Tk()
    root.geometry('650x300')
    list_of_algorithms = ["K-nearest neighbors", "Naive Bayes Classifier", "Decision Tree", "Random Forest"]

    train_dataset_path = StringVar()
    train_dataset_path.set("First column should contain target")
    test_dataset_path = StringVar()
    test_dataset_path.set("Should contain the same columns as train set, except one with targets")
    prediction_path = StringVar()

    Label(root, text="Choose a training dataset:").grid(row=0, column=1, sticky=W)
    Button(root, text='Browse Data Set',command=lambda: import_csv_data('train')).grid(row=0, column=2, sticky=E)
    entry = Entry(root, textvariable=train_dataset_path).grid(row=1, column=1,columnspan=2,sticky='we')

    Label(root, text="Choose a test dataset:").grid(row=2, column=1, sticky=W)
    Button(root, text='Browse Data Set',command=lambda: import_csv_data('test')).grid(row=2, column=2, sticky=E)
    entry = Entry(root, textvariable=test_dataset_path).grid(row=3, column=1,columnspan=2,sticky='we')

    Label(root, text="Choose a place to save predictions:").grid(row=4, column=1, sticky=W)
    Button(root, text='Browse',command=lambda: import_csv_data('preds')).grid(row=4, column=2, sticky=E)
    entry = Entry(root, textvariable=prediction_path).grid(row=5, column=1,columnspan=2,sticky='we')

    Label(root, text="Algorithms:").grid(row=0, sticky=W)

    vars = {}
    for i, name in enumerate(list_of_algorithms):
        vars[name] = IntVar()
        Checkbutton(root, text=name, variable=vars[name]).grid(row=i+1, column=0,sticky="W")

    start = Button(root, text='Start', command=lambda: training(vars,train_dataset_path,test_dataset_path,prediction_path))
    start.place(rely=1.0, relx=0.5, x=0, y=0, anchor=S)

    close = Button(root, text='Close',command=root.destroy)
    close.place(rely=1.0, relx=1.0, x=0, y=0, anchor=SE)

    text = Text(root,height=5)
    text.grid(row=max(7,len(list_of_algorithms)+1), column=0,columnspan=3,pady=5,sticky="ne")
    text.config(state="normal")
    t = """This is ClassificationApp. Add your train and test sets in csv format. 
           Choose algorithms to work on them and click start."""
    text.insert(END, t)
    text.config(state="disabled")

    mainloop()
