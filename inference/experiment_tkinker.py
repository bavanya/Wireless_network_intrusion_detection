from tkinter import *
from helper_functions.prepare_data_for_model2 import prepare_data_for_model2
from helper_functions.load_data import load_data
from sklearn.metrics import classification_report,confusion_matrix

import sys, signal, os
import subprocess
import threading
from time import sleep
import pickle
import pandas as pd
import numpy as np
import warnings
import signal


warnings.filterwarnings("ignore")

process_id = -1

def signal_handler(sig, frame):
    print("\n(!) CTRL-C Pressed")
    print(process_id)
    os.kill(process_id, signal.SIGTERM)
    sys.exit(0)

def clicked():

    top= Toplevel(r)
    top.geometry("750x250")
    top.title("Child Window")
    Label(top, text= "100 Packets successfully captured!", font=('Mistral 18 bold')).place(x=150,y=80)    


    f = open('data_test/test.csv', "w")
    proc = subprocess.Popen(['tshark', '-c', '100', '-i', 'wlan0mon', '-T', 'fields', '-e', 'frame.encap_type', '-e', 'frame.len', '-e', 'frame.number', 
    '-e', 'frame.time_delta', '-e', 'frame.time_delta_displayed', '-e', 'frame.time_epoch', '-e', 'frame.time_relative', '-e', 'radiotap.channel.freq', 
    '-e', 'radiotap.length', '-e', 'wlan.duration', '-e', 'wlan.fc.ds', '-e', 'wlan.fc.frag', '-e', 'wlan.fc.order', '-e', 'wlan.fc.moredata', '-e', 
    'wlan.fc.protected', '-e', 'wlan.fc.pwrmgt', '-e', 'wlan.fc.type', '-e', 'wlan.fc.retry', '-e', 'wlan.fc.subtype', '-e', 'wlan.ra','-E' ,'header=y',
    '-E' ,'separator=* '], stdout = f)

    process_id = proc.pid

    random_forest_model = pickle.load(open('../intrusion_detector/rev_random_forest_model_demo.sav','rb'))

    l = ['Botnet','Deauth','Evil_Twin','Normal','SQL_Injection','Website_spoofing']
    l = np.array(l)

    n = 1

    while(n <= 100):
        df = load_data('data_test/deauth_packets.csv', n, 1)
        if len(df) == 0: continue

        df = prepare_data_for_model2(df)
        
        inference_data = df.to_numpy()

        predictions = random_forest_model.predict(inference_data)
        
        l = ['Botnet','Deauth','Evil_Twin','Normal','SQL_Injection','Website_spoofing']
        l = np.array(l)

        for x in predictions:
            for y in range(len(x)):
                if(x[y]==1):
                    t = "Label for packet " + str(n) + " is " + str(l[y])
                    listbox.insert(END, t)
                    listbox.yview(END) 
                    break   
            n +=1

def demo():

    top= Toplevel(r)
    top.geometry("750x250")
    top.title("Child Window")
    Label(top, text= "Inference successful!", font=('Mistral 18 bold')).place(x=150,y=80)    

    m = pickle.load(open('../intrusion_detector/rev_random_forest_model.sav','rb'))

    xten = pd.read_pickle('data_test/demo_x.pkl')
    yten = pd.read_pickle('data_test/demo_y.pkl')

    Labels = ['Label_Botnet','Label_Deauth','Label_Evil_Twin','Label_Normal','Label_SQL_Injection','Label_Website_spoofing']

    for i in range(len(Labels)):
        print("No of entries belonging to label: " + str(i) + " are: " + str(yten[Labels[i]].sum()))


    predictions=m.predict(xten)
    l = ['Botnet','Deauth','Evil_Twin','Normal','SQL_Injection','Website_spoofing']
    l = np.array(l)

    n=1
    for x in predictions:
        for y in range(len(x)):
            if(x[y]==1):
                t = "Label for packet " + str(n) + " is " + str(l[y])
                listbox.insert(END, t)
                listbox.yview(END) 
                break 
        n+=1  

    print(classification_report(yten,predictions))
    ytt=yten.to_numpy()
    #ptt=pred.to_numpy()s
    print(confusion_matrix(ytt.argmax(axis=1),predictions.argmax(axis=1)))



def close():
    signal.signal(signal.SIGINT, signal_handler)
    r.destroy()
    

r = Tk()
r.title('Counting Seconds')
r.geometry('1050x900')
r.config(bg = "#464342")
r.resizable(0, 0)

listbox = Listbox(r, height = 40, 
                    width = 40, 
                    bg = "grey",
                    activestyle = 'dotbox', 
                    font = "Helvetica",
                    fg="white")  
scrollbar = Scrollbar(r)

listbox.config(yscrollcommand = scrollbar.set)
scrollbar.config(command = listbox.yview)
listbox.grid(column=200, row=350)


btn = Button(r, text = "Start Intrusion Detector", width=25, fg = "blue", command=clicked)
btn.grid(column=400, row=350)

btn = Button(r, text = "Try Demo", width=25, fg = "blue", command=demo)
btn.grid(column=500, row=350)

button = Button(r, text='Close', fg = "red", width=25, command=close)
button.grid(column=600, row=350)

r.mainloop()


