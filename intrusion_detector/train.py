from sklearn.metrics import classification_report,confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
import pickle
import warnings
warnings.filterwarnings("ignore")

clf1 = RandomForestClassifier()

xn = pd.read_pickle('filtered_x.pkl')
yn = pd.read_pickle('filtered_y.pkl')

Labels = ['Label_Botnet','Label_Deauth','Label_Evil_Twin','Label_Normal','Label_SQL_Injection','Label_Website_spoofing']

for i in range(len(Labels)):
    print("No of entries belonging to label: " + str(i) + " are: " + str(yn[Labels[i]].sum()))


xtrn,xten,ytrn,yten=train_test_split(xn,yn,test_size=0.35,random_state=69)

scores = cross_val_score(clf1, xtrn, ytrn, cv=3, scoring='accuracy')
print("Accuracy: %.2f (+/- %.2f) [%s]" %(scores.mean(), scores.std(), 'Random Forest'))
m = OneVsRestClassifier(clf1)
m.fit(xtrn, ytrn)
pred=m.predict(xten)
print(classification_report(yten,pred))
ytt=yten.to_numpy()
#ptt=pred.to_numpy()
print(confusion_matrix(ytt.argmax(axis=1),pred.argmax(axis=1)))


filename = 'rev_random_forest_model.sav'
    
#pickle.dump(m, open(filename, 'wb'))

