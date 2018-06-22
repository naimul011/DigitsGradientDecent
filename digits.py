from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn import svm
import numpy as np

k = 5 + (80)%4
digits = datasets.load_digits()

#print(digits.data)

#print(digits.target)

#print(digits.images[0])

clf = svm.SVC(gamma=0.00001,C=100)
#clf = SGDClassifier()
#print(len(digits.data))
x,y = digits.data[:],digits.target[:]

kf = KFold(n_splits=k,random_state = 33,shuffle = True)


kf.get_n_splits(x)

accuracy = []

for train_index, test_index in kf.split(x):
       #print("TRAIN:", train_index, "TEST:", test_index)
       X_train, X_test = x[train_index], x[test_index]
       y_train, y_test = y[train_index], y[test_index]
       
       scaler = preprocessing.StandardScaler().fit(X_train)
       
       X_train = scaler.transform(X_train)
       X_test = scaler.transform(X_test)
       
       clf.fit(X_train, y_train)
       y_pred = clf.predict(X_test)
       
       #print('Prediction ',y_pred==y_test)
       #metrics.accuracy_score(y_test, y_pred)
       
       accuracy.append( metrics.accuracy_score(y_test, y_pred))
       print ( metrics.classification_report(y_test, y_pred))

#print('Prediction',clf.predict(digits.data[-1]))
print ("Avg Accuracy: ",np.mean(accuracy))
#print(np.mean(accuracy),"/t",np.mean(precision),"/t",np.mean(f1))


