import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#% matplotlib inline
import pickle

import os
for dirname, _, filenames in os.walk('/home/karan/Downloads/data/phishing.csv'):
    for filename in filenames:
        path = os.path.join(dirname, filename)


raw_dataset = pd.read_csv(path)

raw_dataset.shape


raw_dataset.describe()



selected_features = [i for (i,j) in features_selected if i != 'status']
selected_features

X_selected = original_dataset[selected_features]
X_selected




X_selected.shape


y = original_dataset['status']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_selected, y,test_size=0.2,random_state=42,shuffle = True)

from sklearn.ensemble import RandomForestClassifier

model_random_forest = RandomForestClassifier(n_estimators=350,random_state=42,)

model_random_forest.fit(X_train,y_train)


from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

def custom_accuracy_set (model, X_train, X_test, y_train, y_test, train=True):
    
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_train)
    
    
    if train:
        x = X_train
        y = y_train
    elif not train:
        x = X_test
        y = y_test
        
    y_predicted = model.predict(x)
    
    accuracy = accuracy_score(y, y_predicted)
    print('model accuracy: {0:4f}'.format(accuracy))
    oconfusion_matrix = confusion_matrix(y, y_predicted)
    print('Confusion matrix: \n {}'.format(oconfusion_matrix))
    oroc_auc_score = lb.transform(y), lb.transform(y_predicted)


custom_accuracy_set(model_random_forest, X_train, X_test, y_train, y_test, train=True)

custom_accuracy_set(model_random_forest, X_train, X_test, y_train, y_test, train=False)

 #export the model

#import pickle

with open('model_phishing_webpage_classifer','wb') as file:
    pickle.dump(model_random_forest,file)


                                            