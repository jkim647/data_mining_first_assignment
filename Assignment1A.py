import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score,auc
from sklearn.metrics import roc_curve,roc_auc_score,plot_roc_curve
from sklearn import tree

path = "/Users/jskim/PycharmProjects/assignment1/Forest.xlsx"
rawdata = pd.read_excel(path)
nrow, ncol = rawdata.shape
print(nrow)
print(ncol)
print(rawdata)
predictors = rawdata.iloc[:,1:ncol]
target = rawdata.iloc[:,0:1]

pred_train, pred_test, tar_train, tar_test = train_test_split(predictors,target,test_size=0.3,random_state=0,stratify=target)
split_threshold = 3
for i in range(2, split_threshold):
    # 1)-----------------------------------------------------------------------------------------------
    classifier = DecisionTreeClassifier(criterion="entropy", random_state=999, min_samples_split=50)  # configure the classifier
    classifier = classifier.fit(pred_train, tar_train)  # train a decision tree model
    predictions = classifier.predict(pred_test)  # deploy model and make predictions on test set
    print(predictions)
    print("Accuracy score of our model with Decision Tree:", i, accuracy_score(tar_test, predictions))  #overall accuracy_score
    confusion = confusion_matrix(tar_test, predictions)
    print(confusion)  #display confusion matrix
    # 2a)-----------------------------------------------------------------------------------------------
    classification_report =classification_report(tar_test, predictions)
    print(classification_report)
    precision = precision_score(y_true=tar_test, y_pred=predictions, average='micro')
    print("Precision score of our model with Decision Tree:", precision)
    recall = recall_score(y_true=tar_test, y_pred=predictions, average='micro')
    print("Recall score of our model with Decision Tree :", recall)
    tree.plot_tree(classifier, filled=True)
    prob = classifier.predict_proba(pred_test)
    print(prob)    #get the probability of each sample.

#MLP Classification

clf = MLPClassifier(activation = 'logistic', solver="sgd", learning_rate_init = 0.1, alpha= 0.000005, hidden_layer_sizes=(5,2),
                    random_state=1, max_iter=500)
clf.fit(pred_train,np.ravel(tar_train, order="C"))
predictions_mlp = clf.predict(pred_test)
print("Accuracy score of our model with MLP:", accuracy_score(tar_test,predictions_mlp))
prob_mlp = clf.predict_proba(pred_test)
print(prob_mlp)
#scores = cross_val_score(clf, predictions_mlp, target, cv=10)
#print("Accuracy score of our model with MLP under cross validation", scores.mean())



