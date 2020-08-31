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

def get_conditional_prop(max,num_of_class,total,prem):
    denominator = num_of_class / total
    numerator = denominator * prem
    conditional_prop = max * (numerator / denominator)

    return conditional_prop

pred_train, pred_test, tar_train, tar_test = train_test_split(predictors,target,test_size=0.3,random_state=0,stratify=target)
split_threshold = 3
for i in range(2, split_threshold):
    # 1)-----------------------------------------------------------------------------------------------
    classifier = DecisionTreeClassifier(criterion="entropy", random_state=999, min_samples_split=50)  # configure the classifier
    classifier = classifier.fit(pred_train, tar_train)  # train a decision tree model
    predictions = classifier.predict(pred_test)  # deploy model and make predictions on test set
    print(predictions)
    print("Accuracy score of our model with Decision Tree:", i, accuracy_score(tar_test, predictions))  #overall accuracy_score
    confusion_Dt = confusion_matrix(tar_test, predictions)
    print(confusion_Dt)  #display confusion matrix
    # 2a)-----------------------------------------------------------------------------------------------
    #classification_report =classification_report(tar_test, predictions)
    #print(classification_report)
    precision = precision_score(y_true=tar_test, y_pred=predictions, average='micro')
    print("Precision score of our model with Decision Tree:", precision)
    recall = recall_score(y_true=tar_test, y_pred=predictions, average='micro')
    print("Recall score of our model with Decision Tree :", recall)
    tree.plot_tree(classifier, filled=True)
    prob_tree = classifier.predict_proba(pred_test)
    print(prob_tree)    #get the probability of each sample.


#MLP Classification

clf = MLPClassifier(activation='logistic', solver="sgd", learning_rate_init =0.1, alpha= 0.00000001, hidden_layer_sizes=(15,5),
                    random_state=1, max_iter=800)
clf.fit(pred_train,np.ravel(tar_train, order="C"))
predictions_mlp = clf.predict(pred_test)
print("Accuracy score of our model with MLP:", accuracy_score(tar_test,predictions_mlp))
prob_mlp = clf.predict_proba(pred_test)
confusion_mlp = confusion_matrix(tar_test,predictions_mlp)
classification_report_mlp = classification_report(tar_test,predictions_mlp)
print(classification_report_mlp)
#scores = cross_val_score(clf, predictions_mlp, target, cv=10)
#print("Accuracy score of our model with MLP under cross validation", scores.mean())

confusion_mlp = confusion_matrix(tar_test, predictions_mlp)

print(confusion_mlp)

save_class = []

def check_index(highest_index):
    if highest_index == 0:
        save_class.append(['d '])
    elif highest_index == 1:
        save_class.append(['h '])
    elif highest_index == 2:
        save_class.append(['o '])
    elif highest_index == 3:
        save_class.append(['s '])

def check_prem(index):
    if index == 0:
        return [0.86,48]
    elif index == 1:
        return [0.80,26]
    elif index == 2:
        return [0.81,25]
    elif index == 3:
        return [0.84,58]


for i in range(0,len(prob_mlp)):
    mlp_list = prob_mlp[i]
    tree_list = prob_tree[i]
    test = np.add(mlp_list,tree_list)/2
    print(test)

    highest_index = max(test)

    highest_index = list(test).index(highest_index)
    check_index(highest_index)



print(save_class)
print(np.array(tar_test))
print("Accuracy score of our model with MLP:", accuracy_score(tar_test,save_class))
print(prob_mlp)
total_class_dt = 0
prem_dt = 0

for i in range(0, len(prob_mlp)):
    #print(tree_list_con)
    mlp_list = list(prob_tree[i])

    print("prop_tree")
    max_prop = (max(prob_tree[i]))
    print(max_prop)
    index = mlp_list.index(max_prop)
    #get_conditional_prop()
    print(index)
    premision = check_prem(index)
    total_class_dt = premision[1]
    prem_dt = premision[0]
    print(total_class_dt)
    print(prem_dt)
    print(max_prop)


    result = get_conditional_prop(max_prop,total_class_dt,157,prem_dt)
    print(result)







#get_conditional_prop(max,num_of_class,total,prem)



