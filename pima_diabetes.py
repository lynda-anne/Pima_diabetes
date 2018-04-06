import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import scale
import seaborn as sns

pima = pd.read_csv('diabetes.csv')

print(pima.head())
print(pima.info())
# print(pima.describe())
cm = pima.corr()
sns.heatmap(cm, square=True)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()


#data to be modeled
X= pima.drop(['Outcome'], axis=1).values
X_scaled= scale(X)
y= pima['Outcome'].values
print(X.shape, y.shape)

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)
Xs_train, Xs_test, ys_train, ys_test= train_test_split(X_scaled, y, test_size=0.3, random_state=21, stratify=y)

#model with SVC
svm= SVC()
svm.fit(X_train, y_train)
y_pred= svm.predict(X_test)
# print('Test set predicitons: \n {}'.format(ysvm.fit(X_train, y_train)
print('Score of SVC {}' .format(svm.score(X_test, y_test)))

#Model using SCV with scaling x
svm= SVC()
svm.fit(Xs_train, ys_train)
# print('Test set predicitons: \n {}'.format(ysvm.fit(X_train, y_train)
ys_pred= svm.predict(Xs_test)
print('Score of SVC, X_scaled  {}' .format(svm.score(Xs_test, ys_test)))

#Model using KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred= knn.predict(X_test)
# print('Test set predicitons: \n {}'.format(y_pred))
print('Score of KNN, n_neigh=5 {}' .format(knn.score(X_test, y_test)))

# #Model using KNeighborsClassifier with scaled X
knn= KNeighborsClassifier(n_neighbors=5)
knn.fit(Xs_train, ys_train)
ys_pred= knn.predict(Xs_test)
# print('Test set predicitons: \n {}'.format(y_pred))
print('Score of KNN, n_neigh=5, X_scaled {}' .format(knn.score(Xs_test, ys_test)))
#
neighbors= np.arange(1, 11)
train_accuracy= np.empty(len(neighbors))
test_accuracy= np.empty(len(neighbors))

for i, k in enumerate(neighbors) :
    knn= KNeighborsClassifier(n_neighbors= k)
    knn.fit(X_train, y_train)
    train_accuracy[i]=knn.score(X_train, y_train)
    test_accuracy[i]=knn.score(X_test, y_test)

plt.plot(neighbors, train_accuracy, label='Training')
plt.plot(neighbors, test_accuracy, label='Test')
plt.xlabel('k_value')
plt.ylabel('accuracy')
plt.title('Accuracy Train vs Test')
plt.legend(loc='best', fontsize='small', markerscale=0.7)
plt.show()
#

neighbors= np.arange(1, 11)
train_accuracy= np.empty(len(neighbors))
test_accuracy= np.empty(len(neighbors))

for i, k in enumerate(neighbors) :
    knn= KNeighborsClassifier(n_neighbors= k)
    knn.fit(Xs_train, ys_train)
    train_accuracy[i]=knn.score(Xs_train, ys_train)
    test_accuracy[i]=knn.score(Xs_test, ys_test)

plt.plot(neighbors, train_accuracy, label='Training')
plt.plot(neighbors, test_accuracy, label='Test')
plt.xlabel('k_value')
plt.ylabel('accuracy')
plt.title('Scaled Accuracy Train vs Test')
plt.legend(loc='best', fontsize='small', markerscale=0.7)
plt.show()


# knn= KNeighborsClassifier(n_neighbors=8)
# knn.fit(X_train, y_train)
# y_pred= knn.predict(X_test)
#
# # print('Test set predicitons: \n {}'.format(y_pred))
# print('Score of KNN, n_negh=8 {}' .format(knn.score(X_test, y_test)))
