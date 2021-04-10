from sklearn import datasets
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import  train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import  matplotlib.pyplot as plt

import streamlit as st
import  numpy as np

#Main Section
st.title("50 Days of Code")

st.write('''
# Figuring out which Classifier is best
Let's Go figures
''')

#SideBars

ds_name = st.sidebar.selectbox("Select your Dataset", ("Iris", "Wine"))
classifier_name = st.sidebar.selectbox("Select your classifiers", ("KNN", "SVM" , "Random Forest"))

def get_dataset(ds_name):
    if ds_name == "Iris":
        data = datasets.load_iris()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(ds_name)

st.write("Shape of Dataset", X.shape)
st.write("Number of Unique Values", len(np.unique(y)))

def add_param(classifier_name):
    param = dict()

    if classifier_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        param["K"] = K
    elif classifier_name == "SVM":
        
        C = st.sidebar.slider("C", 0.01, 10.0)
        param["C"] = C

    else:
        
        max_depth = st.sidebar.slider("max_depth", 2, 20)
        n_estimators = st.sidebar.slider("n_estimators", 1, 150)
        param["max_depth"] = max_depth
        param["n_estimators"]= n_estimators

    return param

param =  add_param(classifier_name)

def get_classifier(classifier_name, param):
    

    if classifier_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=param["K"])
    elif classifier_name == "SVM":
       clf = SVC(C=param["C"])
    else:
        clf = RandomForestClassifier(n_estimators=param["n_estimators"], max_depth=param["max_depth"], random_state=1452)
    return clf


clf = get_classifier(classifier_name, param)

#Classification

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state=1452)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.write(f"Classifier name : {classifier_name}")
st.write(f"Accuracy : {acc}")

#Plot
pca = PCA(2)
X_projected= pca.fit_transform(X)

x1 = X_projected[: , 0]
x2 = X_projected[: , 1]

fig = plt.figure()
plt.scatter(x1, x2,c=y, alpha=0.8, cmap="virdis")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.colorbar()

st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()