#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns


# ## Importing dataset
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')
df = pd.read_csv('SVM_credit_data.txt', delimiter='\s+', header=None)
df.head(5)


# ## Defining attributes and labels

# #### We have selected 1st and 2nd attributes, because they had highest correlation with the target
X=df.iloc[:,[0,1]]
y=df.iloc[:,-1]


# ## Spilting the dataframe into train test and validation set
from sklearn.model_selection import train_test_split
X_model, X_test, y_model, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train, X_val, y_train, y_val = train_test_split(X_model, y_model, test_size = 0.125, random_state = 0)


# ## Model training using different kernels
from sklearn import svm
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


# ### 1. Linear Kernel 
#Show and plot descision boundary and margin
def show_plot_hyperplane_margin(model, C, title):
    axis = plt.gca()
    xlimit = axis.get_xlim()
    ylimit = axis.get_ylim()
    
    # # create a mesh grid to evaluate model
    xmesh = np.linspace(xlimit[0], xlimit[1], 30)
    ymesh = np.linspace(ylimit[0], ylimit[1], 30)
    Ymesh, Xmesh = np.meshgrid(ymesh, xmesh)
    xygrid = np.vstack([Xmesh.ravel(), Ymesh.ravel()]).T
    R = model.decision_function(xygrid).reshape(Xmesh.shape)
    
    # # Plot both the decision boundary and margins
    axis.contour(Xmesh, Ymesh, R, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    
    # # plot all the support vectors
    axis.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=50,linewidth=1, facecolors='none', edgecolors='k')
    axis.set_xlim(xlimit)
    axis.set_ylim(ylimit)
    plt.xlabel("X")
    plt.ylabel("Y")
    title = title +' when C = {0}'.format(C)
    axis.set_title(title)
    plt.show();
    return


from matplotlib.colors import ListedColormap
X_set, y_set = X_train.to_numpy(), y_train.to_numpy()
Cparams = [0.1, 1, 10]
for C in Cparams:
    svc_linear = svm.SVC(kernel='linear',C=C)
    svc_linear.fit(X_train, y_train)
    y_pred_linear = svc_linear.predict(X_test)
    print("Linear Kernel C=%d ::", C)
    print("accuracy score:", accuracy_score(y_test, y_pred_linear))
    print("f1 score:", f1_score(y_test, y_pred_linear))
    # ## Plot the decision boundary
    plt.scatter(X_set[:, 0], X_set[:, 1], c=y_set, s=30, cmap='autumn')
    show_plot_hyperplane_margin(svc_linear, C, "Linear Kernel")


# ### 2. Rbf Kernel
print()
Cparams = [0.1, 1, 10]
#Gparams = [0.01, 1]
for C in Cparams:
    #for G in Gparams:
    svc_rbf = svm.SVC(kernel='rbf',C=C, gamma=0.01)
    svc_rbf.fit(X_train, y_train)
    y_pred_rbf = svc_rbf.predict(X_test)
    print("Rbf Kernel C=%d ::", C)
    print("accuracy score:", accuracy_score(y_test, y_pred_rbf))
    print("f1 score:", f1_score(y_test, y_pred_linear))
    # ## Plot the decision boundary
    plt.scatter(X_set[:, 0], X_set[:, 1], c=y_set, s=30, cmap='autumn')
    show_plot_hyperplane_margin(svc_rbf, C, "Rbf Kernel")


# ### 3. Polynomial kernel
print()
Cparams = [1, 10, 100]
for C in Cparams:
    #for G in Gparams:
    svc_poly = svm.SVC(kernel='poly',C=C )
    svc_poly.fit(X_train, y_train)
    y_pred_poly = svc_poly.predict(X_test)
    print("Poly Kernel C=%d ::", C)
    print("accuracy score:", accuracy_score(y_test, y_pred_poly))
    print("f1 score:", f1_score(y_test, y_pred_poly))
    # ## Plot the decision boundary
    plt.scatter(X_set[:, 0], X_set[:, 1], c=y_set, s=30, cmap='autumn')
    show_plot_hyperplane_margin(svc_poly, C, "Polynomial Kernel")

