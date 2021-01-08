#!/usr/bin/env python
# coding: utf-8
# ## Importing libraries
import matplotlib.pyplot as plt
#Show and plot descision boundary and margin
def show_plot_hyperplane_margin(model, title):
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
    axis.set_title('{0}'.format(title))
    plt.show();
    return

# ## Importing libraries
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib as mlp
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.datasets.samples_generator import make_blobs
# ##Generate dataset and split into train/test dataset (80:20)
X, y = make_blobs(n_samples=300, centers=2,random_state=0, cluster_std=0.80)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Q1) Display what happens when we use linear SVM function vs any of the Kernel SVM functions
# Model training using different kernels
## 1.  Linear Kernel
svc_linear = svm.SVC(kernel='linear',C=0.1)
svc_linear.fit(X_train, y_train)
y_pred_linear = svc_linear.predict(X_test)
print("Linear Kernel ::")
print("accuracy score:", accuracy_score(y_test, y_pred_linear))
print("f1 score:", f1_score(y_test, y_pred_linear))
# ## Plot the decision boundary
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap='autumn')
show_plot_hyperplane_margin(svc_linear, "Linear SVM")

## 2. Rbf kernel
svc_rbf = svm.SVC(kernel='rbf', C=100, gamma=0.099)
svc_rbf.fit(X_train, y_train)
y_pred_rbf = svc_rbf.predict(X_test)
print()
print("Rbf Kernel ::")
print("accuracy score:", accuracy_score(y_test, y_pred_rbf))
print("f1 score:", f1_score(y_test, y_pred_rbf))
# ## Plot the decision boundary
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap='summer')
show_plot_hyperplane_margin(svc_rbf, "RBF Kernel SVM")
