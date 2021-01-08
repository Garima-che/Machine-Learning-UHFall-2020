import numpy as np

#Defined function for Linear SVM
class LinearSVM_GD:
    def __init__(self, C):
        self.X = X
        self.y = y
        self.C = C
        self.beta0 = None
        self.beta = None 
        self.supportvector = None
        self.hyperplane = None
        
    def train(self, X, y, learning_rate, epochs):
        [n,d] = X.shape
        self.beta = np.zeros(d)
        self.beta0 = 0
        
        for _ in range(epochs):
            #y = w.beta + beta0 is the equation for the decision boundary or hyperplane
            self.hyperplane = y * (X.dot(self.beta) + self.beta0) 
            idx_miss_classified_pts = np.where(self.hyperplane<1)[0]

            #Calculating the weight vector, beta
            beta_gradient = self.beta - (self.C * (y[idx_miss_classified_pts].dot(X[idx_miss_classified_pts])))
            self.beta = self.beta - (learning_rate * beta_gradient)

            #Calculating the bias vector beta0
            beta0_gradient = - self.C * np.sum(y[idx_miss_classified_pts]);
            self.beta0 = self.beta0 - (learning_rate * beta0_gradient)
            
        self.supportvector = np.where(self.hyperplane < 1)[0];
        
    def predict(self, X):
        return np.sign(X.dot(self.beta) + self.beta0);

   
    def score(self, X, y):
        prediction = np.sign(X.dot(self.beta) + self.beta0);
        return np.mean(y == prediction)



#Defined function for Kernel SVM
import numpy as np

class KernelSVM_GD:
    def __init__(self, C, sigma):
        self.X = X
        self.y = y
        self.C = C
        self.kernel = self.rbfkernel
        self.sigma = sigma
        self.beta0 = None
        self.beta = None 
        self.supportvector = None
        self.hyperplane = None
        
    def rbfkernel(self, Xi, Xj):
        return np.exp(-(1/self.sigma ** 2) * np.linalg.norm(Xi[:,np.newaxis] - Xj[np.newaxis,:], axis=2) ** 2)
        
    def train(self, X, y, learning_rate, epochs):
#         Initialize Beta and b
        self.beta = np.zeros(X.shape[0])
        self.beta0 = 0
        self.X = X
        self.y = y
        self.K = self.kernel(X, X)
        
        for _ in range(epochs):
            #y = w.(K(x,x)) + beta0 is the equation for the decision boundary or hyperplane
            self.hyperplane = y * (self.beta.dot(self.kernel(self.X, X)) + self.beta0)
            idx_miss_classified_pts = np.where(self.hyperplane<1)[0]

            #Calculating the weight vector, beta
            beta_gradient = self.K.dot(self.beta) - (self.C * y[idx_miss_classified_pts].dot(self.K[idx_miss_classified_pts]))
            self.beta = self.beta - (learning_rate * beta_gradient)

            #Calculating the bias vector beta0
            beta0_gradient = - self.C * np.sum(y[idx_miss_classified_pts]);
            self.beta0 = self.beta0 - (learning_rate * beta0_gradient)
            
        self.supportvector = np.where(self.hyperplane<1)[0]; 

    def predict(self, X):
        return np.sign(self.beta.dot(self.kernel(self.X, X)) + self.beta0);

    def score(self, X, y):
        prediction = np.sign(self.beta.dot(self.kernel(self.X, X)) + self.beta0);
        return np.mean(y == prediction)



#Defined functions to load training and test data 
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

def load_train_data():
    dataX =  pd.read_csv('X_train_new.csv',index_col=[0])
    dataY = pd.read_csv('y_train_new.csv',index_col=[0])
    xy_data=pd.DataFrame(dataX)
    xy_data['DEATH_EVENT']=dataY
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(xy_data["DEATH_EVENT"])
    X = xy_data.drop(["DEATH_EVENT"], axis=1)
    return X.values, y

def load_test_data():
    data =  pd.read_csv('df_test_scaled.csv',index_col=[0])
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(data["DEATH_EVENT"])
    X = data.drop(["DEATH_EVENT"], axis=1)
    return X.values, y

def load_data():
    data =  pd.read_csv('heart_failure_clinical_records_dataset.csv',index_col=[0])
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(data["DEATH_EVENT"])
    X = data.drop(["DEATH_EVENT"], axis=1)
    return X.values, y


#Load data from dataset (output of CBSO cluster technique)
X, y = load_train_data()
Xtest, ytest = load_test_data()
# For SVM the targets should always be -1, +1
y[y == 0] = -1
ytest[ytest == 0] = -1

#Scale the data
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize Linear SVM parameters
C=4;
model = LinearSVM_GD(C);
model.train(X, y, 0.0001, 100);
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
ypred = model.predict(Xtest);
print("Linear SVM")
print("Accuracy :",accuracy_score(ytest, ypred, normalize=True))
print("F1 score :",f1_score(ytest, ypred, average='weighted'))
print("MCC Score:",matthews_corrcoef(ytest, ypred))
print()

# Initialize Kernel SVM parameters
C=200;
sigma=2.0;
model = KernelSVM_GD(C,sigma);
model.train(X, y, 1e-5, 500);
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
ypred = model.predict(Xtest);
print("Kernel SVM")
print("Accuracy :",accuracy_score(ytest, ypred, normalize=True))
print("F1 score :",f1_score(ytest, ypred, average='weighted'))
print("MCC Score:",matthews_corrcoef(ytest, ypred))








