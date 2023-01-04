import numpy as np
import pandas as pd
import sys

X = np.loadtxt(sys.argv[1], delimiter = ",")
X = np.append(X, np.reshape(np.sin(X[:, 0]), newshape = (X.shape[0], 1)), axis=1)
X = np.append(X, np.reshape(np.sin(X[:, 1]), newshape = (X.shape[0], 1)), axis=1)
Y = np.loadtxt(sys.argv[2], delimiter = ",")
Y = np.reshape(Y, newshape=(Y.shape[0],1))

batch_size = 15
lr = 0.12
epochs = 150

def initialize():
    global W1
    global W2
    global outH
    global netH
    global outO
    global netO
    global b1
    global b2
    W1 = np.random.randn(4,7)/np.sqrt(7)
    b1 = np.zeros([7,])/np.sqrt(7)
    W2 = np.random.randn(7,1)/np.sqrt(7)
    b2 = np.zeros([1,1])
    outH = np.zeros([batch_size,7])
    netH = np.zeros([batch_size,7])
    outO = np.zeros([batch_size,1])
    netO = np.zeros([batch_size,1])

def sigmoid(x):
    return 1/(1+np.exp(-x))

def binary_cross_entropy(actual,predicted):
    return -((actual*np.log(predicted)) + (1-actual)*np.log(1-predicted))

def hyperTan(x):
    return np.tanh(x)

def vectorize(ar):
    return np.vectorize(sigmoid)(ar)    

def forwardPass(x):
    global outH
    global netH
    global outO
    global netO
    global b1
    global b2
    ip = x
    outH = np.matmul(ip,W1)+b1
    netH = vectorize(outH)
    outO = np.matmul(netH,W2)+b2
    netO = vectorize(outO)

    return netO

def backwardPass(x,y,predicted):
    global W1
    global W2
    global b1
    global b2
    
    W1 = np.subtract(W1, lr*((np.matmul(np.multiply(np.matmul(np.subtract(predicted, y), W2.T), np.multiply(netH, np.subtract(1,netH))).T, x))/x.shape[0]).T)
    W2 = np.subtract(W2,lr*(np.matmul(np.subtract(predicted,y).T,netH).T)/x.shape[0])
    b1 = np.subtract(b1, lr*(np.reshape((np.matmul(np.multiply(np.matmul(np.subtract(predicted, y), W2.T), np.multiply(netH, np.subtract(1,netH))).T, np.ones([x.shape[0], 1]))), newshape=(7,))/x.shape[0]))
    b2 = np.subtract(b2,lr*(np.matmul(np.subtract(predicted,y).T,np.ones([x.shape[0],1])))/x.shape[0])

def train():
    global lr
    totalBatch = X.shape[0]//batch_size
    
    for i in range(epochs):
        if i>100:
            lr = 0.1
        perm = np.random.permutation(X.shape[0])
        x_perm = X[perm]
        y_perm = Y[perm]
        loss=0
        for j in range(totalBatch):
            x_train = x_perm[j*batch_size:min(j*batch_size+batch_size,x_perm.shape[0])]
            y_train = y_perm[j*batch_size:min(j*batch_size+batch_size,y_perm.shape[0])]
            
            prediction = forwardPass(x_train)
            backwardPass(x_train,y_train,prediction)
            if loss==0:
                loss = np.mean(binary_cross_entropy(y_train,prediction))
            else:
                loss = (loss+np.mean(binary_cross_entropy(y_train,prediction)))/2
        print("Epoch : ",i)
        print("Loss : ",loss)
            
def predict():
    global X_t
    X_t = np.loadtxt(sys.argv[3], delimiter = ",")
    X_t = np.append(X_t, np.reshape(np.sin(X_t[:, 0]), newshape = (X_t.shape[0], 1)), axis=1)
    X_t = np.append(X_t, np.reshape(np.sin(X_t[:, 1]), newshape = (X_t.shape[0], 1)), axis=1)
    
    pred_list = forwardPass(X_t)
    pred_list = np.around(pred_list)
    np.savetxt("test_predictions.csv", pred_list, delimiter=',', fmt='%d')

initialize()
train()
predict()




