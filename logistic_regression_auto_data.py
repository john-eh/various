import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import default_rng

def load_auto():

	# import data
	Auto = pd.read_csv('Auto.csv', na_values='?', dtype={'ID': str}).dropna().reset_index()

	# Extract relevant data features
	X_train = Auto[['mpg','cylinders','displacement','horsepower','weight', 'acceleration','year']].values
	Y_train = Auto[['origin']].values

	return X_train, Y_train

def normalize_input(X_train, Y_train):
        
    X_train = ( X_train -  np.mean(X_train,0) )/ np.std(X_train,0)
    
    # add another row of ones to input to represent bias term
    X2 = np.ones(X_train.shape[0]).reshape((X_train.shape[0],1))
    X_train = np.hstack( (X_train, X2) )
     
    # Create matrix for y data to be compared to softmax probability outputs
    Y2 = np.zeros(( 3, Y_train.shape[0] ))
    Y2[0][ np.where(np.any(Y_train == 1, axis=1)) ] = 1
    Y2[1][ np.where(np.any(Y_train == 2, axis=1)) ] = 1
    Y2[2][ np.where(np.any(Y_train == 3, axis=1)) ] = 1
    
    return X_train, Y2

def sort_sets(X, Y, test_size, sort_random):
    # sorts data into training and testing, either randomly 
    #  or by slicing by test_size
    
    if (sort_random):
        rng = default_rng()
        ind_array = np.arange(Y.shape[0])
        inds = rng.choice(ind_array, test_size, replace=False)
        
        Y_test = Y[inds]
        Y_train = np.delete (Y, inds, axis = 0)
    
        X_test = X[inds]
        X_train = np.delete (X, inds, axis = 0)
    else:
        X_train = X[test_size:,:]
        Y_train = Y[test_size:]

        X_test = X[:test_size,:]
        Y_test = Y[:test_size]
        
    return X_test, Y_test, X_train, Y_train

def initialize_parameters(nRows, nCols ):
    #initializes parameters w to ones
    
    w = np.ones((nRows, nCols))
    
    return w

def stablesoftmax(w, X):
    wX = w @ X
    shiftWx = wX- np.max(wX)
    exps = np.exp(shiftWx)
    return exps / np.sum(exps, 0)

def softmax(w, X):
    wX = w @ X
    denom=  np.sum(np.exp(wX),0)
    g =  np.exp(wX)/denom
    return g

def compute_cost( X_train, nRows, w , Y2, soft_max):
    # computes cost function 
    J = -np.sum( np.log (soft_max )*Y2)/ nRows
    
    return J

def model_backward( X_train, w, Y2, soft_max):
    # computes and returns gradient of cost function
    
    grad_w = - ( (Y2  -  soft_max ) @ X_train )
    
    return grad_w

def update_parameters(w, grad_w, alpha):
    # update parameters w,b using their gradients and learning rate alpha
    
    w_new = w - alpha*grad_w
    
    return w_new  
    
def train_logistic_model( X_train, Y_train, iterations, alpha):
    
    (nRows, nWeights), nClasses = X_train.shape, Y_train.shape[0]
    w = initialize_parameters(nClasses, nWeights )
    loss_history = []
    
    for i in range(iterations):
        soft_max = softmax(w, X_train.T) 
        loss_history.append(compute_cost(X_train, nRows, w, Y_train , soft_max))
        grad_w = model_backward(X_train, w, Y_train, soft_max)
        w = update_parameters(w, grad_w, alpha)
    return w, loss_history

def check_accuracy(w, X_test, Y_test):
    
    # Give softmax probabilities for test inputs and weights
    test = ( softmax( w,  X_test.T) )
    # Take class with highest probability 
    test_max = np.argmax(test, 0)
    # Compare test results with y data
    test_diff = np.abs (test_max - ( Y_test -1 ).T)
    test_diff[ test_diff >1 ] = 1
    # 
    accuracy = 1 - np.sum(test_diff)/test_diff.shape[1]
    
    return accuracy

def main():

    X_train, Y_train = load_auto()
    
    learning_rate = 0.01
    iterations = 1000
    test_size = 300
    rand = 1
    
    X_test, Y_test, X_train, Y_train = sort_sets(X_train, Y_train, 
                                                 test_size, rand)
    
    X_train, Y_vec = normalize_input(X_train, Y_train)
    X_test, _ = normalize_input(X_test, Y_test)
    
    w, loss = train_logistic_model( X_train, Y_vec, iterations, learning_rate)

    plt.plot(loss)
    plt.title('Loss function for logistic regression')    
    plt.xlabel('Iteration')
    plt.ylabel('Cost')   
    print('Final loss:', loss[-1] )
    
    accuracy = check_accuracy(w, X_test, Y_test)
    print('Accuracy for computed weights:    ',accuracy)
    
    random_test = check_accuracy(np.zeros((3, 8)), X_test, Y_test)
    print('Accuracy when using random values:', random_test)

if __name__ == "__main__":
    main()