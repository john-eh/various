import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def load_auto():

	# import data
	Auto = pd.read_csv('Auto.csv', na_values='?', dtype={'ID': str}).dropna().reset_index()

	# Extract relevant data features
	X_train = Auto[['cylinders','displacement','horsepower','weight', 'acceleration','year','origin']].values
	Y_train = Auto[['mpg']].values

	return X_train, Y_train

def normalize_input(X_train):
        
    X_train = ( X_train -  np.mean(X_train,0) )/ np.std(X_train,0)
    
    X2 = np.ones(X_train.shape[0]).reshape((X_train.shape[0],1))
    X_train = np.hstack( (X_train, X2) )
    
    return X_train

def initialize_parameters( nCols ):
    #initializes parameters w to ones
    
    w = np.ones((1, nCols))
    
    return w

def model_forward(X_train, w):
    #evaluates model using parameters w
    
    z =  w @ X_train.T
    
    return z

def compute_cost(X_train, nRows, Y_train, w):
    #computes cost J between y and z

    z = model_forward(X_train, w)
    J = np.square( Y_train.T - z )
    L = np.sum(J)/ nRows
    
    return L

def model_backward(X_train, Y_train, z, nRows, nCols):
    # computes and returns gradient of cost function
    
    y_list = []
    for _ in range(nCols):
        y_list.append( z - Y_train.T )
    ybar = np.vstack(y_list)
    
    # ybar =  z - Y_train.T 
    # for _ in range(nCols-1):
    #      ybar = np.vstack( (ybar, z - Y_train.T  ) ) 
    
    grad_w = (-2 * np.sum (ybar * X_train.T, 1)/nRows).reshape(nCols, 1)

    return grad_w

def update_parameters(w, grad_w, alpha):
    #updates parameters w,b using their gradients and learning rate alpha
    
    w_new = w + alpha*grad_w.T
    
    return w_new  

def predict(X, w):
    #predicts z based on inputs, parameters w, and b
    
    if ( len( X.shape) == 1 ):
        X = X.reshape(1,X.shape[0])
    
    if(X.shape[1] != w.shape[1]):
         raise ValueError('wrong dimensions between weights and input')
    
    z = model_forward(X, w )
    return z

def train_linear_model( X_train, Y_train, iterations, learning_rate):
    # trains linear model for input X_train, output Y_train using
    # gradient descent for chosen learning rate and number of iterations
    
    nRows, nCols = X_train.shape
    w = initialize_parameters( nCols )
    loss_history = []
    
    for i in range(iterations):
        
        loss_history.append( compute_cost(X_train, nRows, Y_train, w ) )
        z = model_forward(X_train, w )
        grad_w = model_backward(X_train, Y_train, z, nRows, nCols)
        w = update_parameters(w, grad_w , learning_rate)
        
    return w, loss_history

def linear_regression_ex(X_train, Y_train, title, learning_rate, it):

    loss = []
    legend = []
    for alpha in learning_rate:
        w, loss_history = train_linear_model(X_train, Y_train, it, alpha )
        loss.append(loss_history)
        legend.append ( r' $\alpha$ = {:.0e}'.format(alpha) )
        
    for list in loss:
        plt.plot(list)
    plt.title(title)    
    plt.xlabel('Iteration')
    plt.ylabel('Cost')   
    plt.legend(legend,loc='upper right') 
    plt.show()

def main():

    X_train, Y_train = load_auto()
    X_train = normalize_input(X_train )
    X_hp =  normalize_input( np.array( [X_train[:,2]] ).T )
    
    learning_rate = [1e-1, 1e-2, 1e-3, 1e-4]
    iterations = 1000
    
    string = ['All features', 'Using  horsepower as input']
    
    tic = time.perf_counter() 
    linear_regression_ex(X_train, Y_train, string[0], learning_rate, iterations)
    toc = time.perf_counter() 
    print( toc - tic)
    linear_regression_ex(X_hp, Y_train, string[1], learning_rate, iterations)
    
    w_hp, _ = train_linear_model(X_hp, Y_train, iterations, learning_rate[0] )
    
    x = np.linspace(np.amin(X_hp), np.amax(X_hp), len(X_hp))
    y = w_hp[0,0]*x +  w_hp[0,1
                            ]
    plt.scatter( X_hp[:,0], Y_train, s=2)
    plt.plot(x ,y,c= 'r')
    plt.xlabel('Horse-power')
    plt.ylabel('MPG')   
    plt.legend(['Best linear fit ', ' Data points'],loc='upper right') 
    plt.show()

if __name__ == "__main__":
    main()