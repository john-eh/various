import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import imageio
import glob
import time

def load_mnist():    

    # Loads the MNIST dataset from png images    
    NUM_LABELS = 10            
    # create list of image objects    
    test_images = []    
    test_labels = []        
    for label in range(NUM_LABELS):        
    	for image_path in glob.glob("MNIST/Test/" + str(label) + "/*.png"):            
    		image = imageio.imread(image_path)            
    		test_images.append(image)            
    		letter = [0 for _ in range(0,NUM_LABELS)]                
    		letter[label] = 1            
    		test_labels.append(letter)      
    
    # create list of image objects    
    train_images = []    
    train_labels = []        
    
    for label in range(NUM_LABELS):
    	for image_path in glob.glob("MNIST/Train/" + str(label) + "/*.png"):  
    		image = imageio.imread(image_path)            
    		train_images.append(image)            
    		letter = [0 for _ in range(0,NUM_LABELS)]                
    		letter[label] = 1            
    		train_labels.append(letter)                      
    
    X_train= np.array(train_images).reshape(-1,784)/255.0    
    Y_train= np.array(train_labels)    
    X_test= np.array(test_images).reshape(-1,784)/255.0    
    Y_test= np.array(test_labels)
    
    return X_train, Y_train, X_test, Y_test

def initialize_parameters(sizes, layers):
    ''' initializes weights and biases randomly for given sizes and layers '''
    
    weights = [None]*layers
    bias = [None]*layers

    for i in range(layers):
        
        rng = default_rng()
        w = np.zeros((sizes[i+1],  sizes[i] ))
        
        weights[i] = rng.normal(w, 0.01)
        bias[i] = np.zeros( (sizes[i+1], 1) )
        
    return  weights, bias            

def sigmoid( x ):

    return 1/(1 + np.exp(-x))
    
def sigmoid_backward( x ):
    ''' Computes derivative of sigmoid function '''
    
    return (np.exp(-x))/((np.exp(-x) + 1)**2)
    
def relu( x ):
    ''' Computes rectifier activation function '''
    
    return  np.maximum(x,0)

def relu_backward( x):
    ''' Computes derivative of rectifier activation function '''
    
    re = np.copy(x)
    re[re < 0] = 0
    re[re >= 0] = 1
    return re

def softmax( x ):
    ''' computes shifted softmax function to achieve more stable results '''

    exps = np.exp(x - x.max())
    
    return exps/np.sum(exps, axis=0)


def activate_forward( x, activation_func):
    ''' returns activation function depending on which activation 
    function is chosen for layer, 0 = sigmoid, 1 = ReLu'''    
    if (activation_func == 0):
        return sigmoid(x)
    return relu(x)

def model_forward( X_train, weights, bias, layers, act_funcs):
    ''' computes parameter values going forward through neural network 
        and saves them in list model_state for easier backpropagation '''
    
    model_state = [ [None]*2 for _ in range(layers) ]
    model_state[0][0] = X_train.T
    model_state[0][1] = (weights[0] @ model_state[0][0]) + bias[0]
    
    for i in range(1, layers):

        model_state[i][0] = activate_forward(model_state[i-1][1], act_funcs[i-1])
        model_state[i][1] = (weights[i] @ model_state[i][0]) + bias[i]
                
    y = softmax(model_state[-1][1])    
    
    return y, model_state


def compute_loss(z, Y, n):
    ''' computes cross-entropy loss between outputs z and test data Y '''
    
    L = np.log(np.sum(np.exp(z), 0)) - np.sum((Y.T * z), 0)
    J = np.sum(L)/n
    return J

def activate_backward(x, activation_func):
    ''' returns derivative of activation function depending on which 
        activation function is chosen, 0 = sigmoid, 1 = ReLu'''
        
    if (activation_func == 0):
        return sigmoid_backward(x)
    
    return relu_backward(x)
    
    
def model_backward(y, y_out, model_state, weights, bias, layers, act_funcs):
    ''' computes and stores gradients for weights and
        biases in list grad_state using back-propagation '''

    n = (1/y_out.shape[0])
    grad_state = [ [None]*2 for _ in range(layers) ]
    
    # outermost gradient of cross-entropy cost function = sigmoid - one-hot(Y)
    grad = (y_out - y.T) 
    grad_state[0][0] = n * np.sum(grad, axis=1, keepdims=True)
    grad_state[0][1] = n * (grad @ (model_state[layers - 1][0]).T)
    
    # gradients are back-propagated through layers using 
    # previous grad and outputs stored in model-state 
    for i in range(1, layers):     
 
        grad = (weights[layers- (i)].T @ grad) * \
          activate_backward(model_state[layers-(1+i)][1], act_funcs[layers-i])
            
        grad_state[i][0] = n * np.sum(grad, axis=1, keepdims=True)
        grad_state[i][1] = n * (grad @ (model_state[layers - (1 + i)][0]).T)

    return grad_state
    
def update_param(grad_state, weights, bias, alpha):
    ''' updates weights and biases using gradients from grad-state '''
    
    for g, w, b in zip(grad_state[::-1], weights, bias):
        w -= alpha * g[1]
        b -= alpha * g[0]
        
    return weights, bias

def update_momentum( grad_state, weights, bias, alpha, momentum):
    ''' Updates parameters using gradients and stored momentum term '''

    if not momentum:
        momentum = [ [ (np.zeros( (grad.shape[0], grad.shape[1]) )) 
                           for grad in layer ] for layer in grad_state ]    
    
    for g, m, in zip(grad_state, momentum):
        
        m[1] = 0.9 * m[1] + alpha * g[1]
        m[0] = 0.9 * m[0] + alpha * g[0]

    for m, w, b in zip(momentum[::-1], weights, bias):        
        w -= m[1]
        b -= m[0]
        
    return weights, bias, momentum

def update_adam(grad_state, weights, bias, alpha, m, v, k, layers):
    ''' Updates parameters using gradients and stored m and v terms '''

    # Parameter values as explained in paper https://arxiv.org/abs/1412.6980
    beta1 = 0.9
    beta2 = 0.99
    epsilon = 1e-8 

    if not m:
        m = [ [ (np.zeros( (grad.shape[0], grad.shape[1]) ))
                          for grad in layer ] for layer in grad_state ]  
        v = [x[:] for x in m]

    for i in range(layers):
        m[i][1] = (1 - beta1) * grad_state[i][1] + beta1 * m[i][1] 
        m[i][0] = (1 - beta1) * grad_state[i][0] + beta1 * m[i][0] 
        
        v[i][1] = (1 - beta2) * np.square(grad_state[i][1]) + beta2 * v[i][1] 
        v[i][0] = (1 - beta2) * np.square(grad_state[i][0]) + beta2 * v[i][0]          

        mB_shifted =  m[i][0] / (1 - np.power(beta1,k))
        mW_shifted =  m[i][1] / (1 - np.power(beta1,k))

        vB_shifted =  v[i][0] / (1 - np.power(beta2,k))
        vW_shifted =  v[i][1] / (1 - np.power(beta2,k))

        weights[layers - (i+1)] -= alpha * np.divide(mW_shifted, np.add( 
                                            np.sqrt(vW_shifted), epsilon ))
        
        bias[layers - (i+1)]    -= alpha * np.divide(mB_shifted, np.add( 
                                            np.sqrt(vB_shifted), epsilon ))

    return weights, bias, m, v


def draw_minibatch(X, Y, batch_size, batch_N, length_X):
    ''' Draws and returns list of 'batch_N' mini-batches with 
        sizes 'batch_size' from data X, Y after randomizing data  '''    
    
    X_batches = []
    Y_batches = []
    rng = np.random.default_rng()  
    index_permutation = rng.permutation(length_X)
    
    X = X[index_permutation, :]
    Y = Y[index_permutation, :]  
    
    for i in range(batch_N):
        start = i * batch_size
        end = min((i + 1) * batch_size, length_X - 1)
        X_batches.append( X[start:end, :] )
        Y_batches.append( Y[start:end, :] )

    return X_batches, Y_batches

def predict(X, Y, weights, bias, layers, act_funcs):
    ''' computes proportion of inputs X that are predicted correctly
        compared to outputs Y using given parameters weights and bias '''
    
    # let X values go thorugh neural net
    prediction, _ = model_forward(X, weights, bias,layers, act_funcs) 
    
    # pick class with highest probability and compare to class in Y
    z_max = np.argmax(prediction, 0)
    Y_max = np.argmax(Y, 1)
    diff = np.abs (z_max - Y_max)
    diff[diff > 1] = 1
    accuracy = 1 - np.sum(diff)/diff.shape[0]
    
    return accuracy, prediction

def train_neural_network(X_train, Y_train, X_test, Y_test, sizes, iterations, 
                       batch_size, alpha, act_funcs, printout = 0, update = 0):
    
    start_time = time.time()
    length_X = X_train.shape[0]
    layers = len(sizes)-1
    predict_list = []
    loss_test = []
    loss_train = []
    moment = 0
    var = 0
    i = 1
    weights, bias = initialize_parameters(sizes, layers)

    batch_N = int(length_X/batch_size) + (length_X/batch_size > 0)

    for it in range(iterations):

        # draw and iterate over mini-batches
        X_batches, Y_batches = draw_minibatch(
            X_train, Y_train, batch_size, batch_N-1, length_X)
        
        for x_batch, y_batch, k in zip(X_batches, Y_batches, range(batch_N)):
        
            # Propagate forward through network for output for all layers
            y_out, model_state = model_forward(
                x_batch, weights, bias, layers, act_funcs)
            # Propagate backward through network to find gradients 
            grad_state = model_backward(
                y_batch, y_out, model_state, weights, bias, layers,act_funcs)    
            # Update parameters using chosen update alogrithm            
            if update == 1:
                weights, bias, moment = update_momentum(
                    grad_state, weights, bias, alpha, moment)
            elif update == 2:
                weights, bias, moment, var = update_adam(
                    grad_state, weights, bias, alpha, moment, var, i, layers)
            else:
                weights, bias = update_param(grad_state, weights, bias, alpha)
            
            i += 1              
            if k%100 == 0:
                loss_train.append(compute_loss(y_out,y_batch,y_batch.shape[0])) 
            if k%300 == 0:
                prediction, y2 = predict(X_test, Y_test, weights, bias, layers,
                                         act_funcs)
                loss_test.append (compute_loss(y2, Y_test, Y_test.shape[0]) ) 
                predict_list.append (prediction)
                
        if (printout):
            print('Iteration: {0}, Time : {1:.2f}s, Accuracy: {2:.3f}%'
            .format(it+1, time.time() - start_time, prediction * 100))
        
    return (predict_list, loss_test, loss_train, weights, bias )

def main():

    results_and_plots()

def results_and_plots():

    # test and training data from MNIST database
    X_train, Y_train, X_test, Y_test = load_mnist()
    
    iterations = 25         # number of iterations model goes through    
    batch_size = 64         # mini-batch size for mini-batch gradient descent
    learning_rate = 0.1     # model's learning rate
    printout = 1            # toggle 1 or 0 for if output is printed for each iteration
    sizes = [784, 10]       # matrix sizes for neural network
    act_funcs = [0,0]       # activation functions for each layer, 0 = sigmoid, 1 = relu
    
    #one layer neural network
    result_one_layer = train_neural_network(
        X_train, Y_train, X_test, Y_test, sizes, iterations, batch_size, 
        learning_rate, act_funcs, printout, 0)
    
    # plots weight matrices as images
    for w in result_one_layer[1][0]:
        plt.plot( w.reshape((28,28)) )
        plt.show()
    
    sizes = [784, 128, 10]
    act_funcs = [0,0,0]
    
    #standard gradient descent
    learning_rate = 0.1
    result_neural_net = train_neural_network(
        X_train, Y_train, X_test, Y_test, sizes, iterations, batch_size, 
        learning_rate, act_funcs, printout, 0)
    
    # momentum
    iterations2 = 10
    learning_rate = 0.05
    result_momentum = train_neural_network(
        X_train, Y_train, X_test, Y_test, sizes, iterations2, batch_size, 
        learning_rate, act_funcs, printout, 1)
    
    # ADAM
    learning_rate = 0.0075
    result_adam = train_neural_network(
        X_train, Y_train, X_test, Y_test, sizes, iterations2, batch_size, 
        learning_rate, act_funcs, printout, 2)
    
    x1 = np.linspace(0, iterations, len(result_neural_net[2]))
    x2 = np.linspace(0, iterations, len( result_neural_net[1]))
    plt.plot(x1, result_neural_net[2], x2,  result_neural_net[1])
    plt.title('Loss function for training')    
    plt.xlabel('Iteration')
    plt.ylabel('Cost')  
    plt.legend( ['Loss for mini-batch training data','Loss for test data'])
    plt.show()
    x3 = np.linspace(0, iterations, len(result_neural_net[0][1:]) )
    plt.plot(x3,result_neural_net[0][1:])
    plt.title('Accuracy for deep neural network')    
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy') 
    plt.show()
    
    fig = plt.figure()
    a1 = fig.add_axes([0,0,1,1])
    x4 = np.linspace(0, iterations2, len(result_adam[1]))
    a1.plot(x4, result_adam[1],
            x4, result_momentum[1], 
            x4, result_neural_net[1][:len(x4)])
    a1.set_title('Loss function for test data')    
    a1.set_xlabel('Iteration')
    a1.set_ylabel('Cost')  
    a1.set_ylim( min(result_adam[1])-0.01, 1.65 )
    a1.legend( ['Using adam','Using momentum', 'using traditional'])
    plt.show()
    
    fig2 = plt.figure()
    a2 = fig2.add_axes([0,0,1,1])
    x5 = np.linspace(0, iterations2, len(result_adam[0]))
    a2.plot(x5, result_adam[0],
            x5, result_momentum[0], 
            x5, result_neural_net[0][:len(x4)])
    a2.set_title('Accuracy for test data')    
    a2.set_xlabel('Iteration')
    a2.set_ylabel('Cost')  
    a2.set_ylim( 0.9, 1 )
    a2.legend( ['Using adam','Using momentum', 'using traditional'])
    plt.show()

if __name__ == "__main__":
    main() 

