
#Importing the dependencies
import tensorflow as tf 
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import warnings
warnings.filterwarnings('ignore')

#A class which will be used to create different Compact Layers for different types of neural networks
class Compact_Layer(object):
    
    def Compact(self , data, nodes_and_layers):
        length = len(nodes_and_layers) - 1
        # create hidden layers
        hiddenLayers = [0 for i in range(length)]
        for i in range(length): 
            hiddenLayers[i] = {'weights': tf.Variable(tf.random_normal([nodes_and_layers[i] , nodes_and_layers[i+1]])),
                               'biases' :tf.Variable(tf.random_normal([nodes_and_layers[i+1]]))}
        # change the biases of output
        hiddenLayers[-1]['biases'] = tf.Variable(tf.random_normal([nodes_and_layers[-1]])) 
        # relu(input_date * weights + biases)
        layers = data
        for i in range(length-1):
            layers = tf.nn.relu(tf.add(tf.matmul(layers, hiddenLayers[i]['weights'])  ,hiddenLayers[i]['biases']))
        ls = layers[-1].get_shape().as_list()
        return tf.add(tf.matmul(tf.reshape(layers[-1] , [ls[-1] , 1]), hiddenLayers[-1]['weights'] , transpose_a=True) , hiddenLayers[-1]['biases'])


#Getting the mnist dataset
mnist = input_data.read_data_sets('data/fashion',one_hot = True)


#the number of outputs
n_classes = 10  

#Batch size
batch_size = 128


#Number of perceptrons in each layer . This serves as the main input to our neural network
size_of_output = 10
size_of_input = 784
nodes_and_layers = [ size_of_input, 500, 500, 500, size_of_output ]


#Creating placeholders for our computation graph
x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float" ,[None,n_classes] )




#Training our FeedForward NeuralNetwork
def train_nn(x ,nodes_and_layers):
    obj =Compact_Layer()
    prediction  = obj.Compact(x ,nodes_and_layers)
    cost  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction , labels = y))
    optimizer  = tf.train.AdamOptimizer().minimize(cost)
    epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            epoch_loss = 0
            for temp in range(int(mnist.train.num_examples/batch_size)):
                e_x,e_y = mnist.train.next_batch(batch_size)
                temp , c =sess.run([optimizer , cost] , feed_dict = {x:e_x , y:e_y})
                epoch_loss+=c
            print('Epoch' , epoch ,'completed out of ',epochs , 'losses:' , epoch_loss)


        correct = tf.equal(tf.argmax(prediction ,1) ,tf.argmax(y , 1))
        accuracy  = tf.reduce_mean(tf.cast(correct ,'float'))
        print('Accuracy' ,accuracy.eval({x:mnist.test.images , y:mnist.test.labels}))


#Calling the training function on our data
train_nn(x , nodes_and_layers)
