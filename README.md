# MLP

###Neural Networks
Neural networks consist of a number of units (neurons) which are connected by weighted links. These units are typically organised in several layers, namely, an input layer, one or more hidden layers, and an output layer. The input layer receives an external activation vector and passes it via weighted connections to the units in the first hidden layer. 
 
Figure shows input layer with R elements, one hidden layer with S neurons, and output layer with one element. Each neuron in the network is a simple processing unit that computes its activation with respect to its incoming excitation, the so-called net input, where denotes the set of predecessors of unit  denotes the connection weight from unit  to  unit, and  is the unit bias value. The activation of unit, is computed by passing the net input through a non-liner activation function. 

Neural networks are widely used for solving many problems in most science problems of linear and nonlinear cases. Neural network algorithms are always iterative, designed to step by step minimise (targeted minimal error) the difference between the actual output vector of the network and the desired output vector, examples include the Backpropagation (BP) algorithm and the Resilient Propagation (RPROP) algorithm.


### Back Propagation Learning Algorithm
BP is the most widely used algorithm for supervised learning with multilayered feed-forward networks [22]. A Back Propagation network learns by example. You give the algorithm examples of what you want the network to do and it changes the network’s weights so that, when training is finished, it will give you the required output for a particular input. Back Propagation networks are ideal for simple Pattern Recognition and Mapping Tasks. 

 
###Step-wise Procedure: 
1. First apply the inputs to the network and work out the output – remember this initial output could be anything, as the initial weights were random numbers.
2. Next work out the error for neuron B. The error is What you want – What you actually get, in other words:
ErrorB = OutputB (1-OutputB)(TargetB – OutputB)
The “Output(1-Output)” term is necessary in the equation because of the Sigmoid Function – if we were only using a threshold neuron it would just be (Target – Output).
3. Change the weight. Let W+AB be the new (trained) weight and WAB be the initial
weight.
W+AB = WAB + (ErrorB x OutputA)
Notice that it is the output of the connecting neuron (neuron A) we use (not B). We update all the weights in the output layer in this way.
4. Calculate the Errors for the hidden layer neurons. Unlike the output layer we can’t calculate these directly (because we don’t have a Target), so we Back Propagate them from the output layer (hence the name of the algorithm). This is done by taking the Errors from the output neurons and running them back through the weights to get the hidden layer errors. For example if neuron A is connected as shown to B and C then we take the errors from B and C to generate an error for A.
ErrorA = Output A (1 - Output A)(ErrorB WAB + ErrorC WAC)
Again, the factor “Output (1 - Output )” is present because of the sigmoid squashing function.
5. Having obtained the Error for the hidden layer neurons now proceed as in stage 3 to change the hidden layer weights. By repeating this method we can train a network of any number of layers.


Code Implémentation – Back Propagation Algorithm 
The Code uses the following algorithm

create an empty neural network
loop
  generate a candidate set of weights
  load weights into neural network
  foreach training vector
    compute the neural output
    compute cross-entropy error
    accumulate total error
  end foreach
  if current weights are best found so far
    save current weights
  end if
until some stopping condition
return best weights found

