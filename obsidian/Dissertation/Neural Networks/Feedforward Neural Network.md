The simplest form of an [[Artificial Neural Network]] is a Multi Layer Perceptron, also called a Feedforward Neural Network, whose main objective is to approximate to function $f$ that models relationships between input data $x$ and output data $y$ of considerate complexity. It defines a mapping $y=f(x;\theta)$ and learns the best composition of parameters $\theta$ to approximate it to the unknown model. The MLP serves as a fundamental part of developing the other discussed type of ANN which is the Convolutional Neural Network. 

## The Perceptron

![[perceptron.png]]

The main building block of a MLP is the *Perceptron*, a simple computational model initially designed as a binary classificator that mimics biological neurons' behaviour. A neuron might have many inputs $x$ and has a single output $y$. It contains a vector of *weights* $w = (w_1 ... w_m)$, each associated with a single input, and a special weight $b$ called the *bias*. In this context, a perceptron defines a computational operation formulated as the equation portrays.

![[perceptron_eq.png]]

Functions that compute $b + \textbf{w} \cdot \textbf{x} > 0$ are called *linear units* and are identified with $\Sigma$.
An [[Activation Function]] $g$ was introduced to enable the output of non-linear data, the default recommendation is the *Rectified Linear Unit* or *ReLU*, which merely transforms all outputs less than zero as 0. The sigmoid function is also another possibility. 
Feedforward networks are composed of an input layer, formed of the vector of input values, an output layer which is the last layer of neurons, and an arbitrary number of hidden layers, the bigger the number the higher is the \textit{depth} of the network. 
On its own the model amounts only to a complex function, but with a real-world correspondence between input values and associated outputs, we can train a feedforward network to approximate the unknown function of the environment. In more concrete terms, this involves updating all of the different weight and bias values of each neuron to achieve an output as close as possible to the real or desired value, or minimize the total loss which indicates how distant is the network model to the real function to approximate to. *Loss functions* are used to calculate this value.