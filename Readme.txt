https://github.com/cauchyturing/ANRAT

This is an implementation of the ANRAT methods in the paper 

"
Adaptive Normalized Risk-Averting Training For Deep Neural Networks
Zhiguang Wang, Tim Oates, James Lo
Proceedings of The Thirtieth AAAI Conference on Artificial Intelligence
"

Dependencies:
Theano
Lasagne
Keras (Only used for loading the MNIST dataset)

Using a pure simple ConvNets of 32-32-256-10, you should be able to achieve 0.39%-0.40% error rate on MNIST, which is around the state-of-the-art with 'Convlolutional Kernel Network' or 'Deeply Supervised Nets'.

