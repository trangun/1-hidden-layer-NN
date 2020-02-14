Name: Trang Nguyen
Homework 3: Backpropagation

--Files
nguyen_backprop.py: main file
test.py: has the experiment function and add extra args.plot to represent the experiment

--Network Representation
A class for model is created which has some support functions: sigmoid, d_sigmoid (aka derivative sigmoid), 
forward, backprop.
forward(self, xs) and backprop(self, ys, xs) follow the pseudocode provided in the lecture notes 
part 5: Multi Layer Perceptron.

sigmod(x):
   return 1/(1+ exp(-x))

d_sigmoid(x):
   return sigmoid(x)*(1-sigmoid(x))

forward(self, xs):
   ai = wi * zi
   zi = sigmoid(ai) (this can be y_hat)

backprop(self, ys, xs):
   w2 = w2 - lr*delta2*xs.T
   w1 = w1 - lr*delta1*xs.T
   

--To train the model, 
	while in 1 iteration:
	    for y from 1 to N:
		forward(xs)
		backprop(ys, xs)

--To calculate the accuracy:
	for y from 1 to N:
	    forward(xs)
	    if y_hat <= 0.5:
		y = 0
	    else:
		y = 1
	    if y == ys:
		accuracy += 1
	return accuracy/len(ys)

--Experiment:
In the scope of this hw, we only have 1 hidden layer; therefore, the input of sigmoid for that hidden layer is 
x*weights and the imput of sigmoid for the outer layer is output*weights. For this experiment, the weights are 
initalize randomly. As we update the weights after some iterations, the change in weights might become very small
which make the change in forward small small as well. And since we use sigmoid activation function which label any
output of the forward > 0.5 as 1, the forward might eventually will not change after some iterations. In other words,
we can say that the accuracy will eventually converge after some iterations. 

We will do some experiment to observe the performace with vary hyperparameters to see it more clearly.
Hyperparameters: For the experiment, I will use 4 different values {1, 0.25, 0.1, 0.01} and {5, 15, 25, 35} 
for learning rate (or lr) and hidden dimension (or hidden_dim) respectively. I will perform the experiment with 
60 iterations.

At learning rate of 1 with 4 different hidden dimension, the accuracy fluctuate a lot. The accuracies here also 
do not converge for both dev and test set even at the largest hidden dimension. Howevever, if we look at the last
plot with hidden_dim=35, we can see that the accuracy is more 'stable' (still fluctuate a lot) for dataset. For 
more details, please look at "lr1.png". So we can see that large learning rate is not ideal for this network since 
the results, in general, vary a lot and have a lot of noise.

I then reduce the learning rate to 0.25. With smallest hidden_dim of 5, the accuracy still varies after 60 iterations.
It becomes more stable as the hidden_dim increases. However, at the largest hidden_dim, the accuracy seems to be more
stable at first, but then suddenly drop down a bit. For more details, please look at "lr025.png." We might need more
iterations for the last plot to converge with larger hidden_dim.

At lr = 0.1, the accuracy seems to converge for the first 3 plots with hidden_dim = {5, 15, 25} respectively. In my
observation, the larger the hidden_dim is, the more interations needed for the accuracy to be converged. We just need
hidden_dim relatively large to keep the accuracy stable and converge within reasonable iterations. Too big hidden_dim 
might be take extra time for the result to be converged. Please look at "lr01.png" for more details.

At lr = 0.01, overfitting appears here. The accuracy converge and pretty stable in all of the plots. However, too 
small learing rate might give us a overfit value. And as the hidden_dim increases, the number of iterations for the
accuracy to be converged is become smaller and smaller. It is because as the number of nodes increase, the updated 
weight for every node will show the change faster despite the small learning rate. For more details, please look at 
"lr001.png"

In my opinion, using the development data, the best set of hyperparameter is when lr=0.1, hidden_dim = 15 with around
30-40 iterations since the accuracy is pretty stable there and seems to converge as well. The learning rate is also 
not so small that we can have overfitting problem. 









