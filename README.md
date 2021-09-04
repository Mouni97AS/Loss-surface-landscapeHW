# Loss-surface-landscapeHW

the repository related to my homework visualizing the loss surface landscape.
Pytorch implementation of Convolutional neural network for visualizing the loss surface. The project done on google Colab (provide a GPU for training the model).
I have done the project in two different datasets. big datasets consist of traffic signs of 52 classes, and the Mnist handwritten. 

## Visualizing loss surface

we define a Loss metric over data X and labels y. The `loss_landscapes.model_metrics` package contains a number of metrics that cover common use cases, evaluates a loss function.
```
import loss_landscapes
import loss_landscapes.metrics

x, y = next(iter(testLoader))
metric = loss_landscapes.metrics.Loss(criterion, x, y)
```  
return a 2-dimensional array of loss values, and computing the loss at a number of points on the plane defined by the two array:
```
loss_data_fin = loss_landscapes.random_plane(
    model_final.cpu(), 
    metric,
    8, 
    STEPS, 
    normalization='filter', 
    deepcopy_model=True
)
```
## Results 
### Traffic Signs
  (./images result/mnist/mnist1.png)
  (./images result/mnist/mnist 10 epoch 100 step.png)
  (./images result/mnist/100 step 20 epoch.png)
  (./images result/mnist/20 epoch 20 steps)
## Some of the potential candidate parameters that may affect the loss surface are:
Some of the potential candidate parameters that may affect the loss surface are:
* Number of epochs: as for a too low or too high number of epochs the network may underfit or overfit the data, however finding the best epochs number may give a near to truth surface.
* Model depth: the deepest our network is the complex the loss function will be.
* Layer’s order: as the backpropagation literally defines the loss function, and for it layers order is critical.
* Plotting steps: the higher the steps the smoother our plot is:

