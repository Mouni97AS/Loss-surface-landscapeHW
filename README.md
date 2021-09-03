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
sampling two random direction vectors in parameter space, and computing the loss at a number of points on the plane defined by the two vectors:
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
