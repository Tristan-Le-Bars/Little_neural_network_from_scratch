# primitive_sorting_AI
This program is a machine learning algorithm able to guess if an object is a squarre or a rectangle.

## Prerequisites
to use this program you need to install python3 ,pip and the numpy lib:
```bash
sudo apt install python3 python3-pip python3-numpy
```
## How does it work ?
The neural network of this program contain 3 layer of neurons:
- an input layer
- an hidden layer
- an output layer

![alt text](https://github.com/Tristan-Le-Bars/primitive_sorting_AI/blob/master/Neural_network.png)

At the begening of the program, we set up the dataset that containes the values that describes the objects (in this program : a lenght and a width) and the values of the object that the AI will predict the type of this object (a squarre or a rectangle)

## About this AI
The dataset is already writed in the code. But you can modifie it to make it find what you want untill it only need 2 values to describe it because the neural network only have 2 input neurones, you can increment the number of neurones in every if you need more parameters to describes the objectes.
