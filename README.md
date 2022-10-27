# Fashion-MNIST Classification using Convolutional Neural Networks (CNN) in Python

<img src="images/banner-01.jpg" width = "1000" />

## 1. Objective

The objective of this section is to develop a Convolutional Neural Network (CNN) to classify the clothing-articles using the widely used Fashion-MNIST dataset.

## 2. Motivation

The MNIST handwritten digit classification problem is a standard dataset used in computer vision and deep learning.

Since the MINIST dataset is effectively solved,  the Fashion-MNIST has recently been collected and labelled to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine and deep learning algorithms. The Fashion-MNIST dataset shares the same image size and structure of training and testing splits.as the original MNIST dataset

In this section, we shall demonstrate how to develop convolutional neural network for clothing items classification from scratch, using the Fashion-MNIST dataset, including:

* How to prepare the input training and test data 
* How to deploy the model
* How to use the trained model to make predictions
* How to evaluate its performance

## 3. Data

The Fashion-MNIST is a dataset of clothing articles images:
* It contains training set of 60,000 examples
* It contains a test set of 10,000 examples.
  * Each example is a 28x28 grayscale image
  * Each example is associated with a label from 10 classes.
* Recently, the Fashion-MNIST is starting to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine and deep learning algorithms.
  * It shares the same image size and structure of training and testing splits.
  * Additional detailed about the Fashion can be in [1]

Sample images from the MNIST data set are illustrated next:
  * There are significant variations between the different types of clothing articles
  * There are significant variations between different examples of the same clothing article class.
  
<img src="images/fashion-mnist-sprite.png" width = "1000"/>

## 4. Development

In this section, we shall demonstrate how to develop a Convolutional Neural Network (CNN) for clothing-articles classification from scratch, including:

* How to prepare the input training and test data 
* How to deploy the model
* How to use the trained model to make predictions
* How to evaluate its performance

* Author: Mohsen Ghazel (mghazel)
* Date: April 6th, 2021

* Project: FASHION-MNIST clothing articles Classification using Convolutional Neural Networks (CNN):


The objective of this project is to demonstrate how to develop a Convolutional Neural Network (CNN) to classify images of clothing articles from the Fashion-MNIST dataset:

* Fashion-MNIST is a dataset of Zalando's article images
 * It contains training set of 60,000 examples
 * It contains a test set of 10,000 examples.
  * Each example is a 28x28 grayscale image
  * Each example is associated with a label from 10 classes.
 * Recently, the Fashion-MNIST is starting to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine and deep learning algorithms.
   * It shares the same image size and structure of training and testing splits.
   * Additional detailed about the Fashion can be found here:
     * https://github.com/zalandoresearch/fashion-mnist

We shall apply the standard Machine and Deep Learning model development and evaluation process, with the following steps:

1. Load the MNIST dataset of handwritten digits:
 * 60,000 labelled training examples
 * 10,000 labelled test examples
  * Each handwritten example is 28x28 pixels binary image.


2. Build a simple CNN model
3. Train the selected ML model
4. Deploy the trained on the test data
5. Evaluate the performance of the trained model using evaluation metrics:
   * Accuracy
   * Confusion Matrix
   * Other metrics derived form the confusion matrix.

### 4.1. Part 1: Imports and global variables:

#### 4.1.1 Standard scientific Python imports:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Numpy</span>
<span style="color:#800000; font-weight:bold; ">import</span> numpy <span style="color:#800000; font-weight:bold; ">as</span> np
<span style="color:#696969; "># matplotlib</span>
<span style="color:#800000; font-weight:bold; ">import</span> matplotlib<span style="color:#808030; ">.</span>pyplot <span style="color:#800000; font-weight:bold; ">as</span> plt
<span style="color:#696969; "># - import sklearn to use the confusion matrix function</span>
<span style="color:#800000; font-weight:bold; ">from</span> sklearn<span style="color:#808030; ">.</span>metrics <span style="color:#800000; font-weight:bold; ">import</span> confusion_matrix
<span style="color:#696969; "># import itertools</span>
<span style="color:#800000; font-weight:bold; ">import</span> itertools
<span style="color:#696969; "># opencv</span>
<span style="color:#800000; font-weight:bold; ">import</span> cv2
<span style="color:#696969; "># tensorflow</span>
<span style="color:#800000; font-weight:bold; ">import</span> tensorflow <span style="color:#800000; font-weight:bold; ">as</span> tf

<span style="color:#696969; "># keras input layer</span>
<span style="color:#800000; font-weight:bold; ">from</span> tensorflow<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers <span style="color:#800000; font-weight:bold; ">import</span> <span style="color:#400000; ">Input</span>
<span style="color:#696969; "># keras conv2D layer</span>
<span style="color:#800000; font-weight:bold; ">from</span> tensorflow<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers <span style="color:#800000; font-weight:bold; ">import</span> Conv2D
<span style="color:#696969; "># keras MaxPooling2D layer</span>
<span style="color:#800000; font-weight:bold; ">from</span> tensorflow<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers <span style="color:#800000; font-weight:bold; ">import</span> MaxPooling2D
<span style="color:#696969; "># keras Dense layer</span>
<span style="color:#800000; font-weight:bold; ">from</span> tensorflow<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers <span style="color:#800000; font-weight:bold; ">import</span> Dense
<span style="color:#696969; "># keras Flatten layer</span>
<span style="color:#800000; font-weight:bold; ">from</span> tensorflow<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers <span style="color:#800000; font-weight:bold; ">import</span> Flatten
<span style="color:#696969; "># keras Dropout layer</span>
<span style="color:#800000; font-weight:bold; ">from</span> tensorflow<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>layers <span style="color:#800000; font-weight:bold; ">import</span> Dropout
<span style="color:#696969; "># keras model</span>
<span style="color:#800000; font-weight:bold; ">from</span> tensorflow<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>models <span style="color:#800000; font-weight:bold; ">import</span> Model
<span style="color:#696969; "># keras sequential model</span>
<span style="color:#800000; font-weight:bold; ">from</span> tensorflow<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>models <span style="color:#800000; font-weight:bold; ">import</span> Sequential
<span style="color:#696969; "># optimizers</span>
<span style="color:#800000; font-weight:bold; ">from</span> tensorflow<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>optimizers <span style="color:#800000; font-weight:bold; ">import</span> SGD

<span style="color:#696969; "># random number generators values</span>
<span style="color:#696969; "># seed for reproducing the random number generation</span>
<span style="color:#800000; font-weight:bold; ">from</span> random <span style="color:#800000; font-weight:bold; ">import</span> seed
<span style="color:#696969; "># random integers: I(0,M)</span>
<span style="color:#800000; font-weight:bold; ">from</span> random <span style="color:#800000; font-weight:bold; ">import</span> randint
<span style="color:#696969; "># random standard unform: U(0,1)</span>
<span style="color:#800000; font-weight:bold; ">from</span> random <span style="color:#800000; font-weight:bold; ">import</span> random
<span style="color:#696969; "># time</span>
<span style="color:#800000; font-weight:bold; ">import</span> datetime
<span style="color:#696969; "># I/O</span>
<span style="color:#800000; font-weight:bold; ">import</span> os
<span style="color:#696969; "># sys</span>
<span style="color:#800000; font-weight:bold; ">import</span> sys

<span style="color:#696969; "># check for successful package imports and versions</span>
<span style="color:#696969; "># python</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Python version : {0} "</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>sys<span style="color:#808030; ">.</span>version<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># OpenCV</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"OpenCV version : {0} "</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>cv2<span style="color:#808030; ">.</span>__version__<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># numpy</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Numpy version  : {0}"</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>np<span style="color:#808030; ">.</span>__version__<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># tensorflow</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Tensorflow version  : {0}"</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>tf<span style="color:#808030; ">.</span>__version__<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>

Python version <span style="color:#808030; ">:</span> <span style="color:#008000; ">3.7</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">10</span> <span style="color:#808030; ">(</span>default<span style="color:#808030; ">,</span> Feb <span style="color:#008c00; ">20</span> <span style="color:#008c00; ">2021</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">21</span><span style="color:#808030; ">:</span><span style="color:#008c00; ">17</span><span style="color:#808030; ">:</span><span style="color:#008c00; ">23</span><span style="color:#808030; ">)</span> 
<span style="color:#808030; ">[</span>GCC <span style="color:#008000; ">7.5</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span> 
OpenCV version <span style="color:#808030; ">:</span> <span style="color:#008000; ">4.1</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">2</span> 
Numpy version  <span style="color:#808030; ">:</span> <span style="color:#008000; ">1.19</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">5</span>
Tensorflow version  <span style="color:#808030; ">:</span> <span style="color:#008000; ">2.4</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">1</span>
</pre>

#### 4.1.2. Global variables:
  
  
<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># -set the random_state seed = 100 for reproducibilty</span>
random_state_seed <span style="color:#808030; ">=</span> <span style="color:#008c00; ">100</span>

<span style="color:#696969; "># the number of visualized images</span>
num_visualized_images <span style="color:#808030; ">=</span> <span style="color:#008c00; ">25</span>
</pre>


### 4.2. Part 2: Load FASHION-MNIST Dataset

#### 4.2.1. Load the FASHION-MNIST dataset :
* Load the MNIST dataset of handwritten digits:
  * 60,000 labelled training examples
  * 10,000 labelled test examples
    * Each handwritten example is 28x28 pixels binary image.

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Load in the FASHION-MNIST data set</span>
<span style="color:#696969; "># - It has 10 classes</span>
fashion_mnist <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>datasets<span style="color:#808030; ">.</span>fashion_mnist
<span style="color:#696969; "># split the data into teaining and test subsets</span>
<span style="color:#808030; ">(</span>x_train<span style="color:#808030; ">,</span> y_train<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span>x_test<span style="color:#808030; ">,</span> y_test<span style="color:#808030; ">)</span> <span style="color:#808030; ">=</span> fashion_mnist<span style="color:#808030; ">.</span>load_data<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
</pre>

#### 4.2.2. Explore training and test images:

##### 4.2.2.1. Display the number and shape of the training and test subsets:
  

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># Training data:</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># the number of training images</span>
num_train_images <span style="color:#808030; ">=</span> x_train<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Training data:"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"x_train.shape: "</span><span style="color:#808030; ">,</span> x_train<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Number of training images: "</span><span style="color:#808030; ">,</span> num_train_images<span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Image size: "</span><span style="color:#808030; ">,</span> x_train<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">:</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>

<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># Test data:</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># the number of test images</span>
num_test_images <span style="color:#808030; ">=</span> x_test<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Test data:"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"x_test.shape: "</span><span style="color:#808030; ">,</span> x_test<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Number of test images: "</span><span style="color:#808030; ">,</span> num_test_images<span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Image size: "</span><span style="color:#808030; ">,</span> x_test<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">:</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>

<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Training data<span style="color:#808030; ">:</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
x_train<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">:</span>  <span style="color:#808030; ">(</span><span style="color:#008c00; ">60000</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#808030; ">)</span>
Number of training images<span style="color:#808030; ">:</span>  <span style="color:#008c00; ">60000</span>
Image size<span style="color:#808030; ">:</span>  <span style="color:#808030; ">(</span><span style="color:#008c00; ">28</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#808030; ">)</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Test data<span style="color:#808030; ">:</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
x_test<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">:</span>  <span style="color:#808030; ">(</span><span style="color:#008c00; ">10000</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#808030; ">)</span>
Number of test images<span style="color:#808030; ">:</span>  <span style="color:#008c00; ">10000</span>
Image size<span style="color:#808030; ">:</span>  <span style="color:#808030; ">(</span><span style="color:#008c00; ">28</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#808030; ">)</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
</pre>

#### 4.2.2. Reshape the training and test images to 3D:

The training and test images are 2D grayscale/binary:
 * CNN expect the images to be of shape:
   * height x width x color
 * We need to add a fourth color dimension to:
   * The training images: x_train
   * The test images: x_test
   

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># reshape the x_train and x_test images 4D:</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># Add a fourth color dimension to x_train</span>
x_train <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>expand_dims<span style="color:#808030; ">(</span>x_train<span style="color:#808030; ">,</span> <span style="color:#44aadd; ">-</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span> 
<span style="color:#696969; "># add a fourth color dimension to x_test</span>
x_test <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>expand_dims<span style="color:#808030; ">(</span>x_test<span style="color:#808030; ">,</span> <span style="color:#44aadd; ">-</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># display the new shapes of x_train and x_test</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Re-shaped x_train:"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"x_train.shape: "</span><span style="color:#808030; ">,</span> x_train<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Re-shaped x_test:"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"x_test.shape: "</span><span style="color:#808030; ">,</span> x_test<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>

<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Re<span style="color:#44aadd; ">-</span>shaped x_train<span style="color:#808030; ">:</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
x_train<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">:</span>  <span style="color:#808030; ">(</span><span style="color:#008c00; ">60000</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Re<span style="color:#44aadd; ">-</span>shaped x_test<span style="color:#808030; ">:</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
x_test<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">:</span>  <span style="color:#808030; ">(</span><span style="color:#008c00; ">10000</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
</pre>

#### 4.2.3. Display the targets/classes:

* There 10 classes:
   * Each training and test example is assigned to one of the following labels:
   
<img src="images/Fashion-MNIST-Labels.JPG" width = "500"/>
   
##### 4.2.3.1. Display the number of classes:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># The number of classes</span>
num_classes <span style="color:#808030; ">=</span> <span style="color:#400000; ">len</span><span style="color:#808030; ">(</span><span style="color:#400000; ">set</span><span style="color:#808030; ">(</span>y_train<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"The number of classes:"</span><span style="color:#808030; ">,</span> num_classes<span style="color:#808030; ">)</span>

The number of classes<span style="color:#808030; ">:</span> <span style="color:#008c00; ">10</span>

</pre>

##### 4.2.3.2. Create meaningful labels for the different classes:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Create a class-label mapping:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; ">#  - Create a string containg all the classification labels </span>
<span style="color:#696969; ">#  - Seperated by new line character</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
labels <span style="color:#808030; ">=</span> <span style="color:#696969; ">'''T-shirt/top</span>
<span style="color:#696969; ">Trouser</span>
<span style="color:#696969; ">Pullover</span>
<span style="color:#696969; ">Dress</span>
<span style="color:#696969; ">Coat</span>
<span style="color:#696969; ">Sandal</span>
<span style="color:#696969; ">Shirt</span>
<span style="color:#696969; ">Sneaker</span>
<span style="color:#696969; ">Bag</span>
<span style="color:#696969; ">Ankle boot'''</span><span style="color:#808030; ">.</span>split<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"</span><span style="color:#0f69ff; ">\n</span><span style="color:#0000e6; ">"</span><span style="color:#808030; ">)</span>

</pre>

#### 4.2.3.3. Display the created class-label mapping:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># display the classes</span>
<span style="color:#800000; font-weight:bold; ">for</span> counter <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span>num_classes<span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
  <span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Class ID = {}, Class name = {}'</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>counter<span style="color:#808030; ">,</span> labels<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>

<span style="color:#800000; font-weight:bold; ">Class</span> <span style="color:#400000; ">ID</span> <span style="color:#808030; ">=</span> <span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span> <span style="color:#800000; font-weight:bold; ">Class</span> name <span style="color:#808030; ">=</span> T<span style="color:#44aadd; ">-</span>shirt<span style="color:#44aadd; ">/</span>top
<span style="color:#800000; font-weight:bold; ">Class</span> <span style="color:#400000; ">ID</span> <span style="color:#808030; ">=</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">,</span> <span style="color:#800000; font-weight:bold; ">Class</span> name <span style="color:#808030; ">=</span> Trouser
<span style="color:#800000; font-weight:bold; ">Class</span> <span style="color:#400000; ">ID</span> <span style="color:#808030; ">=</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span> <span style="color:#800000; font-weight:bold; ">Class</span> name <span style="color:#808030; ">=</span> Pullover
<span style="color:#800000; font-weight:bold; ">Class</span> <span style="color:#400000; ">ID</span> <span style="color:#808030; ">=</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span> <span style="color:#800000; font-weight:bold; ">Class</span> name <span style="color:#808030; ">=</span> Dress
<span style="color:#800000; font-weight:bold; ">Class</span> <span style="color:#400000; ">ID</span> <span style="color:#808030; ">=</span> <span style="color:#008c00; ">4</span><span style="color:#808030; ">,</span> <span style="color:#800000; font-weight:bold; ">Class</span> name <span style="color:#808030; ">=</span> Coat
<span style="color:#800000; font-weight:bold; ">Class</span> <span style="color:#400000; ">ID</span> <span style="color:#808030; ">=</span> <span style="color:#008c00; ">5</span><span style="color:#808030; ">,</span> <span style="color:#800000; font-weight:bold; ">Class</span> name <span style="color:#808030; ">=</span> Sandal
<span style="color:#800000; font-weight:bold; ">Class</span> <span style="color:#400000; ">ID</span> <span style="color:#808030; ">=</span> <span style="color:#008c00; ">6</span><span style="color:#808030; ">,</span> <span style="color:#800000; font-weight:bold; ">Class</span> name <span style="color:#808030; ">=</span> Shirt
<span style="color:#800000; font-weight:bold; ">Class</span> <span style="color:#400000; ">ID</span> <span style="color:#808030; ">=</span> <span style="color:#008c00; ">7</span><span style="color:#808030; ">,</span> <span style="color:#800000; font-weight:bold; ">Class</span> name <span style="color:#808030; ">=</span> Sneaker
<span style="color:#800000; font-weight:bold; ">Class</span> <span style="color:#400000; ">ID</span> <span style="color:#808030; ">=</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">,</span> <span style="color:#800000; font-weight:bold; ">Class</span> name <span style="color:#808030; ">=</span> Bag
<span style="color:#800000; font-weight:bold; ">Class</span> <span style="color:#400000; ">ID</span> <span style="color:#808030; ">=</span> <span style="color:#008c00; ">9</span><span style="color:#808030; ">,</span> <span style="color:#800000; font-weight:bold; ">Class</span> name <span style="color:#808030; ">=</span> Ankle boot
</pre>


#### 4.2.4. Examine the number of images for each class of the training and testing subsets:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Create a histogram of the number of images in each class/digit:</span>
<span style="color:#800000; font-weight:bold; ">def</span> plot_bar<span style="color:#808030; ">(</span>y<span style="color:#808030; ">,</span> loc<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'left'</span><span style="color:#808030; ">,</span> relative<span style="color:#808030; ">=</span><span style="color:#074726; ">True</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
    width <span style="color:#808030; ">=</span> <span style="color:#008000; ">0.35</span>
    <span style="color:#800000; font-weight:bold; ">if</span> loc <span style="color:#44aadd; ">==</span> <span style="color:#0000e6; ">'left'</span><span style="color:#808030; ">:</span>
        n <span style="color:#808030; ">=</span> <span style="color:#44aadd; ">-</span><span style="color:#008000; ">0.5</span>
    <span style="color:#800000; font-weight:bold; ">elif</span> loc <span style="color:#44aadd; ">==</span> <span style="color:#0000e6; ">'right'</span><span style="color:#808030; ">:</span>
        n <span style="color:#808030; ">=</span> <span style="color:#008000; ">0.5</span>
     
    <span style="color:#696969; "># calculate counts per type and sort, to ensure their order</span>
    unique<span style="color:#808030; ">,</span> counts <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>unique<span style="color:#808030; ">(</span>y<span style="color:#808030; ">,</span> return_counts<span style="color:#808030; ">=</span><span style="color:#074726; ">True</span><span style="color:#808030; ">)</span>
    sorted_index <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>argsort<span style="color:#808030; ">(</span>unique<span style="color:#808030; ">)</span>
    unique <span style="color:#808030; ">=</span> unique<span style="color:#808030; ">[</span>sorted_index<span style="color:#808030; ">]</span>
     
    <span style="color:#800000; font-weight:bold; ">if</span> relative<span style="color:#808030; ">:</span>
        <span style="color:#696969; "># plot as a percentage</span>
        counts <span style="color:#808030; ">=</span> <span style="color:#008c00; ">100</span><span style="color:#44aadd; ">*</span>counts<span style="color:#808030; ">[</span>sorted_index<span style="color:#808030; ">]</span><span style="color:#44aadd; ">/</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>y<span style="color:#808030; ">)</span>
        ylabel_text <span style="color:#808030; ">=</span> <span style="color:#0000e6; ">'% count'</span>
    <span style="color:#800000; font-weight:bold; ">else</span><span style="color:#808030; ">:</span>
        <span style="color:#696969; "># plot counts</span>
        counts <span style="color:#808030; ">=</span> counts<span style="color:#808030; ">[</span>sorted_index<span style="color:#808030; ">]</span>
        ylabel_text <span style="color:#808030; ">=</span> <span style="color:#0000e6; ">'count'</span>
         
    xtemp <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>arange<span style="color:#808030; ">(</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>unique<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
    plt<span style="color:#808030; ">.</span>bar<span style="color:#808030; ">(</span>xtemp <span style="color:#44aadd; ">+</span> n<span style="color:#44aadd; ">*</span>width<span style="color:#808030; ">,</span> counts<span style="color:#808030; ">,</span> align<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'center'</span><span style="color:#808030; ">,</span> alpha<span style="color:#808030; ">=</span><span style="color:#008000; ">.7</span><span style="color:#808030; ">,</span> width<span style="color:#808030; ">=</span>width<span style="color:#808030; ">)</span>
    plt<span style="color:#808030; ">.</span>xticks<span style="color:#808030; ">(</span>xtemp<span style="color:#808030; ">,</span> unique<span style="color:#808030; ">,</span> rotation<span style="color:#808030; ">=</span><span style="color:#008c00; ">45</span><span style="color:#808030; ">)</span>
    plt<span style="color:#808030; ">.</span>xlabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'digit'</span><span style="color:#808030; ">)</span>
    plt<span style="color:#808030; ">.</span>ylabel<span style="color:#808030; ">(</span>ylabel_text<span style="color:#808030; ">)</span>
 
plt<span style="color:#808030; ">.</span>suptitle<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Frequency of images per digit'</span><span style="color:#808030; ">)</span>
plot_bar<span style="color:#808030; ">(</span>y_train<span style="color:#808030; ">,</span> loc<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'left'</span><span style="color:#808030; ">)</span>
plot_bar<span style="color:#808030; ">(</span>y_test<span style="color:#808030; ">,</span> loc<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'right'</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>legend<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span>
    <span style="color:#0000e6; ">'train ({0} images)'</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>y_train<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> 
    <span style="color:#0000e6; ">'test ({0} images)'</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>y_test<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> 
<span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
</pre>

 <img src="images/10-examples-distribution.png" width = "1000"/>
 
 #### 42.2.5. Visualize some of the training and test images and their associated targets:

* First implement a visualization functionality to visualize the number of randomly selected images:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">"""</span>
<span style="color:#696969; "># A utility function to visualize multiple images:</span>
<span style="color:#696969; ">"""</span>
<span style="color:#800000; font-weight:bold; ">def</span> visualize_images_and_labels<span style="color:#808030; ">(</span>num_visualized_images <span style="color:#808030; ">=</span> <span style="color:#008c00; ">25</span><span style="color:#808030; ">,</span> dataset_flag <span style="color:#808030; ">=</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
  <span style="color:#696969; ">"""To visualize images.</span>
<span style="color:#696969; "></span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Keyword arguments:</span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- num_visualized_images -- the number of visualized images (deafult 25)</span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- dataset_flag -- 1: training dataset, 2: test dataset</span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Return:</span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- None</span>
<span style="color:#696969; ">&nbsp;&nbsp;"""</span>
  <span style="color:#696969; ">#--------------------------------------------</span>
  <span style="color:#696969; "># the suplot grid shape:</span>
  <span style="color:#696969; ">#--------------------------------------------</span>
  num_rows <span style="color:#808030; ">=</span> <span style="color:#008c00; ">5</span>
  <span style="color:#696969; "># the number of columns</span>
  num_cols <span style="color:#808030; ">=</span> num_visualized_images <span style="color:#44aadd; ">//</span> num_rows
  <span style="color:#696969; "># setup the subplots axes</span>
  fig<span style="color:#808030; ">,</span> axes <span style="color:#808030; ">=</span> plt<span style="color:#808030; ">.</span>subplots<span style="color:#808030; ">(</span>nrows<span style="color:#808030; ">=</span>num_rows<span style="color:#808030; ">,</span> ncols<span style="color:#808030; ">=</span>num_cols<span style="color:#808030; ">,</span> figsize<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">8</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">10</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
  <span style="color:#696969; "># set a seed random number generator for reproducible results</span>
  seed<span style="color:#808030; ">(</span>random_state_seed<span style="color:#808030; ">)</span>
  <span style="color:#696969; "># iterate over the sub-plots</span>
  <span style="color:#800000; font-weight:bold; ">for</span> row <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span>num_rows<span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
      <span style="color:#800000; font-weight:bold; ">for</span> col <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span>num_cols<span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
        <span style="color:#696969; "># get the next figure axis</span>
        ax <span style="color:#808030; ">=</span> axes<span style="color:#808030; ">[</span>row<span style="color:#808030; ">,</span> col<span style="color:#808030; ">]</span><span style="color:#808030; ">;</span>
        <span style="color:#696969; "># turn-off subplot axis</span>
        ax<span style="color:#808030; ">.</span>set_axis_off<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        <span style="color:#696969; "># if the dataset_flag = 1: Training data set</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        <span style="color:#800000; font-weight:bold; ">if</span> <span style="color:#808030; ">(</span> dataset_flag <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">1</span> <span style="color:#808030; ">)</span><span style="color:#808030; ">:</span> 
          <span style="color:#696969; "># generate a random image counter</span>
          counter <span style="color:#808030; ">=</span> randint<span style="color:#808030; ">(</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span>num_train_images<span style="color:#808030; ">)</span>
          <span style="color:#696969; "># get the training image</span>
          image <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>squeeze<span style="color:#808030; ">(</span>x_train<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">,</span><span style="color:#808030; ">:</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
          <span style="color:#696969; "># get the target associated with the image</span>
          label <span style="color:#808030; ">=</span> labels<span style="color:#808030; ">[</span>y_train<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">]</span><span style="color:#808030; ">]</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        <span style="color:#696969; "># dataset_flag = 2: Test data set</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        <span style="color:#800000; font-weight:bold; ">else</span><span style="color:#808030; ">:</span> 
          <span style="color:#696969; "># generate a random image counter</span>
          counter <span style="color:#808030; ">=</span> randint<span style="color:#808030; ">(</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span>num_test_images<span style="color:#808030; ">)</span>
          <span style="color:#696969; "># get the test image</span>
          image <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>squeeze<span style="color:#808030; ">(</span>x_test<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">,</span><span style="color:#808030; ">:</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
          <span style="color:#696969; "># get the target associated with the image</span>
          label <span style="color:#808030; ">=</span> labels<span style="color:#808030; ">[</span>y_test<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">]</span><span style="color:#808030; ">]</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        <span style="color:#696969; "># display the image</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        ax<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>image<span style="color:#808030; ">,</span> cmap<span style="color:#808030; ">=</span>plt<span style="color:#808030; ">.</span>cm<span style="color:#808030; ">.</span>gray_r<span style="color:#808030; ">,</span> interpolation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'nearest'</span><span style="color:#808030; ">)</span>
        <span style="color:#696969; "># set the title showing the image label</span>
        ax<span style="color:#808030; ">.</span>set_title<span style="color:#808030; ">(</span><span style="color:#400000; ">str</span><span style="color:#808030; ">(</span>label<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> size <span style="color:#808030; ">=</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">)</span>
</pre>


##### 4.2.5.1. Visualize some of the training images and their associated targets:

<img src="images/25-sample-train-images.png" width = "1000"/>

##### 4.2.5.2. Visualize some of the test images and their associated targets:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># call the function to visualize the training images</span>
visualize_images_and_labels<span style="color:#808030; ">(</span>num_visualized_images<span style="color:#808030; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span>
</pre>

<img src="images/25-sample-test-images.png" width = "1000"/>

#### 4.2.6. Normalize the training and test images to the interval: [0, 1]:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Normalize the training images</span>
x_train <span style="color:#808030; ">=</span> x_train <span style="color:#44aadd; ">/</span> <span style="color:#008000; ">255.0</span>
<span style="color:#696969; "># Normalize the test images</span>
x_test <span style="color:#808030; ">=</span> x_test <span style="color:#44aadd; ">/</span> <span style="color:#008000; ">255.0</span>
</pre>


### 4.3. Part 3: Build the CNN model architecture

#### 4.3.1. Design the structure of the CNN model to classify the FASHION-MINIST images:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Build the sequential CNN model</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Build the model using the functional API</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 1: Input layer</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - input images size: (28, 28, 10)</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
i <span style="color:#808030; ">=</span> <span style="color:#400000; ">Input</span><span style="color:#808030; ">(</span>shape<span style="color:#808030; ">=</span>x_train<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">.</span>shape<span style="color:#808030; ">)</span>            
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 2: Convolutional layer</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - 32 filters:  </span>
<span style="color:#696969; ">#   - size: 3x3</span>
<span style="color:#696969; ">#   - same</span>
<span style="color:#696969; ">#   - stride = 2 (non-overlapping)</span>
<span style="color:#696969; "># - Activation function: relu</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------               </span>
x <span style="color:#808030; ">=</span> Conv2D<span style="color:#808030; ">(</span><span style="color:#008c00; ">32</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> strides<span style="color:#808030; ">=</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>i<span style="color:#808030; ">)</span>     
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 3: Convolutional layer</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - 64 filters:  </span>
<span style="color:#696969; ">#   - size: 3x3</span>
<span style="color:#696969; ">#   - same</span>
<span style="color:#696969; ">#   - stride = 2 (non-overlapping)</span>
<span style="color:#696969; "># - Activation function: relu</span>
<span style="color:#696969; ">#------------------------------------------------------------------------------- </span>
x <span style="color:#808030; ">=</span> Conv2D<span style="color:#808030; ">(</span><span style="color:#008c00; ">64</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> strides<span style="color:#808030; ">=</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>x<span style="color:#808030; ">)</span>     
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 4: Convolutional layer</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - 128 filters:  </span>
<span style="color:#696969; ">#   - size: 3x3</span>
<span style="color:#696969; ">#   - same</span>
<span style="color:#696969; ">#   - stride = 2 (non-overlapping)</span>
<span style="color:#696969; "># - Activation function: relu</span>
<span style="color:#696969; ">#------------------------------------------------------------------------------- </span>
x <span style="color:#808030; ">=</span> Conv2D<span style="color:#808030; ">(</span><span style="color:#008c00; ">128</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> strides<span style="color:#808030; ">=</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>x<span style="color:#808030; ">)</span>    
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 5: Flatten</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - Flatten to connect to the next Fully-Connected Dense layer</span>
<span style="color:#696969; ">#------------------------------------------------------------------------------- </span>
x <span style="color:#808030; ">=</span> Flatten<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>x<span style="color:#808030; ">)</span>                                           
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 6: Dropout layer</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - p = 0.20  </span>
<span style="color:#696969; ">#------------------------------------------------------------------------------- </span>
x <span style="color:#808030; ">=</span> Dropout<span style="color:#808030; ">(</span><span style="color:#008000; ">0.2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>x<span style="color:#808030; ">)</span>                                        
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 7: Dense layer</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - 512 neurons</span>
<span style="color:#696969; "># - Activation function: relu</span>
<span style="color:#696969; ">#------------------------------------------------------------------------------- </span>
x <span style="color:#808030; ">=</span> Dense<span style="color:#808030; ">(</span><span style="color:#008c00; ">512</span><span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'relu'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>x<span style="color:#808030; ">)</span>                        
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 8: Dropout layer</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - p = 0.20  </span>
<span style="color:#696969; ">#------------------------------------------------------------------------------- </span>
x <span style="color:#808030; ">=</span> Dropout<span style="color:#808030; ">(</span><span style="color:#008000; ">0.2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>x<span style="color:#808030; ">)</span>                                        
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Layer # 9: Output layer</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># - Number of neurons: num_classes </span>
<span style="color:#696969; "># - Activation function: softwmax:</span>
<span style="color:#696969; ">#   - Suitable for multi-class classification.</span>
<span style="color:#696969; ">#------------------------------------------------------------------------------- </span>
x <span style="color:#808030; ">=</span> Dense<span style="color:#808030; ">(</span>num_classes<span style="color:#808030; ">,</span> activation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'softmax'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">(</span>x<span style="color:#808030; ">)</span>           
<span style="color:#696969; ">#-------------------------------------------------------------------------------          </span>
<span style="color:#696969; "># Create the model with above structure:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
model <span style="color:#808030; ">=</span> Model<span style="color:#808030; ">(</span>i<span style="color:#808030; ">,</span> x<span style="color:#808030; ">)</span>
</pre>


#### 4.3.2. Print the designed model summary:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># print the model summary</span>
model<span style="color:#808030; ">.</span>summary<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
</pre>


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;">Model<span style="color:#808030; ">:</span> <span style="color:#0000e6; ">"model"</span>
_________________________________________________________________
Layer <span style="color:#808030; ">(</span><span style="color:#400000; ">type</span><span style="color:#808030; ">)</span>                 Output Shape              Param <span style="color:#696969; ">#   </span>
<span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span>
input_2 <span style="color:#808030; ">(</span>InputLayer<span style="color:#808030; ">)</span>         <span style="color:#808030; ">[</span><span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span><span style="color:#808030; ">]</span>       <span style="color:#008c00; ">0</span>         
_________________________________________________________________
conv2d_3 <span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">)</span>            <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">13</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">13</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">)</span>        <span style="color:#008c00; ">320</span>       
_________________________________________________________________
conv2d_4 <span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">)</span>            <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">6</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">6</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#808030; ">)</span>          <span style="color:#008c00; ">18496</span>     
_________________________________________________________________
conv2d_5 <span style="color:#808030; ">(</span>Conv2D<span style="color:#808030; ">)</span>            <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#808030; ">)</span>         <span style="color:#008c00; ">73856</span>     
_________________________________________________________________
flatten_1 <span style="color:#808030; ">(</span>Flatten<span style="color:#808030; ">)</span>          <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">512</span><span style="color:#808030; ">)</span>               <span style="color:#008c00; ">0</span>         
_________________________________________________________________
dropout_2 <span style="color:#808030; ">(</span>Dropout<span style="color:#808030; ">)</span>          <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">512</span><span style="color:#808030; ">)</span>               <span style="color:#008c00; ">0</span>         
_________________________________________________________________
dense_1 <span style="color:#808030; ">(</span>Dense<span style="color:#808030; ">)</span>              <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">512</span><span style="color:#808030; ">)</span>               <span style="color:#008c00; ">262656</span>    
_________________________________________________________________
dropout_3 <span style="color:#808030; ">(</span>Dropout<span style="color:#808030; ">)</span>          <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">512</span><span style="color:#808030; ">)</span>               <span style="color:#008c00; ">0</span>         
_________________________________________________________________
dense_2 <span style="color:#808030; ">(</span>Dense<span style="color:#808030; ">)</span>              <span style="color:#808030; ">(</span><span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">10</span><span style="color:#808030; ">)</span>                <span style="color:#008c00; ">5130</span>      
<span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span>
Total params<span style="color:#808030; ">:</span> <span style="color:#008c00; ">360</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">458</span>
Trainable params<span style="color:#808030; ">:</span> <span style="color:#008c00; ">360</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">458</span>
Non<span style="color:#44aadd; ">-</span>trainable params<span style="color:#808030; ">:</span> <span style="color:#008c00; ">0</span>
_________________________________________________________________
</pre>

### 4.4. Part 4: Compile the CNN model

* Compile the CNN model, developed above:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Compile the model</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># experiment with the optimizer</span>
<span style="color:#696969; "># opt = SGD(lr=0.01, momentum=0.9)</span>
<span style="color:#696969; "># compile the model</span>
model<span style="color:#808030; ">.</span><span style="color:#400000; ">compile</span><span style="color:#808030; ">(</span>optimizer<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'adam'</span><span style="color:#808030; ">,</span>                       <span style="color:#696969; "># optimzer: adam</span>
              loss<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'sparse_categorical_crossentropy'</span><span style="color:#808030; ">,</span> <span style="color:#696969; "># used for multi-class models</span>
              metrics<span style="color:#808030; ">=</span><span style="color:#808030; ">[</span><span style="color:#0000e6; ">'accuracy'</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>                   <span style="color:#696969; "># performance evaluation metric</span>
</pre>


### 4.5. Part 5: Train/Fit the model:

* Start training the compiled CNN model:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Train/fit the model:</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
r <span style="color:#808030; ">=</span> model<span style="color:#808030; ">.</span>fit<span style="color:#808030; ">(</span>x_train<span style="color:#808030; ">,</span> y_train<span style="color:#808030; ">,</span> validation_data<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span>x_test<span style="color:#808030; ">,</span> y_test<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> epochs<span style="color:#808030; ">=</span><span style="color:#008c00; ">100</span><span style="color:#808030; ">)</span>


Epoch <span style="color:#008c00; ">1</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1875</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1875</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">30</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">16</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.1246</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9526</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.3747</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8960</span>
Epoch <span style="color:#008c00; ">2</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1875</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1875</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">30</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">16</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.1190</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9555</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.3917</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9009</span>
Epoch <span style="color:#008c00; ">3</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1875</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1875</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">28</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">15</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.1147</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9563</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.3773</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8982</span>
Epoch <span style="color:#008c00; ">4</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1875</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1875</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">29</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">15</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.1069</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9591</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.4193</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8992</span>
Epoch <span style="color:#008c00; ">5</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1875</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1875</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">28</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">15</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.1042</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9596</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.4038</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8966</span>
<span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span>
<span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span>
Epoch <span style="color:#008c00; ">95</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1875</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1875</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">28</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">15</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0509</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9840</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8433</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8945</span>
Epoch <span style="color:#008c00; ">96</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1875</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1875</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">28</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">15</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0467</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9857</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8165</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8963</span>
Epoch <span style="color:#008c00; ">97</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1875</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1875</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">28</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">15</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0509</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9842</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8469</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8994</span>
Epoch <span style="color:#008c00; ">98</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1875</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1875</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">28</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">15</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0519</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9847</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8615</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8966</span>
Epoch <span style="color:#008c00; ">99</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1875</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1875</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">28</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">15</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0545</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9837</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8267</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8981</span>
Epoch <span style="color:#008c00; ">100</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">100</span>
<span style="color:#008c00; ">1875</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">1875</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">=</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">28</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">15</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.0485</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.9853</span> <span style="color:#44aadd; ">-</span> val_loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8747</span> <span style="color:#44aadd; ">-</span> val_accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8993</span>
</pre>


### 4.6. Part 6: Evaluate the model

* Evaluate the trained CNN model on the test data using different evaluation metrics:
  * Loss function
  * Accuracy
  * Confusion matrix.

### 4.6.1. Loss function:

* Display the variations of the training and validation loss function with the number of epochs:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Plot loss per iteration</span>
<span style="color:#800000; font-weight:bold; ">import</span> matplotlib<span style="color:#808030; ">.</span>pyplot <span style="color:#800000; font-weight:bold; ">as</span> plt
plt<span style="color:#808030; ">.</span>plot<span style="color:#808030; ">(</span>result<span style="color:#808030; ">.</span>history<span style="color:#808030; ">[</span><span style="color:#0000e6; ">'loss'</span><span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> label<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'loss'</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>plot<span style="color:#808030; ">(</span>result<span style="color:#808030; ">.</span>history<span style="color:#808030; ">[</span><span style="color:#0000e6; ">'val_loss'</span><span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> label<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'val_loss'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>legend<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>xlabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Epoch Iteration'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>ylabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Loss'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
</pre>

<img src="images/loss-function.png" width = "1000"/>

### 4.6.2. Accuracy:

* Display the variations of the training and validation accuracy with the number of epochs:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Plot accuracy per iteration</span>
plt<span style="color:#808030; ">.</span>plot<span style="color:#808030; ">(</span>result<span style="color:#808030; ">.</span>history<span style="color:#808030; ">[</span><span style="color:#0000e6; ">'accuracy'</span><span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> label<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'acc'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>plot<span style="color:#808030; ">(</span>result<span style="color:#808030; ">.</span>history<span style="color:#808030; ">[</span><span style="color:#0000e6; ">'val_accuracy'</span><span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> label<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'val_acc'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>legend<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>xlabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Epoch Iteration'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
plt<span style="color:#808030; ">.</span>ylabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Accuracy'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
</pre>

<img src="images/accuracy-function.png" width = "1000"/>


#### 4.6.3. Compute the test-data Accuracy:

* Compute and display the accuracy on the test-data:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Compute the model accuracy on the test data</span>
accuracy_test_data <span style="color:#808030; ">=</span> model<span style="color:#808030; ">.</span>evaluate<span style="color:#808030; ">(</span>x_test<span style="color:#808030; ">,</span> y_test<span style="color:#808030; ">)</span>
<span style="color:#696969; "># display the atest-data accuracy</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'-------------------------------------------------------'</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'The test-data accuracy = '</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#808030; ">(</span>accuracy_test_data<span style="color:#808030; ">[</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'-------------------------------------------------------'</span><span style="color:#808030; ">)</span>

<span style="color:#008c00; ">313</span><span style="color:#44aadd; ">/</span><span style="color:#008c00; ">313</span> <span style="color:#808030; ">[</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#44aadd; ">==</span><span style="color:#808030; ">]</span> <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">1</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">s</span> <span style="color:#008c00; ">5</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">ms</span><span style="color:#44aadd; ">/</span>step <span style="color:#44aadd; ">-</span> loss<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8747</span> <span style="color:#44aadd; ">-</span> accuracy<span style="color:#808030; ">:</span> <span style="color:#008000; ">0.8993</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
The test<span style="color:#44aadd; ">-</span>data accuracy <span style="color:#808030; ">=</span> <span style="color:#008000; ">0.8992999792098999</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
</pre>


#### 4.6.4. Confusion Matrix Visualizations:

* Compute the confusion matrix:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Compute the confusion matrix</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#800000; font-weight:bold; ">def</span> plot_confusion_matrix<span style="color:#808030; ">(</span>cm<span style="color:#808030; ">,</span> classes<span style="color:#808030; ">,</span>
                          normalize<span style="color:#808030; ">=</span><span style="color:#074726; ">False</span><span style="color:#808030; ">,</span>
                          title<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'Confusion matrix'</span><span style="color:#808030; ">,</span>
                          cmap<span style="color:#808030; ">=</span>plt<span style="color:#808030; ">.</span>cm<span style="color:#808030; ">.</span>Blues<span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
  <span style="color:#696969; ">"""</span>
<span style="color:#696969; ">&nbsp;&nbsp;This function prints and plots the confusion matrix.</span>
<span style="color:#696969; ">&nbsp;&nbsp;Normalization can be applied by setting `normalize=True`.</span>
<span style="color:#696969; ">&nbsp;&nbsp;"""</span>
  <span style="color:#800000; font-weight:bold; ">if</span> normalize<span style="color:#808030; ">:</span>
      cm <span style="color:#808030; ">=</span> cm<span style="color:#808030; ">.</span>astype<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'float'</span><span style="color:#808030; ">)</span> <span style="color:#44aadd; ">/</span> cm<span style="color:#808030; ">.</span><span style="color:#400000; ">sum</span><span style="color:#808030; ">(</span>axis<span style="color:#808030; ">=</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span><span style="color:#808030; ">[</span><span style="color:#808030; ">:</span><span style="color:#808030; ">,</span> np<span style="color:#808030; ">.</span>newaxis<span style="color:#808030; ">]</span>
      <span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Normalized confusion matrix"</span><span style="color:#808030; ">)</span>
  <span style="color:#800000; font-weight:bold; ">else</span><span style="color:#808030; ">:</span>
      <span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Confusion matrix, without normalization'</span><span style="color:#808030; ">)</span>

  <span style="color:#696969; "># Display the confusuon matrix</span>
  <span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span>cm<span style="color:#808030; ">)</span>
  <span style="color:#696969; "># display the confusion matrix</span>
  plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>cm<span style="color:#808030; ">,</span> interpolation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'nearest'</span><span style="color:#808030; ">,</span> cmap<span style="color:#808030; ">=</span>cmap<span style="color:#808030; ">)</span>
  plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span>title<span style="color:#808030; ">)</span>
  plt<span style="color:#808030; ">.</span>colorbar<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
  tick_marks <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>arange<span style="color:#808030; ">(</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>classes<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
  plt<span style="color:#808030; ">.</span>xticks<span style="color:#808030; ">(</span>tick_marks<span style="color:#808030; ">,</span> classes<span style="color:#808030; ">,</span> rotation<span style="color:#808030; ">=</span><span style="color:#008c00; ">45</span><span style="color:#808030; ">)</span>
  plt<span style="color:#808030; ">.</span>yticks<span style="color:#808030; ">(</span>tick_marks<span style="color:#808030; ">,</span> classes<span style="color:#808030; ">)</span>
  
  fmt <span style="color:#808030; ">=</span> <span style="color:#0000e6; ">'.2f'</span> <span style="color:#800000; font-weight:bold; ">if</span> normalize <span style="color:#800000; font-weight:bold; ">else</span> <span style="color:#0000e6; ">'d'</span>
  thresh <span style="color:#808030; ">=</span> cm<span style="color:#808030; ">.</span><span style="color:#400000; ">max</span><span style="color:#808030; ">(</span><span style="color:#808030; ">)</span> <span style="color:#44aadd; ">/</span> <span style="color:#008000; ">2.</span>
  <span style="color:#800000; font-weight:bold; ">for</span> i<span style="color:#808030; ">,</span> j <span style="color:#800000; font-weight:bold; ">in</span> itertools<span style="color:#808030; ">.</span>product<span style="color:#808030; ">(</span><span style="color:#400000; ">range</span><span style="color:#808030; ">(</span>cm<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span>cm<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
      plt<span style="color:#808030; ">.</span>text<span style="color:#808030; ">(</span>j<span style="color:#808030; ">,</span> i<span style="color:#808030; ">,</span> format<span style="color:#808030; ">(</span>cm<span style="color:#808030; ">[</span>i<span style="color:#808030; ">,</span> j<span style="color:#808030; ">]</span><span style="color:#808030; ">,</span> fmt<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span>
               horizontalalignment<span style="color:#808030; ">=</span><span style="color:#0000e6; ">"center"</span><span style="color:#808030; ">,</span>
               color<span style="color:#808030; ">=</span><span style="color:#0000e6; ">"white"</span> <span style="color:#800000; font-weight:bold; ">if</span> cm<span style="color:#808030; ">[</span>i<span style="color:#808030; ">,</span> j<span style="color:#808030; ">]</span> <span style="color:#44aadd; ">&gt;</span> thresh <span style="color:#800000; font-weight:bold; ">else</span> <span style="color:#0000e6; ">"black"</span><span style="color:#808030; ">)</span>

  plt<span style="color:#808030; ">.</span>tight_layout<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
  plt<span style="color:#808030; ">.</span>ylabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'True label'</span><span style="color:#808030; ">)</span>
  plt<span style="color:#808030; ">.</span>xlabel<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Predicted label'</span><span style="color:#808030; ">)</span>
  plt<span style="color:#808030; ">.</span>show<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>

<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
<span style="color:#696969; "># Predict the targets for the test data</span>
<span style="color:#696969; ">#-------------------------------------------------------------------------------</span>
p_test <span style="color:#808030; ">=</span> model<span style="color:#808030; ">.</span>predict<span style="color:#808030; ">(</span>x_test<span style="color:#808030; ">)</span><span style="color:#808030; ">.</span>argmax<span style="color:#808030; ">(</span>axis<span style="color:#808030; ">=</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># construct the confusion matrix</span>
cm <span style="color:#808030; ">=</span> confusion_matrix<span style="color:#808030; ">(</span>y_test<span style="color:#808030; ">,</span> p_test<span style="color:#808030; ">)</span>
<span style="color:#696969; "># plot the confusion matrix</span>
plot_confusion_matrix<span style="color:#808030; ">(</span>cm<span style="color:#808030; ">,</span> <span style="color:#400000; ">list</span><span style="color:#808030; ">(</span><span style="color:#400000; ">range</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">10</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> 
                      <span style="color:#074726; ">False</span><span style="color:#808030; ">,</span> 
                      <span style="color:#0000e6; ">'Confusion matrix'</span><span style="color:#808030; ">,</span> 
                      plt<span style="color:#808030; ">.</span>cm<span style="color:#808030; ">.</span>Greens<span style="color:#808030; ">)</span>
</pre>


<img src="images/confusion-matrix.JPG" width = "1000"/>

#### 4.6.5. Examine some of the misclassified digits:

* Display some of the misclassified items:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># - Find the indices of all the mis-classified examples</span>
misclassified_idx <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>where<span style="color:#808030; ">(</span>p_test <span style="color:#44aadd; ">!=</span> y_test<span style="color:#808030; ">)</span><span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span> <span style="color:#696969; "># select the index</span>
<span style="color:#696969; "># setup the subplot grid for the visualized images</span>
 <span style="color:#696969; "># the suplot grid shape</span>
num_rows <span style="color:#808030; ">=</span> <span style="color:#008c00; ">5</span>
<span style="color:#696969; "># the number of columns</span>
num_cols <span style="color:#808030; ">=</span> num_visualized_images <span style="color:#44aadd; ">//</span> num_rows
<span style="color:#696969; "># setup the subplots axes</span>
fig<span style="color:#808030; ">,</span> axes <span style="color:#808030; ">=</span> plt<span style="color:#808030; ">.</span>subplots<span style="color:#808030; ">(</span>nrows<span style="color:#808030; ">=</span>num_rows<span style="color:#808030; ">,</span> ncols<span style="color:#808030; ">=</span>num_cols<span style="color:#808030; ">,</span> figsize<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">12</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># set a seed random number generator for reproducible results</span>
seed<span style="color:#808030; ">(</span>random_state_seed<span style="color:#808030; ">)</span>
<span style="color:#696969; "># iterate over the sub-plots</span>
<span style="color:#800000; font-weight:bold; ">for</span> row <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span>num_rows<span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
  <span style="color:#800000; font-weight:bold; ">for</span> col <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span>num_cols<span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
    <span style="color:#696969; "># get the next figure axis</span>
    ax <span style="color:#808030; ">=</span> axes<span style="color:#808030; ">[</span>row<span style="color:#808030; ">,</span> col<span style="color:#808030; ">]</span><span style="color:#808030; ">;</span>
    <span style="color:#696969; "># turn-off subplot axis</span>
    ax<span style="color:#808030; ">.</span>set_axis_off<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># select a random mis-classified example</span>
    counter <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>random<span style="color:#808030; ">.</span>choice<span style="color:#808030; ">(</span>misclassified_idx<span style="color:#808030; ">)</span>
    <span style="color:#696969; "># get test image </span>
    image <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>squeeze<span style="color:#808030; ">(</span>x_test<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">,</span><span style="color:#808030; ">:</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># get the true labels of the selected image</span>
    label <span style="color:#808030; ">=</span> labels<span style="color:#808030; ">[</span>y_test<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">]</span><span style="color:#808030; ">]</span>
    <span style="color:#696969; "># get the predicted label of the test image</span>
    yhat <span style="color:#808030; ">=</span> labels<span style="color:#808030; ">[</span>p_test<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">]</span><span style="color:#808030; ">]</span>
    <span style="color:#696969; "># display the image </span>
    ax<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>image<span style="color:#808030; ">,</span> cmap<span style="color:#808030; ">=</span>plt<span style="color:#808030; ">.</span>cm<span style="color:#808030; ">.</span>gray_r<span style="color:#808030; ">,</span> interpolation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'nearest'</span><span style="color:#808030; ">)</span>
    <span style="color:#696969; "># display the true and predicted labels on the title of tehe image</span>
    ax<span style="color:#808030; ">.</span>set_title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Y = %s, $</span><span style="color:#0f69ff; ">\h</span><span style="color:#0000e6; ">at{Y}$ = %s'</span> <span style="color:#44aadd; ">%</span> <span style="color:#808030; ">(</span><span style="color:#808030; ">(</span>label<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span>yhat<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> size <span style="color:#808030; ">=</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">)</span>
</pre>

<img src="images/25-mis-classified-images.png" width = "1000"/>

### 4.7. Part 7: Display a final message after successful execution completion:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># display a final message</span>
<span style="color:#696969; "># current time</span>
now <span style="color:#808030; ">=</span> datetime<span style="color:#808030; ">.</span>datetime<span style="color:#808030; ">.</span>now<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># display a message</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Program executed successfully on: '</span><span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#808030; ">(</span>now<span style="color:#808030; ">.</span>strftime<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"%Y-%m-%d %H:%M:%S"</span><span style="color:#808030; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#0000e6; ">"...Goodbye!</span><span style="color:#0f69ff; ">\n</span><span style="color:#0000e6; ">"</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>

Program executed successfully on<span style="color:#808030; ">:</span> <span style="color:#008c00; ">2021</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">04</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">05</span> <span style="color:#008c00; ">19</span><span style="color:#808030; ">:</span><span style="color:#008c00; ">42</span><span style="color:#808030; ">:</span><span style="color:#008000; ">07.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span>Goodbye!
</pre>


## 5. Analysis

* In view of the presented results, we make the following observations:

 * The simple designed CNN achieves reasonably high accuracy (90%) of the FASHION-MNIST data classification.
 * The few misclassifications appear reasonable:
 * It is reasonable to confuse shirt with coat
 * It is reasonable to confuse shirt with dress
 * It is reasonable to confuse shirt with T-shirt/top.

## 6. Future Work

* We plan to explore the following related issues:

 * To explore ways of improving the performance of this simple CNN, including fine-tuning the following hyper-parameters:
   * The validation loss function is increasing while the training loss function is decreasing:
   * This indicates over-fitting
 * We shall address this over-fitting by adding dropout and batch normalization layers.
 * We shall also explore fine-tuning some of the hyper-parameters, including:
   * The number of filters and layers
   * The dropout rate
   * The optimizer
   * The learning rate.

## 7. References

1. Fashion-MNIST. Fashion-MNIST. Retrieved from: https://github.com/zalandoresearch/fashion-mnist. 
2. Jason Brownlee. Deep Learning CNN for Fashion-MNIST Clothing Classification. Retrieved from: https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-fashion-mnist-clothing-classification/. 
3. Manish Bhobé. Classifying Fashion with a Keras CNN (achieving 94% accuracy) — Part 1. Retrieved from: https://medium.com/@mjbhobe/classifying-fashion-with-a-keras-cnn-achieving-94-accuracy-part-1-1ffcb7e5f61a. 
4. Gabriel Preda. CNN with Tensorflow|Keras for Fashion MNIST. Retrieved from: https://www.kaggle.com/gpreda/cnn-with-tensorflow-keras-for-fashion-mnist. 
5. Adrian Rosebrock. Fashion MNIST with Keras and Deep Learning. Retrieved from: https://www.pyimagesearch.com/2019/02/11/fashion-mnist-with-keras-and-deep-learning/. 
6. Deepak Singh. Fashion-MNIST using Deep Learning with TensorFlow Keras. Retrieved from: https://cloudxlab.com/blog/fashion-mnist-using-deep-learning-with-tensorflow-keras/. 
7. Yue Zhang. Evaluation of CNN Models with Fashion MNIST Data. Retrieved from: https://lib.dr.iastate.edu/cgi/viewcontent.cgi?article=1402&context=creativecomponents. 
