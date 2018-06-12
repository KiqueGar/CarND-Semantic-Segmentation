# Semantic Segmentation

## Introduction
The goal of this project is to implement a convolutional neural network to perform semantic segmentation on road images.
The project is trained and tested over the KITTI dataset and implemented over the VGG-16 architecture

## Architecture

A pre-trained VGG-16 network was converted to a fully convolutional network, converting it's fully connected layer into
1x1 convolution and upsampling to it's previous layers.
Only 2 categories are present: road and not road

- Loss function is cross entropy
- Adam optimizer is used
- Dropout is 0.5
- Learning rate starts at 0.001
- Batch size = 1
- Epochs = 50

An Amazon EC2 g2.8xlarge instance was used for training, however, by memory constraints, batch size is defined as 1

Loss is decaying as expected, it might seem that it's tarting to rise after epoch 40
A first training was made wich seems to converge around epoch 40, fromwhere it starts to rise.
Another training was made using a g3.4xlarge instance, a lower learning rate (0.00003) and higher dropout rate (0.75)
helped to counterbalance for a higher batch size (8 vs 2)

Below we can see the comparisson between 2 training runs, we see a almost the same steady value, however, we see a smoother
loss function with a lower learning rate (also, initial error is bigger)

![Loss plot](./Loss_plot.png)

## Results
The roa is mostly correctly classified as driveable, also, the model performs as expeted on some of the videos propossed


![Sample1](./runs/1528510779.656766/uu_000056.png)
![Sample2](./runs/1528510779.656766/umm_000085.png)
![Sample3](./runs/1528510779.656766/um_000078.png)

Original Udacity Readme ahead
---

### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
