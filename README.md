# Leaf_Recognition_RealLifeProblem

# Install
This project files requires Python 3 and the following Python libraries installed:
* Open CV - https://opencv.org/  
* Numpy - http://numpy.org/
* Keras - http://keras.io/

# Problem 
The leaf recognition is one of the most challenging problems in computer vision because of almost similar shape, texture and color of the leaves.There has been a lot of work done using various traditional machine learning algorithm and hand-made leaves features. 
But in this challenge, I tried to make an end-to-end deep learning model for multi-class leaf recognition from scratch.

# Challenges
Looking at the dataset the images were highly augmented and was seemingly captured from different orientations.

# About
* Loading of the dataset which contains train and test folders which contains images of 185 different classes.
* Stacking of all the images using their path by os for both train and test data.
* Getting features and labels in this created data for both train and test.
* Running the function inception_block and DeceptiNet.
* Compiling and fitting of our model to get the scores.

# How to use
Concatenate all the python files and run the code.

# Result
At an epoch of 50 and a batch size of 32, the model gives an accuracy of 86.45% on the test set.


