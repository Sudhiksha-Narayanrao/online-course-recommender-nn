# Online Course Recommender Neural Network (NumPy)

This project implements a basic neural network from scratch using only NumPy. The network is trained to predict whether a user would enroll in an online course based on six binary features. The goal was to understand the core concepts of forward propagation, backpropagation, and weight updates using gradient descent.


## Project Overview

The neural network consists of:
- A single input layer with 6 binary features
- One output neuron with a sigmoid activation function
- Manual backpropagation (no ML libraries used)
- A prediction function for new course inputs
- Training and test accuracy evaluation


## Features Used

Each course is represented by the following binary features (1 = yes, 0 = no):

| Feature Index | Feature Name        | Description                                      |
|---------------|---------------------|--------------------------------------------------|
| 0             | `is_project_based`  | Course includes a project                        |
| 1             | `has_videos`        | Includes video content                           |
| 2             | `includes_certification` | Offers a certificate                        |
| 3             | `is_short_term`     | Duration less than 4 weeks                       |
| 4             | `focuses_on_tools`  | Focus on practical tools instead of heavy theory |
| 5             | `is_free`           | Course is completely free                        |



## Training Data

Example training inputs:
[1, 1, 1, 1, 1, 1] → 1

[0, 0, 1, 0, 0, 0] → 0

[1, 1, 0, 1, 1, 1] → 1

[0, 1, 0, 1, 1, 1] → 1

[1, 1, 1, 0, 0, 1] → 0

[0, 0, 0, 0, 0, 0] → 0


## Test Data

To evaluate the model, 7 test cases were used, and the model achieved approximately 85% accuracy.

Example test inputs:
[1, 1, 1, 1, 1, 1] → 1

[0, 0, 1, 0, 0, 0] → 0

[1, 1, 0, 1, 1, 1] → 1

[0, 1, 0, 1, 1, 1] → 1

[1, 1, 1, 0, 0, 1] → 0

[0, 0, 0, 1, 1, 1] → 1

[1, 1, 0, 0, 1, 0] → 0


The script will:

-Train the model on the dataset

-Print predictions for each course

-Evaluate training accuracy

-Predict on a test set

-Show test set accuracy


