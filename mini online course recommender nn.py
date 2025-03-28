import numpy as np

# Sigmoid activation function
#maps any value to a value between 0 and 1. We use it to convert numbers to probabilities
#One of the desirable properties of a sigmoid function is that its output can be used to create its derivative.
# If the sigmoid's output is a variable "out"
#then the derivative is simply out * (1-out). This is very efficient.
#just think about the deriv. as the slope of the sigmoid function at a given point

def nonlin(x, deriv=False):   # this means that by default, deriv will be False unless someone specifically passes True when calling this function.
    if deriv:                 # this is asking:“Did the user ask for the derivative instead of the regular sigmoid function?” If yes, it returns the derivative of the sigmoid; derivative of sigmoid is for backpropagation (learning mechanism an NN uses to adjust its weights based on how wrong its predictions were)
        return x * (1 - x)  # Derivative for backpropagation
    return 1 / (1 + np.exp(-x))  # Sigmoid: squashes values between 0 and 1


# input dataset
# Each row is a course: [project based, videos, certificate, short term, tools, free]
X = np.array([
    [1, 1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0, 0],
    [1, 1, 0, 1, 1, 1],
    [0, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 0],
])

# output dataset (user preferences)
# 1 = user would enroll, 0 = would not
y = np.array([[1, 0, 1, 1, 0, 0]]).T

# seed random numbers to make results consistent
np.random.seed(1)

# initialize weights for 6 input features -> 1 output neuron
syn0 = 2 * np.random.random((6, 1)) - 1 #Create a 6x1 column matrix of random numbers between -1 & 1 and store it in syn0 (weight matrix)
                                        #using the random() function from the random module of NumPy


# training loop
for iter in range(10000):

    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))  # prediction
    # l0 is just the input layer.
    # np.dot(l0, syn0) performs matrix multiplication of input X with weights syn0.
    # nonlin(...) applies the sigmoid activation to this result.
    # Resulting l1 is the network's prediction/output.

    # calculate error
    l1_error = y - l1

    # backpropagation step: adjust weights
    l1_delta = l1_error * nonlin(l1, True)  #Gradient = Error × Sensitivity
                                                  #nonlin(l1, True) → the derivative of sigmoid
                                                  #which tells us how sensitive the output is to changes in weights

    # update weights
    syn0 += np.dot(l0.T, l1_delta)   #New weights = old weights + amount to shift


# final output after training
print("Course Preference Predictions:\n")
for i, output in enumerate(l1): #enumerate(l1) gives you-> i: the index (course number) and output: the actual predicted value (e.g. 0.87)
    label = "User would enroll" if output > 0.5 else "User would not enroll"
    print(f"Course {i+1}: {output[0]:.4f} -> {label}") #i+1 makes the index human-friendly (starting from 1 instead of 0)

# Calculate training accuracy
correct = 0
for i in range(len(y)):
    predicted_label = 1 if l1[i][0] > 0.5 else 0
    actual_label = y[i][0]
    if predicted_label == actual_label:
        correct += 1

accuracy = correct / len(y) * 100
print(f"\nTraining Accuracy: {accuracy:.2f}%")


# predict function for new courses
def predict_course(features):
    """
    features: list of 6 binary values [project based, videos, certificate, short term, tools, free]
    returns: prediction + label
    """
    input_data = np.array(features).reshape(1, 6) #.reshape(1, 6) turns it into the shape expected by the neural network:
                                                  #1 row (one course), 6 columns (6 features)

    prediction = nonlin(np.dot(input_data, syn0))[0][0] #[0][0] is used to pull the raw number out of the 2D array
    label = "User would enroll" if prediction > 0.5 else "User would not enroll"
    print(f"\nNew Course Prediction -> {prediction:.4f} -> {label}")


# trying with a new course
predict_course([1, 1, 1, 1, 0, 0])  # strong but not free/tools
predict_course([0, 0, 0, 1, 1, 1])  # short/tools/free but no videos/project

# Test Set Evaluation
X_test = np.array([
    [1, 1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0, 0],
    [1, 1, 0, 1, 1, 1],
    [0, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 1],
    [0, 0, 0, 1, 1, 1],
    [1, 1, 0, 0, 1, 0],
])

y_test = np.array([[1, 0, 1, 1, 0, 1, 0]]).T


print("\nTest Set Predictions:")
predictions = nonlin(np.dot(X_test, syn0))
correct = 0
for i in range(len(y_test)):
    predicted_label = 1 if predictions[i][0] > 0.5 else 0
    actual_label = y_test[i][0]
    label_text = "User would enroll" if predicted_label == 1 else "User would not enroll"
    print(f"Test Course {i+1}: {predictions[i][0]:.4f} -> {label_text}")
    if predicted_label == actual_label:
        correct += 1

test_accuracy = correct / len(y_test) * 100
print(f"\nTest Accuracy: {test_accuracy:.2f}%")
