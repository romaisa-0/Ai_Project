from mnist import MNIST
from tensorflow.keras.datasets import mnist
import random
import numpy as np
import matplotlib.pyplot as plt
import operator
from PIL import Image

# Load MNIST dataset
(images_train, labels_train), (images_test, labels_test) = mnist.load_data()

# Print the shape of the dataset
print(f"Training images shape: {images_train.shape}")
print(f"Training labels shape: {labels_train.shape}")
print(f"Test images shape: {images_test.shape}")
print(f"Test labels shape: {labels_test.shape}")

# Normalize images
images_train = np.array(images_train) / 127.5 - 1
images_test = np.array(images_test) / 127.5 - 1

# Print the shape after normalization
print(f"Normalized training images shape: {images_train.shape}")
print(f"Normalized test images shape: {images_test.shape}")

# Flatten the images from 28x28 to 1D arrays of 784 elements
images_train = images_train.reshape((60000, 28*28))
images_test = images_test.reshape((10000, 28*28))

# Print the shape after flattening
print(f"Flattened training images shape: {images_train.shape}")
print(f"Flattened test images shape: {images_test.shape}")

# Add bias term (column of ones)
images_train = np.hstack((np.ones((60000, 1)), images_train))
images_test = np.hstack((np.ones((10000, 1)), images_test))

# Print the shape after adding bias term
print(f"Training images shape with bias: {images_train.shape}")
print(f"Test images shape with bias: {images_test.shape}")

# Split data into training and validation sets
Images_train = images_train[:48000]
lable_train = labels_train[:48000]
Images_val = images_train[-12000:]
Labels_val = labels_train[-12000:]

# Print the shape of training and validation data
print(f"Training set shape: {Images_train.shape}")
print(f"Validation set shape: {Images_val.shape}")

# Print the shape of training data
print(np.shape(Images_train))  # Output should be (48000, 785)

# One-hot encoding function
def one_hot_enc(data):
    sz = len(data)
    enc = np.zeros((sz, 10))
    enc[range(sz), np.array(data)] = 1
    return enc

# One-hot encode the labels
label_train_oh = one_hot_enc(lable_train)
label_val_oh = one_hot_enc(Labels_val)
label_test_oh = one_hot_enc(labels_test)

# Print the shape of one-hot encoded labels
print(f"One-hot encoded training labels shape: {label_train_oh.shape}")
print(f"One-hot encoded validation labels shape: {label_val_oh.shape}")
print(f"One-hot encoded test labels shape: {label_test_oh.shape}")

# Sigmoid activation function
def Sigmoid(X, tan):
    if tan == 0:
        return 1.0 / (1 + np.exp(-X))
    else:
        return 1.7159 * np.tanh((2 / 3) * X)

# Derivative of Sigmoid function
def SigmoidP(X, tan):
    if tan == 0:
        return Sigmoid(X, tan) * (1 - Sigmoid(X, tan))
    else:
        return 1.7159 * (2 / 3) * (1 - (np.tanh((2 / 3) * X)) ** 2)

# Softmax function for output layer probabilities
def get_softmax_prob(Input_O):
    ex = np.exp(Input_O)
    temp = np.sum(ex, axis=1)
    v = ex / np.resize(temp, (temp.shape[0], 1))
    return v

# Softmax prediction function
def get_softmax_pred(sm_prob):
    for i in range(len(sm_prob)):
        l = sm_prob[i]
        index, value = max(enumerate(l), key=operator.itemgetter(1))
        sm_prob[i] = 0
        sm_prob[i][index] = 1
    return sm_prob

# Feedforward function
def feedforward(WIH, WHO, Images_train, batch_size, tan):
    N = batch_size
    Input_H = np.dot(Images_train, WIH.T)
    Activation_H = Sigmoid(Input_H, tan)
    Activation_H = np.hstack((np.ones((N, 1)), Activation_H))
    Input_O = np.dot(Activation_H, WHO)
    Softmax_Input_O = get_softmax_prob(Input_O)
    return Input_H, Activation_H, Input_O, Softmax_Input_O

# Backpropagation function
def backprop(Images_train, label_train_oh, Input_H, Activation_H, Input_O, Softmax_Input_O, WIH, WHO, l1, l2, batch_size, tan, momentum, M1, M2):
    N = batch_size
    Y_N = Softmax_Input_O
    diff = label_train_oh - Y_N
    gd_O = np.dot(diff.T, Activation_H)
    D_new = 0
    U2_new = 0
    if momentum == 1:
        D_old = M1
        Update = D_old * 0.9 - (l2 * gd_O.T) / N
        WHO = WHO - Update
        D_new = Update
    else:
        WHO = WHO + (l2 * gd_O.T) / N
    who_wb = WHO
    dot1 = np.dot(diff, who_wb.T)
    Ac_T = Activation_H.T
    Acativation_h_wb = Ac_T
    a_sigmoid = SigmoidP(Acativation_h_wb, tan)
    dot2 = np.multiply(dot1, a_sigmoid.T)
    dot3 = np.dot(dot2.T, Images_train)
    if momentum == 1:
        D_old2 = M2
        UPDATE2 = D_old2 * 0.9 - (l1 * dot3[1:]) / N
        WIH = WIH - UPDATE2
        U2_new = UPDATE2
    else:
        WIH = WIH + (l1 * dot3[1:]) / N
    return WIH, WHO, D_new, U2_new

# Function to calculate hidden and output layers
def calOutputHidden(WIH, WHO, Images_val, tan):
    Input_H = np.dot(Images_val, WIH.T)
    Activation_H = Sigmoid(Input_H, tan)
    Activation_H = np.hstack((np.ones((len(Images_val), 1)), Activation_H))
    Input_O = np.dot(Activation_H, WHO)
    Softmax_Input_O = get_softmax_prob(Input_O)
    return Softmax_Input_O

# Function to calculate loss and accuracy
def calculate_loss_and_acc(Y_N, Images_val, label_val_oh):
    err = 0
    acc = []
    for i in range(len(Images_val)):
        y_n = Y_N[i]
        t_n = label_val_oh[i]
        dot = np.dot(np.log(y_n), t_n)
        err -= dot
        idx1 = np.argmax(y_n)
        idx2 = np.argmax(t_n)
        if idx1 == idx2:
            acc.append(1)
        else:
            acc.append(0)
    acc_f = (1.0 * sum(acc)) / len(label_val_oh)
    return err / len(label_val_oh), acc_f

# Main function to train the network
def Main(Images_train, label_train_oh, Images_val, label_val_oh, images_test, label_test_oh, H, l1, l2, epochs, shuffle1, momentum, tan, weight):
    H1 = H + 1
    if weight == 1:
        WIH = np.random.normal(0, (1 / np.sqrt(785)), (H, 785))
        WHO = np.random.normal(0, (1 / np.sqrt(785)), (H1, 10))
    else:
        WIH = np.random.normal(0, 0.00001, (H, 785))
        WHO = np.random.normal(0, 0.00001, (H1, 10))
    
    batch_size = 128
    batch_no = int(len(Images_train) / 128)
    acc_train, loss_train, acc_val, loss_val, acc_test, loss_test, epoch = [], [], [], [], [], [], []
    M1 = 0
    M2 = 0
    
    for i in range(epochs):
        if shuffle1 == 1:
            Train_img_shuffle = []
            Train_label_shuffle = []
            shuffle = np.random.permutation(len(Images_train))
            for sample in shuffle:
                Train_img_shuffle.append(Images_train[sample])
                Train_label_shuffle.append(label_train_oh[sample])
        
        for j in range(batch_no):
            start = j * 128
            end = start + 128
            if shuffle1 == 1:
                Images_T = Train_img_shuffle[start:end]
                Lables_T = Train_label_shuffle[start:end]
            else:
                Images_T = Images_train[start:end]
                Lables_T = label_train_oh[start:end]
            
            Input_H, Activation_H, Input_O, Softmax_Input_O = feedforward(WIH, WHO, Images_T, batch_size, tan)
            WIH_n, WHO_n, M1_n, M2_n = backprop(Images_T, Lables_T, Input_H, Activation_H, Input_O, Softmax_Input_O, WIH, WHO, l1, l2, batch_size, tan, momentum, M1, M2)
            WIH, WHO, M1, M2 = WIH_n, WHO_n, M1_n, M2_n
            
            # Print intermediate results
            print(f"Epoch {i}, Batch {j}")
            print(f"WIH shape: {WIH.shape}, WHO shape: {WHO.shape}")
            print(f"Input_H shape: {Input_H.shape}, Activation_H shape: {Activation_H.shape}")
            print(f"Input_O shape: {Input_O.shape}, Softmax_Input_O shape: {Softmax_Input_O.shape}")
        
        # Train accuracy and loss
        Y_N_train = calOutputHidden(WIH, WHO, Images_train, tan)
        err_t, acc_t = calculate_loss_and_acc(Y_N_train, Images_train, label_train_oh)
        acc_train.append(acc_t)
        loss_train.append(err_t)
        
        # Validation accuracy and loss
        Y_N = calOutputHidden(WIH, WHO, Images_val, tan)
        err, acc1 = calculate_loss_and_acc(Y_N, Images_val, label_val_oh)
        acc_val.append(acc1)
        loss_val.append(err)
        
        # Testing accuracy and loss
        Y_N_test = calOutputHidden(WIH, WHO, images_test, tan)
        err_test, acc_test1 = calculate_loss_and_acc(Y_N_test, images_test, label_test_oh)
        acc_test.append(acc_test1)
        loss_test.append(err_test)
        
        epoch.append(i)
        print(f"Epoch {i} - Training Loss: {err_t}, Training Accuracy: {acc_t}, Validation Accuracy: {acc1}, Test Accuracy: {acc_test1}")
    
    return epoch, acc_train, acc_val, loss_train, loss_val, WIH, WHO

# --- Custom Image Testing Functionality ---

# Function to load and preprocess a custom image
def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Load and convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img_array = np.array(img) / 127.5 - 1  # Normalize
    img_array = img_array.reshape(1, 28*28)  # Flatten
    img_array = np.hstack((np.ones((1, 1)), img_array))  # Add bias
    return img_array

# Function to test a custom image using the trained model
def test_custom_image(image_path, WIH, WHO, tan):
    img = load_and_preprocess_image(image_path)
    print(f"Custom image shape after preprocessing: {img.shape}")
    Y_N = calOutputHidden(WIH, WHO, img, tan)
    print(f"Output probabilities: {Y_N}")
    predicted_digit = np.argmax(Y_N)
    print(f"Predicted digit for the input image: {predicted_digit}")
    return predicted_digit

# --- End Custom Image Testing Functionality ---

if __name__ == "__main__":
    # Training the model
    epochs, acc_train, acc_val, loss_train, loss_val, WIH, WHO = Main(
        Images_train, label_train_oh, Images_val, label_val_oh, 
        images_test, label_test_oh, H=128, l1=0.01, l2=0.01, 
        epochs=10, shuffle1=1, momentum=1, tan=1, weight=1
    )
    
    # Test the model with a custom image
    custom_image_path = './images/eight.png'  # Raw string
    predicted_digit = test_custom_image(custom_image_path, WIH, WHO, tan=1)
