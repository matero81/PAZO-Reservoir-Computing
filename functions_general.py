import matplotlib.pyplot as plt

import os # Used for directories
import pandas as pd # Used for Dataframes and opening and saving csv files
import numpy as np # Used for scientific computation
import random # Used to generate random numbers
import shutil # Used to manage files
import imageio # Used to save a gif
from PIL import Image # Used to save images
from IPython.display import clear_output # Used to clear the output of a cell
from tqdm import tqdm # Used to show a progress bar

# Used for for the feedfodward neural network:
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

def import_mnist_offline(file_path):
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        print("Make sure the file is in the correct location or provide the full path.")
    else:
        with np.load(file_path, allow_pickle=True) as data:
            x_train = data['x_train']
            y_train = data['y_train']
            x_test = data['x_test']
            y_test = data['y_test']
    return (x_train, y_train), (x_test, y_test)

def value_at_time_x_numpy(single_data, time):
    time_vector = single_data[:, 0].tolist()
    value_vector = single_data[:, 1].tolist()
    closest_x_value = min(time_vector, key=lambda x: abs(time - x))
    idx_closest_x_value = time_vector.index(closest_x_value)
    return value_vector[idx_closest_x_value], time_vector[idx_closest_x_value], idx_closest_x_value

def find_nearest_value_index(array, target_value):
    index = (np.abs(array-target_value)).argmin()
    return index

## Function to assign the digit binary data to the PSI
# selector_reservoir:
# 0: Random selection from the PSI data
# 1: Last binary value for static reservoir
# 2: Binary to decimal conversion
# 3: mean of the PSI data of all the sets
# 4: Assign None to every row
def assign_experimental_data_with_digit(data_at_specific_time, digit_rows, num_digits_train, num_digits_test, digit_train, digit_test, selector_reservoir = 0):
    digit_cols_reduced = 1
    digit_train_reduced = [np.zeros(shape=(digit_rows, digit_cols_reduced)) for i in range(0, num_digits_train)]
    digit_test_reduced = [np.zeros(shape=(digit_rows, digit_cols_reduced)) for i in range(0, num_digits_test)]
    for idx_digit, single_digit in tqdm(enumerate(digit_train), desc='Training set', total=num_digits_train):
        for idx_row, single_row in enumerate(single_digit):
            eq_binary_number = ''.join([str(int(x)) for x in single_row])
            if selector_reservoir == 0:
                digit_train_reduced[idx_digit][idx_row] = random.choice(data_at_specific_time[eq_binary_number])
            elif selector_reservoir == 1:
                digit_train_reduced[idx_digit][idx_row] = single_row[-1]
            elif selector_reservoir == 2:
                digit_train_reduced[idx_digit][idx_row] = int(eq_binary_number, 2)
            elif selector_reservoir == 3:
                digit_train_reduced[idx_digit][idx_row] = np.mean(data_at_specific_time[eq_binary_number])
            else:
                digit_train_reduced[idx_digit][idx_row] = None

    for idx_digit, single_digit in tqdm(enumerate(digit_test), desc='Testing set', total=num_digits_test):
        for idx_row, single_row in enumerate(single_digit):
            eq_binary_number = ''.join([str(int(x)) for x in single_row])
            if selector_reservoir == 0:
                digit_test_reduced[idx_digit][idx_row] = random.choice(data_at_specific_time[eq_binary_number])
            elif selector_reservoir == 1:
                digit_test_reduced[idx_digit][idx_row] = single_row[-1]
            elif selector_reservoir == 2:
                digit_test_reduced[idx_digit][idx_row] = int(eq_binary_number, 2)
            elif selector_reservoir == 3:
                digit_test_reduced[idx_digit][idx_row] = np.mean(data_at_specific_time[eq_binary_number])
            else:
                digit_test_reduced[idx_digit][idx_row] = None

    digit_train_reduced_np = np.array(digit_train_reduced).reshape(num_digits_train, digit_rows)
    digit_test_reduced_np = np.array(digit_test_reduced).reshape(num_digits_test, digit_rows)
    return digit_train_reduced_np, digit_test_reduced_np


def perform_training(start_train, start_test, num_digits_train, num_digits_test, digit_train_class, digit_test_class, epochs, batch_size, digit_train_reduced_np, digit_test_reduced_np, verbose_training):
    
    
    hist_train = digit_train_reduced_np
    hist_test = digit_test_reduced_np

    train_in = hist_train[start_train:start_train+num_digits_train,:]
    test_in = hist_test[start_test:start_test+num_digits_test,:]
    
    train_out = np.reshape(digit_train_class[start_train:start_train+num_digits_train], (num_digits_train,1))
    test_out = np.reshape(digit_test_class[start_test:start_test+num_digits_test], (num_digits_test,1))
    
    sc = StandardScaler()
    ohe = OneHotEncoder()
    
    scaler = sc.fit(train_in)
    train_in = scaler.transform(train_in)
    
    train_out = ohe.fit_transform(train_out).toarray()
    
    model = Sequential()
    model.add(Dense(10, activation='softmax'))
    #model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Put verbose at 0 to avoid the printing of the epochs while training
    history = model.fit(train_in, train_out, epochs=epochs, batch_size=batch_size, verbose=verbose_training)
    
    test_in = scaler.transform(test_in)
    for layer in model.layers:
        weights = layer.get_weights()
    
    
    y_pred = model.predict(test_in)    
    predicted = 0
    pred = list()
    for i in range(len(y_pred)):
        pred.append(np.argmax(y_pred[i]))
        if np.argmax(y_pred[i]) == test_out[i]:
            predicted += 1
    test_accuracy = predicted/len(y_pred)
    
    conf_matrix_norm=confusion_matrix(test_out, pred, normalize='true')
    conf_matrix=confusion_matrix(test_out, pred, normalize=None)

    y_pred_class = [np.argmax(y_pred[x]) for x in range(len(y_pred))]

    return history, conf_matrix, conf_matrix_norm, test_accuracy, y_pred_class, test_out

def plot_training_data(out_dir_training_outputs, history, conf_matrix, conf_matrix_norm, repetition):
    plt.figure()
    plt.plot(history.history['accuracy'], 'b', linewidth=2)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1.1])
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['Train'], loc='upper left')
    plt.show()
    plt.savefig(out_dir_training_outputs+f'model_accuracy_rep{repetition}.png', format="png", dpi=1200)
    np.savetxt(out_dir_training_outputs+f'model_accuracy_rep{repetition}.txt', history.history['accuracy'])
    
    plt.figure()
    plt.plot(history.history['loss'], 'b', linewidth=2)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['Train'], loc='upper left')
    plt.show()
    plt.savefig(out_dir_training_outputs+f'model_loss_rep{repetition}.png', format="png", dpi=1200)
    np.savetxt(out_dir_training_outputs+f'model_loss_rep{repetition}.txt', history.history['loss'])

    plt.figure(figsize=(18,9))
    plt.title('Confusion Matrix', fontsize=25)
    plt.imshow(conf_matrix_norm, cmap='Blues')
    plt.colorbar(label='%')
    plt.xlabel('PREDICTED', fontsize=20)
    plt.ylabel('TRUE', fontsize=20)
    plt.xticks(range(10), fontsize = 15)
    plt.yticks(range(10), fontsize = 15, rotation=90)
    manager = plt.get_current_fig_manager()
    #manager.window.showMaximized()
    for i in range(10):
        for j in range(10):
            if round(conf_matrix[i, j],0) != 0:
                text = plt.text(j, i, round(conf_matrix[i, j],0),
                                ha="center", va="center", color="w", fontsize=15)
    plt.show()
    plt.savefig(out_dir_training_outputs+f'conf_matrix_rep{repetition}.png', format="png", dpi=1200)