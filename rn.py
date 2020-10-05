'''
Author: Alberto Cabrera
Contact: albertocabja@gmail.com
Train neural network using X data and y data
'''

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

def main():
    
    # Load variables
    X = np.loadtxt('X.csv', delimiter=',')
    y = np.loadtxt('y.csv', delimiter=',')
    
    model = Sequential()
    model.add(Dense(500, activation='sigmoid', input_dim=18))
    model.add(Dense(500, activation='sigmoid'))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='nadam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    # Built thr model
    history = model.fit(X, y, epochs=20, batch_size=1000)
    
    # Plot history
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.ylabel('J')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

if __name__ == '__main__':
    main()
    