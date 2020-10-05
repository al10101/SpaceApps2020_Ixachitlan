'''
Author: Alberto Cabrera
Contact: albertocabja@gmail.com
'''

import os
from PIL import Image
import numpy as np

def extract(img_file):
    
    # open image 
    image = Image.open(img_file)
    pix = image.load()
    
    # store the image dimensions
    rows, columns = image.size[0], image.size[1]
    print(' Input image dims: {0} x {1}'.format(rows, columns))
    
    X_i = np.zeros([3, 40000])
    x = 0
    
    for i in range(rows):
        
        for j in range(columns):
            
            pixel = np.array([pix[i, j]]).T
            X_i[:, x] += pixel[:, 0]
            x += 1
    
    return X_i

def extract_y(img_file):
    
    # open image and convert to grayscale
    image = Image.open(img_file)
    pix = image.load()
    
    # store the image dimensions
    rows, columns = image.size[0], image.size[1]
    print(' Input image dims: {0} x {1}'.format(rows, columns))
    
    y = np.zeros(40000)
    x = 0
    
    for i in range(rows):
        
        for j in range(columns):
            
            a, b, c, d = pix[i, j]
            
            if a + b + c != 0 :
                y[x] += 1
            else:
                pass
            x += 1
            
    return y
    
    
def main():
    
    # Total
    X = np.zeros([18, 40000])
    
    # Read the images and transfrom to vector
    i = 0
    print('\n Getting X data...\n')
    for img_file in sorted(os.listdir('.')):
        
        if 'RESIZE' in img_file and 'dist' not in img_file:
            
            # Open file and extract data
            print(' FILE: {0}'.format(img_file))
            X_i = extract(img_file)
            
            # Add to X matrix
            print(' Adding X[{0}:{1}, :]\n'.format(3*i,3*i+3))
            X[3*i:3*i+3, :] += X_i
            i += 1
            
    np.savetxt('X.csv', X.T, delimiter=',')
    
    y_file = 'dist5_RESIZE.png'
    print(' Getting y data...')
    print(' FILE: {0}'.format(y_file))
    y = extract_y(y_file)
    np.savetxt('y.csv', y, delimiter=',')
    

if __name__ == '__main__':
    main()
    