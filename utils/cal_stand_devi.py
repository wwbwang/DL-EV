import numpy as np

def cal_stand_devi(img, img_matrix):
    count = img_matrix.shape[0]
    stand_matrix = np.zeros((img_matrix.shape[1], img_matrix.shape[2]))
    
    # cal every pixel's standard deviation
    for i in range(img_matrix.shape[1]):
        for j in range(img_matrix.shape[2]):
            stand_matrix[i][j] = np.std(img_matrix[:,i,j])

    return stand_matrix