from PIL import Image
import numpy as np
import dhash

def images_to_matrix(trainX, testX):
    def reshape(_3Ddata):
        new_shape = ( _3Ddata.shape[0], _3Ddata.shape[1] * _3Ddata.shape[2])
        _2Data = np.reshape (_3Ddata, new_shape )
        return _2Data

    trainX =  reshape(trainX)
    testX =   reshape(testX)

    return trainX, testX

def Images_to_DHash(_3DImages, size = 8):
    # define format
    format_ = '0' + str(size**2) + 'b'
    X = _3DImages
    #preallocate
    X_hashed = np.zeros((X.shape[0], size**2 * 2));
    for idx , Img in enumerate(X):
        row, col = dhash.dhash_row_col( Image.fromarray(Img) , size = size)
        hash_ = format(row, format_) + format(col, format_)
    # hash_ is string
        for colidx,num in enumerate(hash_):
            X_hashed[idx,colidx] = int(num)

    return X_hashed
