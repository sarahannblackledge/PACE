import numpy as np

''' inputs:
    1. X: array-like, bool. 
        Any array of arbitrary size. If not boolean will be converted
    1. Y: array-like, bool
        Any other array of identical size. If not boolean, will be converted
        
    Returns:
    DSC: float
        Dice similarity coefficient as a float on range [0, 1]. '''

def DSC_calc(X, Y):
    X = np.asarray(X).astype(np.bool)
    Y = np.asarray(Y).astype(np.bool)

    if X.shape != Y.shape:
        raise ValueError('Shape mismatch: X and Y must have the same shape')

    #Compute DSC
    intersection = np.logical_and(X, Y)
    DSC = 2. * intersection.sum() / (X.sum() + Y.sum())
    return DSC




