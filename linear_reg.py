import sys
import numpy as np
from pathlib import Path

def LinRegClosed():
    if len(sys.argv) !=5:
        raise ValueError("Expected 4 arguments: train.csv test.csv predictions.txt weights.txt")
    
    train_file = Path(sys.argv[1])
    test_file = Path(sys.argv[2])
    pred_file = Path(sys.argv[3])
    weights_file = Path(sys.argv[4])

    if not train_file.is_file():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not test_file.is_file():
        raise FileNotFoundError(f"Test File not found: {test_file}")
    
    train_data = np.loadtxt(train_file,delimiter=',',skiprows=1)
    X = train_data[:,:-1]
    y = train_data[:,-1]

    Xaug = np.c_[np.ones((X.shape[0],1)),X]
   
    print("Calculating weights")
    w = np.linalg.inv(Xaug.T @ Xaug) @ Xaug.T @y
    intercept = w[0]
    coefficients = w[1:]
    print("saving model parameters")
    np.savetxt(weights_file,w)

    # for prediction
    test_data = np.loadtxt(test_file,delimiter=',', skiprows=1)
    X_test = test_data
    
    y_pred = X_test @ coefficients + intercept
    print("saving predictions")
    np.savetxt(pred_file,y_pred)
    return 0



reg = LinRegClosed()