import hdf5storage
import os
import glob
import numpy as np


# numpy version
def mrae_loss(outputs, label):
    """Computes the rrmse value"""
    diff = label-outputs
    abs_diff = np.abs(diff)
    relative_abs_diff = np.divide(abs_diff, label+np.finfo(float).eps)
    return np.mean(relative_abs_diff)


# numpy version
def rmse_loss(outputs, label):
    """Computes the rmse value"""
    diff = label - outputs
    square_diff = np.power(diff, 2)
    return np.sqrt(np.mean(square_diff))


result_path = 'results'
ground_path = 'NTIRE2020_Validation_Spectral'

result_name = glob.glob(os.path.join(result_path, '*.mat'))
result_name.sort()
ground_name = glob.glob(os.path.join(ground_path, '*.mat'))
ground_name.sort()

record_mrae, record_rmse = [], []
for i in range(len(ground_name)):
    # load rusults hyper
    hs = hdf5storage.loadmat(ground_name[i])['cube']
    ground = np.float32(hs)   # 482,512,31
    hs = hdf5storage.loadmat(result_name[i])['cube']
    result = np.float32(hs)   # 482,512,31
    mrae = mrae_loss(result, ground)
    rmse = rmse_loss(result, ground)
    record_mrae.append(mrae)
    record_rmse.append(rmse)
print("Average mrae_loss [%.9f], average rmse_loss [%.9f]" % (np.array(record_mrae).mean(), np.array(record_rmse).mean()))
