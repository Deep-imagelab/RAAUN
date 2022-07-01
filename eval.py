import torch
import os
import numpy as np
import cv2
from RAAUN import RAAUN
import glob
import hdf5storage


os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model_path = 'model.pth'
result_path = 'results'
img_path = 'NTIRE2020_Validation_Clean'
var_name = 'cube'

# save results
if not os.path.exists(result_path):
    os.makedirs(result_path)
model = RAAUN()
save_point = torch.load(model_path)
model_param = save_point['state_dict']
model_dict = {}
for k1, k2 in zip(model.state_dict(), model_param):
    model_dict[k1] = model_param[k2]
model.load_state_dict(model_dict)
model = model.cuda()
model.eval()

img_path_name = glob.glob(os.path.join(img_path, '*.png'))
img_path_name.sort()

for i in range(len(img_path_name)):
    # load rgb images
    rgb = cv2.imread(img_path_name[i])
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = np.float32(rgb) / 255.0
    rgb = np.expand_dims(np.transpose(rgb, [2, 0, 1]), axis=0).copy()
    print(img_path_name[i].split('/')[-1])
    rgb = torch.from_numpy(rgb).cuda()
    with torch.no_grad():
        img_res = model(rgb)
    img_res = img_res.cpu().numpy() * 1.0
    img_res = np.transpose(np.squeeze(img_res), [1, 2, 0])
    img_res = np.minimum(img_res, 1.0)
    img_res = np.maximum(img_res, 0)
    mat_name = img_path_name[i].split('/')[-1][:-10] + '.mat'
    mat_dir = os.path.join(result_path, mat_name)

    hdf5storage.savemat(mat_dir, {var_name: img_res}, format='7.3', store_python_metadata=True)







