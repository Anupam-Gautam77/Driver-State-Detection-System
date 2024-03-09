# Define model-related constants
from keras.models import load_model
import pickle

BEST_MODEL_1 = r"D:\MAJOR_FINAL\major_final\myapp\model\xception_weights_aug_extralayer_alltrained_sgd2_V2.hdf5"

BEST_MODEL_2 = r"D:\MAJOR_FINAL\major_final\myapp\model\xception_weights_aug_alltrained_setval_sgd3 (1).hdf5"

BEST_MODEL_3 = r"D:\MAJOR_FINAL\major_final\myapp\model\resnet_weights_aug_alltrained_sgd2_setval.hdf5"

BEST_MODEL_4 = r"D:\MAJOR_FINAL\major_final\myapp\model\resnet_weights_aug_extralayers_sgd_setval(3).hdf5"

BEST_MODEL_5 = r"D:\MAJOR_FINAL\major_final\myapp\model\mobilenet_sgd_nolayers.hdf5"
with open(r"C:\Users\aashi\Desktop\falgun_6\Driver-State-Detection-System\Distraction\model\labels_list.pkl", "rb") as handle:
    labels_id = pickle.load(handle)