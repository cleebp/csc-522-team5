import numpy as np
import dicom
import glob
import os
#import cv2
import xgboost as xgb
import pandas as pd
import mxnet as mx

from sklearn.model_selection import StratifiedKFold
from scipy.stats import gmean
from matplotlib import pyplot as plt

INPUT_FOLDER = 'sample_images/'
n_patients = len(os.listdir(INPUT_FOLDER))

def get_conv_model():
    model = mx.model.FeedForward.load('pretrained_models/resnet-50', 0, ctx=mx.gpu(), numpy_batch_size=1)
    fea_symbol = model.symbol.get_internals()["flatten0_output"]
    feature_extractor = mx.model.FeedForward(ctx=mx.gpu(), symbol=fea_symbol, numpy_batch_size=64, arg_params=model.arg_params, aux_params=model.aux_params, allow_extra_params=True)

    return feature_extractor

def get_3d_data(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    return np.stack([s.pixel_array for s in slices])

def get_scan(path):
    sample_image = get_3d_data(path)
    sample_image[sample_image == -2000] = 0

    batch = []
    cnt = 0
    dx = 40
    ds = 512
    for i in range(0, sample_image.shape[0] - 3, 3):
        tmp = []
        for j in range(3):
            img = sample_image[i + j]
            img = 255.0 / np.amax(img) * img
            img = cv2.equalizeHist(img.astype(np.uint8))
            img = img[dx: ds - dx, dx: ds - dx]
            img = cv2.resize(img, (224, 224))
            tmp.append(img)

        tmp = np.array(tmp)
        batch.append(np.array(tmp))

    batch = np.array(batch)
    return batch

def calc_features():
    net = get_conv_model()
    for i, folder in enumerate(glob.glob('stage1/*')):
        batch = get_scan(folder)
        feats = net.predict(batch)
        
        if i % 1 == 0:
            print("Processed {0} of {1}".format(i, n_patients))
            print("Shape of result:", feats.shape)
            print()

        np.save(folder.replace("stage1", "res_feats_stage1"), feats)

def train_xgboost():
    df = pd.read_csv("metadata/stage1_labels.csv")

    x = np.array([np.mean(np.load('res_feats_stage1/%s.npy' % str(id)), axis=0) for id in df['id'].tolist()])
    y = df['cancer'].as_matrix()

    skf = StratifiedKFold(n_splits=5, random_state=2017, shuffle=True)

    clfs = []
    for train_index, test_index in skf.split(x, y):
        trn_x, val_x = x[train_index,:], x[test_index,:]
        trn_y, val_y = y[train_index], y[test_index]

        clf = xgb.XGBRegressor(max_depth=6,
                               n_estimators=1500,
                               min_child_weight=100,
                               learning_rate=0.037,
                               nthread=8,
                               subsample=0.9,
                               colsample_bytree=0.95,
                               seed=42)

        clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss', early_stopping_rounds=50)
        clfs.append(clf)

    return clfs

def make_submit():
    clfs = train_xgboost()

    df = pd.read_csv("submissions/stage1_sample_submission.csv")

    x = np.array([np.mean(np.load('res_feats_stage1/%s.npy' % str(id)), axis=0) for id in df['id'].values])
    
    preds = []
    for clf in clfs:
        preds.append(np.clip(clf.predict(x), 0.001, 1))

    pred = gmean(np.array(preds), axis=0)
    df['cancer'] = pred
    df.to_csv('submissions/-LB_xgb_0.037LR_1500EST_MXD6_5FOLD_E.csv', index=False)
    print(df.head())



calc_features()    
make_submit()


