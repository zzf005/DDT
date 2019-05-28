import tensorflow as tf
import keras_resnet.models
import numpy as np
import keras
import cv2,os

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    inputs = keras.layers.Input(shape=(None, None, 3))
    resnet = keras_resnet.models.ResNet50(inputs, include_top=False, freeze_bn=False)
    resnet.load_weights('model/ResNet-50-model.keras.h5', by_name=True)
    files = os.listdir('tiger')
    images = []
    dimen = 512
    all_dimen = []
    for f in files:
        img = cv2.imread('tiger/'+f)
        images.append(img)
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        C2, C3, C4, C5 = resnet.predict(img)
        all_dimen.append(C3)
    mean = np.zeros((dimen))
    cov = np.zeros((dimen, dimen))
    for X in all_dimen:
        mean += X.squeeze().sum(0).sum(0)
    mean /= len(all_dimen) * X.shape[1] * X.shape[2]
    for X in all_dimen:
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                cov += np.matmul((X[0, j, k, :] - mean).transpose(), (X[0, j, k, :] - mean))
    cov /= len(all_dimen) * X.shape[1] * X.shape[2]
    eigenvalue, featurevec = np.linalg.eig(cov)
    index = np.argsort(eigenvalue)
    Num = 0
    all_dimen = all_dimen[:50]
    for i, X in enumerate(all_dimen):
        print('solve {}th image'.format(i))
        p = np.zeros((1, X.shape[1] * X.shape[2]))
        for j in range(2):
            p += np.matmul(featurevec[:, index[-1-j]].reshape(1, dimen), (X - mean).reshape(-1, dimen).transpose()).astype(np.float32)
        p = p.reshape(X.shape[1], X.shape[2])
        p = cv2.resize(p.astype(np.float32), (images[i].shape[1], images[i].shape[0]), interpolation=cv2.INTER_NEAREST)
        bin_p = np.where(p >= 0, np.ones_like(p), np.zeros_like(p))
        num, labels = cv2.connectedComponents(bin_p.astype(np.uint8))
        area = [(labels==i).sum() for i in range(num)]
        index = np.argsort(np.array(area))
        if labels[labels==index[-1]].sum() != 0:
            pos = index[-1]
        else:
            pos = index[-2]
        flabel = np.where(labels == pos)
        y1, x1 = flabel[0][np.argmin(flabel[0])], flabel[1][np.argmin(flabel[1])]
        y2, x2 = flabel[0][np.argmax(flabel[0])], flabel[1][np.argmax(flabel[1])]
        cv2.imwrite('{}.jpg'.format(Num),cv2.rectangle(images[i], (x1, y1), (x2, y2), (255,255,0), 2))
        Num += 1
    
