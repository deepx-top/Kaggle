# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from PIL import Image, ImageOps


def cvt_img(mn_dir, imgcls):
    _ds = pd.read_csv(mn_dir)
    if imgcls == 'train':
        images = np.asarray(_ds.iloc[:, 1:].values).astype('uint8')
        labels = np.asarray(_ds.iloc[:, 0].values).reshape(-1, )
    else:
        images = np.asarray(_ds.iloc[:, :].values).astype('uint8')
        labels = np.random.randint(0, 10, (images.shape[0], ))

    for i, xt in enumerate(images):
        im = xt.reshape(-1, 28)
        mg = Image.fromarray(im)
        mg = ImageOps.invert(mg)
        mg3 = Image.merge('RGB', (mg, mg, mg))
        img_name = './' + imgcls + '/' +str(labels[i]) + '/' + \
            str(i + 1) + '.jpg'
        mg3.save(img_name)


def main():
    cvt_img('./train.csv', 'train')
    # cvt_img('./test.csv', 'test')

if __name__ == '__main__':
    main()
