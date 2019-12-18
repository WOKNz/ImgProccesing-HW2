import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.util as ut
import pandas as pd
import glob
import os

if __name__ == '__main__':

    def histrun(path, type, out_path):
        """
        function that generates Histogram Equalized pics
        :param path: string where files saved
        :param type: sting of type 'tif', 'jpg'
        :param out_path: string where result will saved
        :return:
        """

        # Loading files
        filenames = glob.glob(path + "*." + type)
        images = [cv2.imread(img) for img in filenames]
        names = [os.path.basename(x) for x in filenames]

        # Plot Original
        f, axarr = plt.subplots(3, 2)
        axarr[0, 0].imshow(images[0][..., ::-1], aspect='equal')
        axarr[0, 0].title.set_text(names[0])
        axarr[1, 0].imshow(images[1][..., ::-1], aspect='equal')
        axarr[1, 0].title.set_text(names[1])
        axarr[2, 0].imshow(images[2][..., ::-1], aspect='equal')
        axarr[2, 0].title.set_text(names[2])

        # Generating equalized pictures (for explanation of full implamintation lookt the Answers file doc
        for i, img in enumerate(images):
            img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
            img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
            img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
            cv2.imwrite(out_path + names[i], img)
            cv2.imshow(names[i] + 'Result', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Plotting the result
        axarr[0, 1].imshow(images[0][..., ::-1], aspect='equal')
        axarr[0, 1].title.set_text(names[0] + ' After Equalizaiotn')
        axarr[1, 1].imshow(images[1][..., ::-1], aspect='equal')
        axarr[1, 1].title.set_text(names[1] + ' After Equalizaiotn')
        axarr[2, 1].imshow(images[2][..., ::-1], aspect='equal')
        axarr[2, 1].title.set_text(names[2] + ' After Equalizaiotn')
        plt.show()
        f.savefig('Q2_A' + names[0] + '.png', dpi=300)


    # Processing on 3 custom pics
    histrun('raw/1/', 'tif', 'raw/1/result/')
    histrun('input/1/', 'jpg', 'input/1/result/')

    pass
