import matplotlib.pyplot as plt
import cv2
from skimage import data
from skimage import exposure
from skimage.exposure import match_histograms
import glob
import os
import numpy as np

if __name__ == '__main__':
    filenames = glob.glob("raw/2/*.tif")
    images = [cv2.imread(img, 0) for img in filenames]
    names = [os.path.basename(x) for x in filenames]
    reference = images[0]
    # image = data.chelsea()

    # matched = match_histograms(image, reference, multichannel=True)
    #
    # fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
    #                                     sharex=True, sharey=True)
    # for aa in (ax1, ax2, ax3):
    #     aa.set_axis_off()
    #
    # ax1.imshow(image)
    # ax1.set_title('Source')
    # ax2.imshow(reference)
    # ax2.set_title('Reference')
    # ax3.imshow(matched)
    # ax3.set_title('Matched')
    #
    # plt.tight_layout()
    # plt.show()
    #
    # fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))
    #
    #
    # for i, img in enumerate((image, reference, matched)):
    #     for c, c_color in enumerate(('red', 'green', 'blue')):
    #         img_hist, bins = exposure.histogram(img[..., c], source_range='dtype')
    #         axes[c, i].plot(bins, img_hist / img_hist.max())
    #         img_cdf, bins = exposure.cumulative_distribution(img[..., c])
    #         axes[c, i].plot(bins, img_cdf)
    #         axes[c, 0].set_ylabel(c_color)
    #
    # axes[0, 0].set_title('Source')
    # axes[0, 1].set_title('Reference')
    # axes[0, 2].set_title('Matched')
    #
    # plt.tight_layout()
    # plt.show()

    # img 1
    fig, axes = plt.subplots(3, 3)

    img_hist, bins = exposure.histogram(images[0], source_range='dtype')
    axes[0, 0].plot(bins, img_hist / img_hist.max())
    img_cdf, bins = exposure.cumulative_distribution(images[0])
    axes[0, 0].plot(bins, img_cdf)

    mu, sigma, lam = 128, 15, 0.03488


    def nrm(mu, sigma, bins):
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2))


    max = np.max(nrm(mu, sigma, np.arange(256)))
    newimg = nrm(mu, sigma, images[0]) * 255 / np.max(nrm(mu, sigma, images[0]))
    axes[1, 0].imshow(newimg, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].plot(np.arange(256), 500 - nrm(mu, sigma, np.arange(256)) * 255 / max, color='r')


    # expo func with lam
    def expo(lam, bins):
        return lam * np.exp(lam * bins)


    newimg = expo(lam, images[0])
    axes[2, 0].imshow(newimg, cmap='gray', vmin=0, vmax=255)
    axes[2, 0].plot(np.arange(256), 500 - expo(lam, np.arange(256)), color='r')

    # img2
    img_hist, bins = exposure.histogram(images[1], source_range='dtype')
    axes[0, 1].plot(bins, img_hist / img_hist.max())
    img_cdf, bins = exposure.cumulative_distribution(images[1])
    axes[0, 1].plot(bins, img_cdf)

    max = np.max(nrm(mu, sigma, np.arange(256)))
    newimg = nrm(mu, sigma, images[1]) * 255 / np.max(nrm(mu, sigma, images[1]))
    axes[1, 1].imshow(newimg, cmap='gray', vmin=0, vmax=255)
    axes[1, 1].plot(np.arange(256), 500 - nrm(mu, sigma, np.arange(256)) * 255 / max, color='r')

    newimg = expo(lam, images[1])
    axes[2, 1].imshow(newimg, cmap='gray', vmin=0, vmax=255)
    axes[2, 1].plot(np.arange(256), 500 - expo(lam, np.arange(256)), color='r')

    # img3
    img_hist, bins = exposure.histogram(images[2], source_range='dtype')
    axes[0, 2].plot(bins, img_hist / img_hist.max())
    img_cdf, bins = exposure.cumulative_distribution(images[2])
    axes[0, 2].plot(bins, img_cdf)

    max = np.max(nrm(mu, sigma, np.arange(256)))
    newimg = nrm(mu, sigma, images[2]) * 255 / np.max(nrm(mu, sigma, images[2]))
    axes[1, 2].imshow(newimg, cmap='gray', vmin=0, vmax=255)
    axes[1, 2].plot(np.arange(256), 500 - nrm(mu, sigma, np.arange(256)) * 255 / max, color='r')

    newimg = expo(lam, images[2])
    axes[2, 2].imshow(newimg, cmap='gray', vmin=0, vmax=255)
    axes[2, 2].plot(np.arange(256), 500 - expo(lam, np.arange(256)), color='r')

    # plt.tight_layout()
    plt.show()
    # plt.hist()
