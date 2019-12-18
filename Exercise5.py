import scipy.io as spIO
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import numpy as np
import glob
import pylab
import imageio

if __name__ == '__main__':

    # load data
    matF = spIO.loadmat('video1.mat')
    video = matF['video1']
    filenames = glob.glob("raw/4/*.tif")
    images = [cv2.imread(img, 0) for img in filenames]
    myimages = []
    pic2s = images[0]
    fig = plt.figure()


    # Matching Function
    def match(pic, pic2s, threshold):
        """

        :param pic: pic matching on
        :param pic2s: pic matched in
        :param threshold:
        :return:
        """
        # part 2
        img_gray = pic

        # Read the template
        template = pic2s

        # Store width and height of template in wide and higth
        w, h = template.shape[::-1]

        # Perform match
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

        # Store the coordinates of matched area in a numpy array
        loc = np.where(res >= threshold)
        color = (0, 0, 255)
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2RGB)
        # Plot a rectangle around the matched region.
        for pt in zip(*loc[::-1]):
            img_gray = cv2.rectangle(img_gray, pt, (pt[0] + w, pt[1] + h), color, 2)
            break  # Take only first rectangle

        #  final result
        result = cv2.cvtColor(img_gray, cv2.COLOR_BGR2RGB)
        return result


    for frame in range(video.shape[2]):
        imgofvideo = video[:, :, frame]
        imgplot = plt.imshow(match(imgofvideo, pic2s, 0.65))
        myimages.append([imgplot])

    ani = animation.ArtistAnimation(fig, myimages, interval=200, blit=True, repeat_delay=1e9)

    plt.axis('off')
    writer = animation.ImageMagickFileWriter()
    ani.save('video.gif', writer=writer)

    # Bonus

    # load data
    matF = spIO.loadmat('video1.mat')
    video = matF['video1']
    myimages = []
    fig = plt.figure()


    # apply Otsu
    def otsu(img):
        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th


    for frame in range(video.shape[2]):
        imgofvideo = video[:, :, frame]
        imgplot = plt.imshow(otsu(imgofvideo), cmap='gray', interpolation='nearest')
        myimages.append([imgplot])

    ani = animation.ArtistAnimation(fig, myimages, interval=200, blit=True, repeat_delay=1e9)

    # Save
    plt.axis('off')
    writer = animation.ImageMagickFileWriter()
    ani.save('OtsuVideo.gif', writer=writer)
