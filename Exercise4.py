import cv2
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    filenames = glob.glob("raw/3/*.tif")
    images = [cv2.imread(img, 0) for img in filenames]
    names = [os.path.basename(x) for x in filenames]
    new_image = cv2.medianBlur(images[0], 3)
    f, axarr = plt.subplots(2, 1)
    axarr[0].imshow(images[0], cmap='gray', vmin=0, vmax=255)
    axarr[0].title.set_text('Before Filter')
    axarr[1].imshow(new_image, cmap='gray', vmin=0, vmax=255)
    axarr[1].title.set_text('After Median Filter')
    plt.show()
    f.savefig('Q4_A.png', dpi=300)


    def match(pic, pic2s, threshold, outpuname):
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

        # Plot final and save
        plt.imshow(cv2.cvtColor(img_gray, cv2.COLOR_BGR2RGB))
        plt.show()
        cv2.imwrite(outpuname + '.png', img_gray)
        return loc


    location_unclean = match(images[0], images[1], 0.65, 'Q4_b_unclean')
    location_clean = match(new_image, images[1], 0.65, 'Q4_b_clean')
    print('Location of match on unclean ', location_unclean)
    print('Location of match on clean ', location_clean)
