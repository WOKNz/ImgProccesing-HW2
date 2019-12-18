import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.util as ut
import pandas as pd
import glob

if __name__ == '__main__':

    def deriv(img, pixel):
        """

        :param img: nd.array of image
        :param pixel: nd.array of point (x,y)
        :return: derivative on x,y and normal
        """
        x = pixel[0]
        y = pixel[1]
        der = np.array([[1, 0, -1]])
        img_x = np.array([img[x - 1, y], img[x, y], img[x + 1, y]])
        img_y = np.array([img[x, y - 1], img[x, y], img[x, y + 1]])
        dx = np.inner(der, img_x)
        dy = np.inner(der, img_y)
        return dx, dy, np.sqrt(dx ** 2 + dy ** 2)


    def img_his_equal(image, number_bins=256):
        # Generate histogram
        image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
        cdf = image_histogram.cumsum()  # cdf
        cdf = 255 * cdf / cdf[-1]  # normalize

        # linear interpolation of cdf
        image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

        return image_equalized.reshape(image.shape), cdf


    # Creating black square and adding circle
    black_img = np.zeros((300, 300))
    black_img_circle = cv2.circle(black_img, (100, 100), 50, 100, -1)
    # plt.imshow(black_img_circle,aspect='equal',cmap='gray', vmin=0, vmax=255)
    # plt.show()

    # Applying Sobel filter
    # Creating kernel for x and y directions
    kernel_fox_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_fox_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobel_result_x = cv2.filter2D(black_img_circle, -1, kernel_fox_x)
    sobel_result_y = cv2.filter2D(black_img_circle, -1, kernel_fox_y)
    sobel_result = np.sqrt(sobel_result_x ** 2 + sobel_result_y ** 2)

    # plt.imshow(sobel_result, aspect='equal', cmap='gray', vmin=0, vmax=255)
    # plt.title('Sobel filter')
    # plt.show()

    # Applying Prewitt filter
    # Creating kernel for x and y directions
    kernel_fox_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernel_fox_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    prewitt_result_x = cv2.filter2D(black_img_circle, -1, kernel_fox_x)
    prewitt_result_y = cv2.filter2D(black_img_circle, -1, kernel_fox_y)
    prewitt_result = np.sqrt(prewitt_result_x ** 2 + prewitt_result_y ** 2)

    # plt.imshow(prewitt_result, aspect='equal', cmap='gray', vmin=0, vmax=255)
    # plt.title('Prewitt filter')
    # plt.show()

    # Applying Binomial filters
    # Creating kernel for two different binomial filters
    kernel_ofsize3 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    kernel_ofsize5 = np.array(
        [[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]) / 256
    bin_result_ofsize3 = cv2.filter2D(black_img_circle, -1, kernel_ofsize3)
    bin_result_ofsize5 = cv2.filter2D(black_img_circle, -1, kernel_ofsize5)

    f, axarr = plt.subplots(3, 2)
    axarr[0, 0].imshow(black_img_circle, aspect='equal', cmap='gray', vmin=0, vmax=255)
    axarr[0, 0].title.set_text('Original img')
    axarr[1, 0].imshow(sobel_result, aspect='equal', cmap='gray', vmin=0, vmax=255)
    axarr[1, 0].title.set_text('Sobel filter')
    axarr[2, 0].imshow(prewitt_result, aspect='equal', cmap='gray', vmin=0, vmax=255)
    axarr[2, 0].title.set_text('Prewitt filter')
    axarr[0, 1].imshow(bin_result_ofsize3, aspect='equal', cmap='gray', vmin=0, vmax=255)
    axarr[0, 1].title.set_text('Binomial filter size=3')
    axarr[1, 1].imshow(bin_result_ofsize5, aspect='equal', cmap='gray', vmin=0, vmax=255)
    axarr[1, 1].title.set_text('Binomial filter size=5')
    f.delaxes(axarr[2, 1])
    plt.show()
    f.savefig('Q1_A.png', dpi=1200)

    # 83 53

    # Applying Gausian noise
    noise_clean = np.zeros((300, 300))
    black_img_circle_sig2_on = np.clip(
        (ut.random_noise(noise_clean, mode='gaussian', var=2) * 255).astype(int) + black_img_circle, 0, 255)
    black_img_circle_sig10_on = np.clip(
        (ut.random_noise(noise_clean, mode='gaussian', var=2) * 255).astype(int) + black_img_circle, 0, 255)
    black_img_circle_sig20_on = np.clip(
        (ut.random_noise(noise_clean, mode='gaussian', var=2) * 255).astype(int) + black_img_circle, 0, 255)
    black_img_circle_sig2 = ut.random_noise(black_img_circle, mode='gaussian', var=2, clip=False)
    black_img_circle_sig10 = ut.random_noise(black_img_circle, mode='gaussian', var=10, clip=False)
    black_img_circle_sig20 = ut.random_noise(black_img_circle, mode='gaussian', var=20, clip=False)

    f2, axarr = plt.subplots(3, 2)
    axarr[0, 0].imshow(black_img_circle_sig2, aspect='equal', cmap='gray', vmin=0, vmax=255)
    axarr[0, 0].title.set_text('Gausian noise var=2')
    axarr[1, 0].imshow(black_img_circle_sig10, aspect='equal', cmap='gray', vmin=0, vmax=255)
    axarr[1, 0].title.set_text('Gausian noise var=10')
    axarr[2, 0].imshow(black_img_circle_sig20, aspect='equal', cmap='gray', vmin=0, vmax=255)
    axarr[2, 0].title.set_text('Gausian noise var=20')
    axarr[0, 1].imshow(black_img_circle_sig2_on, aspect='equal', cmap='gray', vmin=0, vmax=255)
    axarr[0, 1].title.set_text('Gausian noise var=2 on black also')
    axarr[1, 1].imshow(black_img_circle_sig10_on, aspect='equal', cmap='gray', vmin=0, vmax=255)
    axarr[1, 1].title.set_text('Gausian noise var=10 on black also')
    axarr[2, 1].imshow(black_img_circle_sig20_on, aspect='equal', cmap='gray', vmin=0, vmax=255)
    axarr[2, 1].title.set_text('Gausian noise var=20 on black also')
    plt.show()
    f2.savefig('Q1_C.png', dpi=600)

    # Before and after filter derivatives
    points = np.array([[83, 53], [84, 53], [85, 53], [86, 53]])
    deriv_result = np.empty((4, 15))
    for i in range(4):
        deriv_result[i, 0:3] = deriv(black_img_circle, np.array(points[i]))
        deriv_result[i, 3:6] = deriv(sobel_result, np.array(points[i]))
        deriv_result[i, 6:9] = deriv(prewitt_result, np.array(points[i]))
        deriv_result[i, 9:12] = deriv(bin_result_ofsize3, np.array(points[i]))
        deriv_result[i, 12:15] = deriv(bin_result_ofsize5, np.array(points[i]))
    derivAns_table = pd.DataFrame(deriv_result, \
                                  )
    derivAns_table = derivAns_table.rename(index={0: 'p(83,53)', 1: 'p(84,53)', 2: 'p(85,53)', 3: 'p(86,53)'}, \
                                           columns={0: 'Original(x,y,r)', 3: 'Sobel(x,y,r)', 6: 'Prewitt(x,y,r)', \
                                                    9: 'Binom3(x,y,r)', 12: 'Binom5(x,y,r)'})
    derivAns_table.to_excel('Q1_filters_compare.xlsx')
    print(derivAns_table)

    # After filter derivatives on gausian result
    deriv_result = np.empty((4, 9))
    for i in range(4):
        deriv_result[i, 0:3] = deriv(black_img_circle_sig2, np.array(points[i]))
        deriv_result[i, 3:6] = deriv(black_img_circle_sig10, np.array(points[i]))
        deriv_result[i, 6:9] = deriv(black_img_circle_sig20, np.array(points[i]))
    derivAns_table = pd.DataFrame(deriv_result, \
                                  )
    derivAns_table = derivAns_table.rename(index={0: 'p(83,53)', 1: 'p(84,53)', 2: 'p(85,53)', 3: 'p(86,53)'}, \
                                           columns={0: 'Sig=2 (x,y,r)', 3: 'Sig=10 (x,y,r)', 6: 'Sig=20 (x,y,r)'})
    derivAns_table.to_excel('Q1_filters_compare_gaus.xlsx')

    filenames = glob.glob("raw/1/*.tif")
    images = [cv2.imread(img) for img in filenames]

    for img in images:
        equal = cv2.equalizeHist(img)
        res = np.hstack((img, equal))  # stacking images side-by-side
        cv2.imwrite('res.png', res)
        cv2.imshow('image', img_equalized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    pass
