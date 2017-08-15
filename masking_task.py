import numpy as np
from sklearn.cluster import DBSCAN
from astropy.io import fits
import os
import time
from read_rdf import *

openCV = False
try:
    import cv2

    openCV = True
except ImportError:
    from scipy import ndimage


def read_radar_img(filename):
    """
    loads image from supported formats:
         .dat (using info from .hdr file)
         .fits or .fit
         .txt (excluding those that end in mask.txt)
         .rdf
    :param filename: string containing path to file
    :return: matrix containing pixel values
    """

    # check that file exists
    if not os.path.isfile(filename):
        print("ERROR: File does not exist.")
        return None

    # deal with .dat (and .hdr)
    if filename.endswith(".dat"):
        filenameCore = filename[:-4]

        # check that there's a corresponding .hdr file for .dat file
        if not os.path.isfile(filenameCore + '.hdr'):
            print(("ERROR: I cannot read this .dat file because no .hdr associated. "
                   "To be associated, an .hdr file must have same filename as its .dat file. "
                   "e.g., filename.dat and filename.hdr"))
            return None

        # format .dat file as matrix based off information in .hdr
        height = width = auxpix = None
        for line in open(filenameCore + '.hdr', 'r'):
            if "Height" in line:
                height = int(line.split()[2])
            elif "Width" in line:
                width = int(line.split()[2])
            auxpix = []
        for line in open(filenameCore + '.dat', 'r'):
            auxpix.append(float(line))
        pix = np.reshape(auxpix, (height, width))

    # deal with case of .hdr being input into function
    elif filename.endswith(".hdr"):
        return None

    # deal with .fits (or .fit)
    elif filename.endswith(".fits") or filename.endswith(".fit"):
        fo = fits.open(filename)
        pix = fo[0].data
        fo.close()

    # deal with .txt
    elif filename.endswith(".txt"):
        if filename.endswith("mask.txt"):
            return None
        lines = np.loadtxt(filename, unpack=False)
        pix = np.array(lines)

    # deal with .rdf files
    elif filename.endswith(".rdf"):
        pix = read_rdf(filename)['data']

    elif filename.endswith(".DS_Store"):
        return None

    # deal with other cases
    else:
        print('ERROR w/ read_radar_img: I cannot read ' + filename +
              ' Files must end with .fits (or .fit),'
              '.txt (NOT mask.txt), .dat (with an .hdr of same filename), or .rdf.')
        return None

    return pix


def threshold(mat, nStdev=1):
    """
    function selects only pixel values above a threshold.
    :param mat: array with pixel values 0 -- 255 (e.g., output from scale_and_blur)
    :param nStdev: number standard deviations from mean; pixels below this value will be replaced with 0
    :return: thresholded array
    """

    # define threshold value
    # only pixel values/255 > thresh will remain.
    thresh = mat.mean() + nStdev * mat.std()
    thresh /= 255.0

    # standardize mat so its values are 0.0-1.0 (instead of 0-255)
    A_thresh = mat / 255.0

    # convert values less than threshold to zero
    A_thresh -= thresh
    A_thresh[A_thresh < 0] = 0  # turn negative values to zero

    # re-scale matrix so that values are 0-255 instead of 0.0-1.0
    A_thresh /= (1 - thresh)
    A_thresh *= 255.0

    # return matrix
    return A_thresh


def scale_and_blur(A, rOrig=False, pix_blur=(5, 5)):
    """
    function scales and blurs image matrix
    :param A: image array (e.g., output from read_radar_img)
    :param rOrig: =True will return the original image (and NOT the filtered image)
    :param pix_blur: the parameter input to cv2.blur
    :return: scaled and blurred matrix
    """

    # scale image so it's a grayscale with values 0 - 255
    A_scaled = A - A.min()
    A_scaled = A_scaled / (A.max() - A.min())
    A_scaled *= 255

    # return original (scaled) image if rOrig = True
    if rOrig:
        return A_scaled

    # blur image
    if openCV:
        A_scale_and_blur = cv2.blur(A_scaled, pix_blur)
    else:
        k = np.ones(pix_blur) / float(pix_blur[0] * pix_blur[1])
        A_scale_and_blur = ndimage.convolve(A_scaled, k, mode='mirror')

    # return scaled and blurred image
    return A_scale_and_blur


def findClusters(A, disp_axes=None):
    """
    finds clusters among points in input matrix using DBSCAN
    :param A: image matrix (e.g., output from scale_and_blur)
    :param disp_axes: disp_axes=None will not display DBSCAN results
    :return:
    """

    # save image length and width
    n_rows = A.shape[0]
    n_col = A.shape[1]

    # define DBSCAN parameters
    eps = 5
    min_samples = 70

    # turn array into set of x,y points for DBSCAN input
    points = []
    for x, y in zip(*np.where(A > 0)):
        points.append((y, x))
    points = np.array(points)
    # print("Number of points input to DBSCAN: {}".format(points.shape))

    # Perform a DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    unique_labels = set(labels)

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    # print("Number of clusters : {}".format(n_clusters_))

    # Find the centroids and plot dbscan results
    cluster_pts = []
    for k, col in zip(unique_labels, colors):
        if k == -1:
            continue
        class_member_mask = (labels == k)
        xy_all = points[class_member_mask]
        cluster_pts.append(xy_all)
        # print("No. pts in cluster {}: {}".format(k, xy_all.shape[0]))

        # plot DBSCAN clusters
        if not (disp_axes is None):
            xy = points[class_member_mask & core_samples_mask]
            # print("Centroid for cluster {}: {}".format(k, xy.mean(axis=0)))
            disp_axes.set_aspect('equal')
            disp_axes.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                      markeredgecolor='k', markersize=14)
    if not (disp_axes is None):
        disp_axes.set_title('DBSCAN result')

    if n_clusters_ == 0:
        print("No clusters found, so mask contains all 0s. "
              "Try lowering nStdev, or check if there's a DC spike "
              "in the original image (set cropDCspike=True).")
        cluster_pts = []

    return [cluster_pts, n_rows, n_col]


def clustersToMask(cluster_pts, n_rows, n_col, win=(5, 10)):
    """
    converts points in cluster into mask
    :param cluster_pts: a list of y,x points (e.g., output from findClusters)
    :param n_rows: number of rows in image matrix (length in x)
    :param n_col: number of columns in image matrix (length in y)
    :param win: tuple with window size for mask
    :return:
    """

    # assign 1s to mask over points within win of cluster points
    win_x = win[0]
    win_y = win[1]
    mask = np.zeros((n_col, n_rows))
    for group in cluster_pts:
        for pt in range(0, len(group)):
            center_x = group[pt][0]
            center_y = group[pt][1]
            max_x = win_x
            min_x = win_x
            max_y = win_y
            min_y = win_y

            if center_x - min_x < 0:
                min_x = center_x
            if center_y - min_y < 0:
                min_y = center_y
            if center_x + max_x >= n_col:
                max_x = n_col - center_x
            if center_y + max_y >= n_rows:
                max_y = n_rows - center_y

            mask[(center_x - min_x):(center_x + max_x + 1), (center_y - min_y):(center_y + max_y + 1)] = 1

    return np.transpose(mask)


def writeMask(filename, mask):
    """
    writes mask to .txt file
    :param filename: string, filename of image
    :param mask: array containing the 0s and 1s of the mask
    :return: void
    """

    # deal with .dat, .fit, .rdf, .txt extensions
    if filename[-4] == '.':
        mask_name = filename[:-4]

    # deal with .fits extensions
    elif filename[-5] == '.':
        mask_name = filename[:-5]
    else:
        print("ERROR: Unable to save mask. Unknown file extension.")
        return None

    mask_name += "_mask.txt"
    np.savetxt(mask_name, mask, fmt='%u', delimiter=',')
    print("Created {} \n".format(mask_name))

    return


def rmDCspike(A, nMad=150):
    """
    removes DC spike (i.e., points > nMad absolute deviations from median absolute deviation)
    :param A: array
    :param nMad: number of median absolute deviations above median considered outliers
    :return: input array with outliers replaced with 0
    """

    # scale image so it's a grayscale with values 0 - 255
    A_scaled = A - A.min()
    A_scaled /= (A.max() - A.min())
    A_scaled *= 255
    A = A_scaled

    def mad(arr):
        """
        computes Median Absolute Deviation
        :param arr: array
        :return: Median Absolute Deviation
        """
        arr = np.ma.array(arr).compressed()  # should be faster to not use masked arrays.
        med = np.median(arr)
        return np.median(np.abs(arr - med))

    # replace points > 150 mad above median with zero
    A_mad = mad(A)
    A_outliers = (A > (np.median(A) + nMad * A_mad))
    A_outliers = np.invert(A_outliers)

    return A * A_outliers


def create_mask(filename, win=(5, 10), nStdev=1, do_write_mask=True,
               do_write_imgs=True, do_rmDCspike=False):
    """
    creates mask for image based on cluster information
    :param filename: string with path towards the .rdf, .dat, or .txt file
    :param win: =(win_x, win_y) is the window size around cluster points
    :param nStdev: number of standard deviations above mean for threshold
    :param do_write_mask: =True will write mask file as .txt
    :param do_write_imgs: =True will save images as .png files
    :param do_rmDCspike: =True will remove a DC spike from original image [beta]
    :return: True if completed successfully
    """

    start_time = time.time()

    # load image and check for valid output
    try:
        A = read_radar_img(filename)
        if A is None:
            return False
        print("Loading " + filename)
    except:
        print("ERROR: unable to read {}".format(filename))
        return False

    # remove DC spike
    if do_rmDCspike:
        A = rmDCspike(read_radar_img(filename))
        A = scale_and_blur(A, rOrig=True)

    # set graphing parameters
    fig = None
    ax1 = ax2 = ax3 = None
    origImg = None
    if do_write_imgs:
        plt.close('all')
        fig = plt.figure()
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        # save original image
        if do_rmDCspike:
            origImg = A
            ax1.set_title('Original image \n (minus DC spike)')
        else:
            origImg = scale_and_blur(read_radar_img(filename), rOrig=True)
            ax1.set_title('Original image')

        # show original image
        ax1.imshow(origImg, cmap='gray')

    # preprocessed image
    A0 = scale_and_blur(A)
    A1 = threshold(A0, nStdev)

    # find clusters in preprocessed image
    if do_write_imgs:
        [cluster_pts, n_rows, n_col] = findClusters(A1, disp_axes=ax2)
        for axis in [ax1, ax2, ax3]:
            axis.set_xlim([0, n_col])
            axis.set_ylim([0, n_rows])
            axis.invert_yaxis()
    else:
        [cluster_pts, n_rows, n_col] = findClusters(A1)

    # convert cluster points into a mask
    mask = clustersToMask(cluster_pts, n_rows, n_col, win=win)

    # write mask to .txt file
    if do_write_mask:
        writeMask(filename, mask)

    if do_write_imgs:
        # save masked image
        ax3.imshow(origImg * mask, cmap='gray')
        ax3.set_title('Masked image')

        # save plots
        im_name = filename.rsplit('/', 1)[1]
        im_name = im_name.rsplit(".", 1)[0]
        fig.tight_layout(pad=2)
        fig.savefig(im_name + ".png", dpi=300)

    print("Run time: {} \n".format(time.time() - start_time))

    return True


def main():
    # cycle through files
    filenameCore = os.getcwd() + "/test_data"
    for folder in os.listdir(filenameCore):
        if not folder == ".DS_Store":
            dir_str = filenameCore + "/" + folder
            for image in os.listdir(dir_str):
                fn = dir_str + "/" + image
                create_mask(fn, do_write_mask=False, do_write_imgs=False)
    return


if __name__ == '__main__':
    main()
