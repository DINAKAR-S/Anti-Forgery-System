from pylab import *
import numpy as np
from os import listdir
from sklearn.svm import LinearSVC
import cv2
from PIL import Image
from sklearn import svm
import imagehash
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn import linear_model
import preproc
import features
import pickle

def svm_algo():
    genuine_image_filenames = listdir("data/genuine")  # list of names of all the files in directory data/genuine
    forged_image_filenames = listdir("data/forged")  # list of names of all the files in directory data/forged
    genuine_image_paths = "data/genuine"
    forged_image_paths = "data/forged"
    image_test_paths = "static/LineSweep_Results"
    image_test = listdir("static/LineSweep_Results")

    image_features = []
    genuine_image_features = [[] for x in range(29)]  # creates empty list of 29 features.
    forged_image_features = [[] for x in range(29)]
    countSignaturePerUser = 0

    for name in genuine_image_filenames:
        signature_id = int(name.split('_')[0][-3:])
        genuine_image_features[signature_id - 1].append({"name": name})

    for _ in genuine_image_features:
        for __ in _:
            countSignaturePerUser = countSignaturePerUser + 1
        break

    for name in forged_image_filenames:
        signature_id = int(name.split('_')[0][-3:])
        forged_image_features[signature_id - 1].append({"name": name})

    for name in image_test:
        signature_id = int(name.split('_')[0][-3:])
        image_features.append({"name": name})

    def preprocess_image(path, display=False):
        # return 0,1 image
        return preproc.preproc(path, display=display)

    des_list = []

    def sift(im, path, display=False):
        raw_image = cv2.imread(path)
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(im, None)

        if display:
            cv2.drawKeypoints(im, kp, raw_image)
            cv2.imshow('sift_keypoints.jpg', cv2.resize(raw_image, (0, 0), fx=3, fy=3))
            cv2.waitKey()

        return (path, des)

    cor_gen = 0
    wrong_gen = 0
    cor_for = 0
    wrong_for = 0

    im_contour_features = []
    print("processing svm model...")
    for i in range(29):
        des_list = []
        for im in genuine_image_features[i]:
            image_path = genuine_image_paths + "/" + im['name']
            preprocessed_image = preprocess_image(image_path)
            hash = imagehash.phash(Image.open(image_path))

            aspect_ratio, bounding_rect_area, convex_hull_area, contours_area = \
                features.get_contour_features(preprocessed_image.copy(), display=False)

            hash = int(str(hash), 16)
            im['hash'] = hash
            im['aspect_ratio'] = aspect_ratio
            im['hull_area/bounding_area'] = convex_hull_area / bounding_rect_area
            im['contour_area/bounding_area'] = contours_area / bounding_rect_area

            im['ratio'] = features.Ratio(preprocessed_image.copy())
            im['centroid_0'], im['centroid_1'] = features.Centroid(preprocessed_image.copy())

            im['eccentricity'], im['solidity'] = features.EccentricitySolidity(preprocessed_image.copy())
            (im['skewness_0'], im['skewness_1']), (im['kurtosis_0'], im['kurtosis_1']) = features.SkewKurtosis(
                preprocessed_image.copy())
            im_contour_features.append(
                [aspect_ratio, convex_hull_area / bounding_rect_area, contours_area / bounding_rect_area, im['ratio'],
                 im['centroid_0'], im['centroid_1'], im['eccentricity'], im['solidity'], im['skewness_0'], im['skewness_1'],
                 im['kurtosis_0'], im['kurtosis_1']])

            des_list.append(sift(preprocessed_image.copy(), image_path))

        for im in forged_image_features[i]:
            image_path = forged_image_paths + "/" + im['name']
            preprocessed_image = preprocess_image(image_path)
            hash = imagehash.phash(Image.open(image_path))

            aspect_ratio, bounding_rect_area, convex_hull_area, contours_area = \
                features.get_contour_features(preprocessed_image.copy(), display=False)

            hash = int(str(hash), 16)
            im['hash'] = hash
            im['aspect_ratio'] = aspect_ratio
            im['hull_area/bounding_area'] = convex_hull_area / bounding_rect_area
            im['contour_area/bounding_area'] = contours_area / bounding_rect_area

            im['ratio'] = features.Ratio(preprocessed_image.copy())
            im['centroid_0'], im['centroid_1'] = features.Centroid(preprocessed_image.copy())

            im['eccentricity'], im['solidity'] = features.EccentricitySolidity(preprocessed_image.copy())
            (im['skewness_0'], im['skewness_1']), (im['kurtosis_0'], im['kurtosis_1']) = features.SkewKurtosis(
                preprocessed_image.copy())
            im_contour_features.append(
                [aspect_ratio, convex_hull_area / bounding_rect_area, contours_area / bounding_rect_area, im['ratio'],
                 im['centroid_0'], im['centroid_1'], im['eccentricity'], im['solidity'], im['skewness_0'], im['skewness_1'],
                 im['kurtosis_0'], im['kurtosis_1']])

            des_list.append(sift(preprocessed_image.copy(), image_path))

        im_contour_features_test = []
        for im in image_features:
            image_path = image_test_paths + "/" + im['name']
            preprocessed_image = preprocess_image(image_path)
            hash = imagehash.phash(Image.open(image_path))

            aspect_ratio, bounding_rect_area, convex_hull_area, contours_area = \
                features.get_contour_features(preprocessed_image.copy(), display=False)

            hash = int(str(hash), 16)
            im['hash'] = hash
            im['aspect_ratio'] = aspect_ratio
            im['hull_area/bounding_area'] = convex_hull_area / bounding_rect_area
            im['contour_area/bounding_area'] = contours_area / bounding_rect_area

            im['ratio'] = features.Ratio(preprocessed_image.copy())
            im['centroid_0'], im['centroid_1'] = features.Centroid(preprocessed_image.copy())

            im['eccentricity'], im['solidity'] = features.EccentricitySolidity(preprocessed_image.copy())
            (im['skewness_0'], im['skewness_1']), (im['kurtosis_0'], im['kurtosis_1']) = features.SkewKurtosis(
                preprocessed_image.copy())
            im_contour_features_test.append(
                [aspect_ratio, convex_hull_area / bounding_rect_area, contours_area / bounding_rect_area, im['ratio'],
                 im['centroid_0'], im['centroid_1'], im['eccentricity'], im['solidity'], im['skewness_0'], im['skewness_1'],
                 im['kurtosis_0'], im['kurtosis_1']])

            des_list.append(sift(preprocessed_image.copy(), image_path))

        descriptors = des_list[0][1]
        for image_path, descriptor in des_list[1:]:
            descriptors = np.vstack((descriptors, descriptor))

        # Check if the number of descriptors is less than k
        k = 500
        if descriptors.shape[0] < k:
            k = descriptors.shape[0]

        voc, variance = kmeans(descriptors, k, 1)

        im_features = np.zeros((len(genuine_image_features[i]) + len(forged_image_features[i]), k + 12), "float32")
        for ii in range(len(genuine_image_features[i]) + len(forged_image_features[i])):
            words, distance = vq(des_list[ii][1], voc)
            for w in words:
                im_features[ii][w] += 1

            for j in range(12):
                im_features[ii][k + j] = im_contour_features[ii][j]

        im_features_test = np.zeros((len(image_features[0]), k + 12), "float32")
        for ii in range(len(image_features)):
            words, distance = vq(des_list[ii][1], voc)
            for w in words:
                im_features_test[ii][w] += 1

            for j in range(12):
                im_features_test[ii][k + j] = im_contour_features_test[ii][j]

        train_genuine_features, test_genuine_features = im_features[0:3], im_features[3:5]
        train_forged_features, test_forged_features = im_features[5:8], im_features[8:10]
        test_features = im_features_test

        clf = LinearSVC()
        clf.fit(np.concatenate((train_forged_features, train_genuine_features)),
                np.array([1 for x in range(len(train_forged_features))] + [2 for x in range(len(train_genuine_features))]))
        
        model = pickle.load(open('model.pkl', 'rb'))
        genuine_res = clf.predict(test_genuine_features)
        forged_res = clf.predict(test_forged_features)

        for res in genuine_res:
            if int(res) == 2:
                cor_gen += 1
            else:
                wrong_gen += 1

        for res in forged_res:
            if int(res) == 1:
                cor_for += 1
            else:
                wrong_for += 1

    res_forged = float(cor_for/(cor_for + wrong_for))
    res_genuine = float(cor_gen/(cor_gen + wrong_gen))

    result = ""
    if res_genuine > res_forged:
        result = "Genuine"
    else:
        result = "Forged"

    print("Processing Complete.")

    return result
