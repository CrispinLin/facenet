"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
"""
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
from tensorflow.python.ops import data_flow_ops
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
import cv2


def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)


def add_extension(path):
    if os.path.exists(path + '.jpg'):
        return path + '.jpg'
    elif os.path.exists(path + '.png'):
        return path + '.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)


def get_paths(patch_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = add_extension(os.path.join(patch_dir, pair[0], pair[1]))
            path1 = add_extension(os.path.join(patch_dir, pair[0], pair[2]))
            issame = True
        elif len(pair) == 4:
            path0 = add_extension(os.path.join(patch_dir, pair[0], pair[1]))
            path1 = add_extension(os.path.join(patch_dir, pair[2], pair[3]))
            issame = False
        if os.path.exists(path0) and os.path.exists(
                path1):  # Only add the pair if both paths exist
            path_list += (path0, path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, issame_list


def main(args):

    # Read the file containing the pairs used for testing
    pairs = read_pairs(os.path.expanduser(args.lfw_pairs))

    # Get the paths for the corresponding images
    paths, actual_issame = get_paths(os.path.expanduser(args.patch_dir), pairs)

    print(paths[0])
    print(paths[1])
    print(actual_issame[0])

    img1 = cv2.imread(paths[0], cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(paths[1], cv2.IMREAD_GRAYSCALE)

    sift = cv2.xfeatures2d.SIFT_create()
    # surf = cv2.xfeatures2d.SURF_create()

    # create a mask image filled with zeros, the size of original image
    mask = np.zeros(img1.shape[:2], dtype=np.uint8)
    # draw your selected ROI on the mask image
    cv2.rectangle(mask, (24, 24), (40, 40), (255), thickness=-1)

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    print(good)
    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None)

    cv2.imshow("matches", img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(paths[400])
    print(paths[401])
    print(actual_issame[200])

    img1 = cv2.imread(paths[400], cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(paths[401], cv2.IMREAD_GRAYSCALE)

    sift = cv2.xfeatures2d.SIFT_create()
    # surf = cv2.xfeatures2d.SURF_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, img2)

    cv2.imshow("dismatch", img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass


# def evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder,
#              phase_train_placeholder, batch_size_placeholder,
#              control_placeholder, embeddings, labels, image_paths,
#              actual_issame, batch_size, nrof_folds, distance_metric,
#              subtract_mean, use_flipped_images,
#              use_fixed_image_standardization):
#     # Run forward pass to calculate embeddings
#     print('Runnning forward pass on LFW images')

#     # Enqueue one epoch of image paths and labels
#     nrof_embeddings = len(
#         actual_issame) * 2  # nrof_pairs * nrof_images_per_pair
#     nrof_flips = 2 if use_flipped_images else 1
#     nrof_images = nrof_embeddings * nrof_flips
#     labels_array = np.expand_dims(np.arange(0, nrof_images), 1)
#     image_paths_array = np.expand_dims(
#         np.repeat(np.array(image_paths), nrof_flips), 1)
#     control_array = np.zeros_like(labels_array, np.int32)
#     if use_fixed_image_standardization:
#         control_array += np.ones_like(
#             labels_array) * facenet.FIXED_STANDARDIZATION
#     if use_flipped_images:
#         # Flip every second image
#         control_array += (labels_array % 2) * facenet.FLIP
#     sess.run(
#         enqueue_op, {
#             image_paths_placeholder: image_paths_array,
#             labels_placeholder: labels_array,
#             control_placeholder: control_array
#         })

#     embedding_size = int(embeddings.get_shape()[1])
#     print(nrof_images)
#     assert nrof_images % batch_size == 0, 'The number of LFW images must be an integer multiple of the LFW batch size'
#     nrof_batches = nrof_images // batch_size
#     emb_array = np.zeros((nrof_images, embedding_size))
#     lab_array = np.zeros((nrof_images, ))
#     for i in range(nrof_batches):
#         feed_dict = {
#             phase_train_placeholder: False,
#             batch_size_placeholder: batch_size
#         }
#         emb, lab = sess.run([embeddings, labels], feed_dict=feed_dict)
#         lab_array[lab] = lab
#         emb_array[lab, :] = emb
#         if i % 10 == 9:
#             print('.', end='')
#             sys.stdout.flush()
#     embeddings = np.zeros((nrof_embeddings, embedding_size * nrof_flips))
#     if use_flipped_images:
#         # Concatenate embeddings for flipped and non flipped version of the images
#         embeddings[:, :embedding_size] = emb_array[0::2, :]
#         embeddings[:, embedding_size:] = emb_array[1::2, :]
#     else:
#         embeddings = emb_array

#     assert np.array_equal(
#         lab_array, np.arange(nrof_images)
#     ) == True, 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'
#     tpr, fpr, accuracy, val, val_std, far = own.evaluate(
#         embeddings,
#         actual_issame,
#         nrof_folds=nrof_folds,
#         distance_metric=distance_metric,
#         subtract_mean=subtract_mean)

#     print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
#     print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

#     auc = metrics.auc(fpr, tpr)
#     print('Area Under Curve (AUC): %1.3f' % auc)
#     eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
#     print('Equal Error Rate (EER): %1.3f' % eer)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'patch_dir',
        type=str,
        help='Path to the data directory containing patches.')
    parser.add_argument(
        '--lfw_pairs',
        type=str,
        help='The file containing the pairs to use for validation.',
        default='data/own_pairs.txt')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
