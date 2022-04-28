"""
Name:  Giuliani Martinez Herrera
Email: giuliani.martinezherrer04@myhunter.cuny.edu
Resources:  Used python.org as a reminder of Python 3 print statements.
Digit Dimensions
"""
from sklearn.decomposition import PCA
import numpy as np

def select_data(data, target, labels = [0,1]):
    """
    :param data: a numpy array that includes rows of equal size flattened arrays,
    :param target: a numpy array that contains the labels for each row in data.
    :param labels: the labels from target that the rows to be selected.  The default value is [0,1].
    :return: the rows of data and target where the value of target is in labels.
    """
    selected = [(d,t) for (d,t) in zip(data,target) if t in labels]
    d_sel,t_sel = zip(*selected)
    return d_sel, t_sel

def run_pca(xes):
    """
    :param xes: np array of rows of flattened arrays
    :return: returns pca model and values
    """
    pca = PCA()
    return pca, pca.fit_transform(xes)

def capture_85(mod):
    """
    :param mod: pca model fitted to values
    :return: # of elems needed to capture >85% variance
    """
    sv = mod.singular_values_
    s=sv**2/sum(sv**2)
    elem = 0
    var_ = 0
    for x in range(len(s)):
        elem = elem + 1
        var_ = var_ + s[x]
        if var_ > .85:
            return elem

def average_eigenvalue(mod):
    """
    :param mod: pca model fitted to values
    :return: # of elems greater than average
    """
    sv = mod.singular_values_
    ave = np.mean(sv)
    elem = 0
    for x in range(len(sv)):
        if sv[x] > ave:
            elem =  elem + 1
    return elem

def approx_digits(mod, img, numComponents=8):
    """
    :param mod:pca model fitted to dataset
    :param img: flatend image
    :param numComponents: 0<x<64
    :return: approximation image as a flattend array
    """
    # pca = PCA(n_components=numComponents)
    # img_val = pca.fit_transform(img)
    t_img = img[:numComponents]
    for x in range(len(t_img)):
        t_img[x] = t_img[x]*mod.components_[x]
    t_img.insert(0,mod.mean_)
    return t_img