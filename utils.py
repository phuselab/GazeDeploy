#from __future__ import print_function
import numpy as np
import re
import time
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.neighbors import KernelDensity
from scipy.ndimage import gaussian_filter

def sorted_nicely(l):
	""" Sorts the given iterable in the way that is expected.
	Required arguments:
	l -- The iterable to be sorted.
	"""
	convert = lambda text: int(text) if text.isdigit() else text
	alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
	
	return sorted(l, key = alphanum_key)


def center(X):
	n = X.shape[0]
	mu = np.expand_dims(np.mean(X,axis=0), axis=0)
	mu_matr = np.matmul(np.ones([n,1]),mu)
	
	return X-mu_matr

def normalize(X):
	n = X.shape[0]
	X = center(X)
	d = np.expand_dims(np.sqrt(np.sum(np.square(X), axis=0)), axis=0)

	d[d==0] = 1
	X = np.divide(X,np.matmul(np.ones([n,1]),d))

	return X

def mkGaussian(mu, sigma, theta, w, h):
	x1 = np.linspace(0, w, w)
	x2 = np.linspace(0, h, h)
	X1,X2 = np.meshgrid(x1,x2)
	pos = np.empty(X1.shape + (2,))
	pos[:, :, 0] = X1
	pos[:, :, 1] = X2

	theta = np.radians(theta)
	c, s = np.cos(theta), np.sin(theta)
	R = np.array(((c,-s), (s, c)))
	R = np.matrix(R)

	sigma_diag = np.matrix(np.square(np.diag(sigma)*0.5))
	#sigma_diag = np.matrix(np.square(np.diag(sigma)))

	Sigma = R*sigma_diag*R.T

	mvn = multivariate_normal(mu,Sigma)
	F = mvn.pdf(pos)

	return F

def map_in_range(X, target_range):
		
	min_new1 = 0
	max_new1 = target_range[0]-1
	min_new2 = 0
	max_new2 = target_range[1]-1

	min_old1 = -1
	max_old1 = 1
	min_old2 = min_old1
	max_old2 = max_old1

	x1 = X[:,0]
	x2 = X[:,1]

	x1_scaled = ((max_new1-min_new1)/(max_old1-min_old1))*(x1-max_old1) + max_new1
	x2_scaled = ((max_new2-min_new2)/(max_old2-min_old2))*(x2-max_old2) + max_new2

	X_scaled = np.stack([x1_scaled,x2_scaled], axis=1).astype(int)

	return X_scaled

def compute_density_image(points, size, method='conv'):

	points = np.flip(points,1)

	if method == 'KDEsk':

		w = size[0]
		h = size[1]
		sigma=1/0.039
		#sigma = 18.4

		x1 = np.linspace(0, w, w)
		x2 = np.linspace(0, h, h)
		X1,X2 = np.meshgrid(x1,x2)
		positions = np.vstack([X1.ravel(), X2.ravel()]).T
		
		kde_skl = KernelDensity(bandwidth=sigma)
		kde_skl.fit(points)

		Z = np.exp(kde_skl.score_samples(positions))
		Z = np.reshape(Z, X1.shape).T

		#print(Z.shape)
		#plt.imshow(Z.T)
		#plt.show()

	elif method == 'conv':

		sigma=1/0.039
		H, xedges, yedges = np.histogram2d(points[:,0], points[:,1], bins=(range(size[0]+1), range(size[1]+1)))
		Z = gaussian_filter(H, sigma=sigma)
		Z = Z/float(np.sum(Z))
	return Z

def clean_eyedata(eyedata, w, h, check_out_of_video=False):
	eyedata = np.ndarray.astype(eyedata,np.float_)
	eyedata = eyedata[:,~np.any(np.isnan(eyedata),axis=0)]
	eyedata = eyedata[:,~np.any(eyedata<0,axis=0)]
	
	if check_out_of_video:
		eyedata = eyedata[:,~np.any(eyedata[0,:]>w,axis=0)]
		eyedata = eyedata[:,~np.any(eyedata[1,:]>h,axis=0)]
	
	
	eyedata = eyedata.T

	return eyedata

def softmax(x):
	"""Compute softmax values for each sets of scores in x."""
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()