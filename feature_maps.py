#from __future__ import print_function
import numpy as np
import os
import glob
from scipy.stats import multivariate_normal
from utils import sorted_nicely, mkGaussian, compute_density_image, clean_eyedata, normalize, center
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from feat_map import Feat_map
from math import sqrt

'''def get_gauss_params(img): #img is grayscale image of what I want to fit
		
		ret,thresh = cv2.threshold(img,127,255,0)
		_,contours,hierarchy = cv2.findContours(thresh, 1, 2)

		g_params = []

		if len(contours) != 0:
			for cont in contours:
				elps = cv2.fitEllipse(cont)
				mu = np.flip(np.array(elps[0]), axis=0)
				sigma = np.flip(np.array(elps[1])*0.25, axis=0)
				theta = elps[2]
				p = (mu, sigma, theta)
				g_params.append(p)
		
		return g_params, len(g_params)'''

class Feature_maps(object):
	"""Class to handle the datasets"""
 
	def __init__(self, dynDir=None, facemapDir=None):
        
		self.dynDir = dynDir
		self.facemapDir = facemapDir
		self.all_fmaps = []

	def release_fmaps(self):
		self.all_fmaps = []

	def load_feature_maps(self, video_name, vidHeight=None, vidWidth=None):

		sts_saliency_maps = os.listdir(self.dynDir + video_name[:-4])
		sorted_sts_saliency_maps = sorted_nicely(sts_saliency_maps)
		self.num_sts = len(sorted_sts_saliency_maps)

		face1_maps = glob.glob(self.facemapDir + video_name[:-4] + '/*_speaker.png')
		sorted_face1_maps = sorted_nicely(face1_maps)
		self.num_speak = len(sorted_face1_maps)

		face2_maps = glob.glob(self.facemapDir + video_name[:-4] + '/*_nonspeaker.png')
		sorted_face2_maps = sorted_nicely(face2_maps)
		self.num_nspeak = len(sorted_face2_maps)
		
		#mu = np.array([vidWidth/2,vidHeight/2])
		mu = np.array([vidWidth/2,vidHeight/2])
		wy=vidHeight/12	
		wx=vidWidth/12
		sigma = [wx, wy]
		
		F = mkGaussian(mu, sigma, 0, vidWidth, vidHeight).T
		center_bias_map = Feat_map(feat_map=F/np.sum(np.sum(F)), name='CB')
		
		uniform_map = Feat_map(feat_map=np.ones(center_bias_map.shape)/np.prod(center_bias_map.shape), name='Uniform')

		self.sts_names = sorted_sts_saliency_maps
		self.face1_names = sorted_face1_maps
		self.face2_names = sorted_face2_maps
		self.cb = center_bias_map
		self.uniform = uniform_map
		self.video_name = video_name[:-4]
		self.vidHeight = vidHeight
		self.vidWidth = vidWidth

		return sorted_sts_saliency_maps, sorted_face1_maps, sorted_face2_maps, center_bias_map, uniform_map


	def read_current_maps(self, gaze, frame_num, train_perc, d_rate=None, downsample=True):

		if not downsample:
			d_rate = 100

		w = self.vidWidth
		h = self.vidHeight
		nMaps = 5

		curr_sts = cv2.imread(self.dynDir + self.video_name + '/' + self.sts_names[frame_num], 0)
		
		wd = int(curr_sts.shape[1] * d_rate / 100)
		hd = int(curr_sts.shape[0] * d_rate / 100)
		dim = (wd, hd) 

		self.sts = Feat_map(feat_map=curr_sts/float(np.sum(curr_sts)), name='STS')
		if downsample:
			curr_sts = cv2.resize(curr_sts, dim, interpolation=cv2.INTER_AREA)
	
		curr_sts = np.reshape(curr_sts,-1, order='F')

		newShape = wd*hd 		#New Shape after downsampling
		X = np.empty([newShape, nMaps])
		
		curr_face1 = cv2.imread(self.face1_names[frame_num], 0)

		if np.sum(curr_face1) != 0:
			self.speaker = Feat_map(feat_map=curr_face1/float(np.sum(curr_face1)), name='Speaker')
			if downsample:
				curr_face1 = cv2.resize(curr_face1, dim, interpolation=cv2.INTER_AREA)
			curr_face1 = np.reshape(curr_face1,-1, order='F')/np.max(curr_face1)
		else:
			self.speaker = Feat_map(feat_map=curr_face1, name='Speaker')
			if downsample:
				curr_face1 = cv2.resize(curr_face1, dim, interpolation=cv2.INTER_AREA)
			curr_face1 = np.reshape(curr_face1,-1, order='F')
		
		curr_face2 = cv2.imread(self.face2_names[frame_num], 0)

		
		if np.sum(curr_face2) != 0:
			self.non_speaker = Feat_map(feat_map=curr_face2/float(np.sum(curr_face2)), name='non_Speaker')
			if downsample:
				curr_face2 = cv2.resize(curr_face2, dim, interpolation=cv2.INTER_AREA)
			curr_face2 = np.reshape(curr_face2,-1, order='F')/np.max(curr_face2)
		else:
			self.non_speaker = Feat_map(feat_map=curr_face2, name='non_Speaker')
			if downsample:
				curr_face2 = cv2.resize(curr_face2, dim, interpolation=cv2.INTER_AREA)
			curr_face2 = np.reshape(curr_face2,-1, order='F')

		self.all_fmaps.append(self.uniform)
		self.all_fmaps.append(self.cb)
		self.all_fmaps.append(self.sts)
		self.all_fmaps.append(self.speaker)
		self.all_fmaps.append(self.non_speaker)

		if downsample:
			cbValue = cv2.resize(self.cb.value, dim, interpolation=cv2.INTER_AREA)
			unifValue = cv2.resize(self.uniform.value, dim, interpolation=cv2.INTER_AREA)
		else:
			cbValue = self.cb.value
			unifValue = self.uniform.value
		
		cb_reshapaed = np.reshape(cbValue,-1, order='F')
		uniform_reshaped = np.reshape(unifValue,-1, order='F')

		X[:,0] =  curr_sts
		X[:,1] =  cb_reshapaed
		X[:,2] =  uniform_reshaped
		X[:,3] = curr_face1
		X[:,4] = curr_face2

		curr_gaze = clean_eyedata(gaze[:,frame_num,:], w, h)

		nObsTrain = int(np.floor(curr_gaze.shape[0]*train_perc))
		Eye_Position_Map_train = np.zeros([w,h])
		Eye_Position_Map_test = np.zeros([w,h])

		curr_gaze_train_ind = curr_gaze[:nObsTrain,:].astype(int)
		curr_gaze_test_ind = curr_gaze[nObsTrain:,:].astype(int)

		Eye_Position_Map_train = compute_density_image(curr_gaze_train_ind, [w,h])
		if train_perc < 1:
			Eye_Position_Map_test = compute_density_image(curr_gaze_test_ind, [w,h])
		else:
			Eye_Position_Map_test = np.zeros([w,h])

		self.X = X

		self.original_eyeMap = Eye_Position_Map_train.copy()
		
		if downsample:
			Eye_Position_Map_train = cv2.resize(Eye_Position_Map_train, dim, interpolation=cv2.INTER_AREA)
		
		self.eyeMap_train = Eye_Position_Map_train


		return X, Eye_Position_Map_train, Eye_Position_Map_test