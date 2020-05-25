from __future__ import print_function
import cv2
import os
import numpy as np
from gaze import Gaze
from video import Video
from feature_maps import Feature_maps
from sklearn.linear_model import LinearRegression
from utils import normalize, center, compute_density_image, softmax
import matplotlib.pyplot as plt
from pylab import *
from drawnow import drawnow
from MVT_gaze_sampler import GazeSampler
from skimage.draw import circle
from sklearn.neighbors import KernelDensity
from pykalman import KalmanFilter
import os.path

def draw_fig():
	nRows = 2
	nCols = 2

	numfig=1
	subplot(nRows,nCols,numfig)
	imshow(frame)
	title('Original Frame')
	
	numfig+=1
	subplot(nRows,nCols,numfig)
	imshow(featMaps.eyeMap_train)
	title('Real Map')
	
	numfig+=1
	finalFOA = gazeSampler.allFOA[-1].astype(int)
	subplot(nRows,nCols,numfig)
	BW = np.zeros(videoObj.size)
	rr,cc = circle(finalFOA[1], finalFOA[0], FOAsize//8)
	rr[rr>=BW.shape[0]] = BW.shape[0]-1
	cc[cc>=BW.shape[1]] = BW.shape[1]-1
	BW[rr,cc] = 1
	FOAimg = cv2.bitwise_and(cv2.convertScaleAbs(frame),cv2.convertScaleAbs(frame),mask=cv2.convertScaleAbs(BW))
	imshow(FOAimg)

	#Scan Path
	numfig+=1
	subplot(nRows,nCols,numfig)
	imshow(frame)
	sampled = np.concatenate(gazeSampler.sampled_gaze)
	plot(sampled[:,0], sampled[:,1], '-x')
	title('Generated Scan Path')

def compute_patch_measures(featMaps):
	patches = []
	for fmap in featMaps.all_fmaps: #For every feature map
		if fmap.name == 'Uniform':
				continue
		for patch in fmap.patches:	#For every patch
			patches.append(patch)
	return patches

# -------------------------------- Choose data -------------------------------------

vidDir = 'data/raw_videos/'
gazeDir = 'data/fix_data_NEW/'
dynmapDir = 'data/dynamic/'
facemapDir = 'data/merged_maps/'

# -------------------------------- Choose Video ---------------------------------------

chosenVid = ['072.mp4']

# ------------------------------------- Main ------------------------------------------

gazeObj = Gaze(gazeDir)
videoObj = Video(vidDir)
featMaps = Feature_maps(dynmapDir, facemapDir)
videoNames = os.listdir(vidDir)

# Params ------------------------------------------------------------------------------

n_subj = 1		#Number of subjects to simulate
d_rate = 1.5		#downsampling rate (to speed up kalman filtering)
display = True

#For each video
for curr_vid_name in videoNames:

	scanPath = {}
	fname = 'saved/gen_gaze/'+curr_vid_name[:-4] + '_DoubleValue_exp18_alpha65_18viewers.npy'

	if not curr_vid_name.endswith('.mp4'):
		continue
	if curr_vid_name not in chosenVid:
		continue

	print('\n\n\t\t' + curr_vid_name + '\n')

	#Gaze -------------------------------------------------------------------------------------------------------
	gazeObj.load_gaze_data(curr_vid_name)

	#Video ------------------------------------------------------------------------------------------------------	
	videoObj.load_video(curr_vid_name)
	FOAsize = int(np.max(videoObj.size)/6)

    #Feature Maps ----------------------------------------------------------------------------------------------- 
	featMaps.load_feature_maps(curr_vid_name, videoObj.vidHeight, videoObj.vidWidth)

	nFrames = min([len(videoObj.videoFrames), featMaps.num_sts, featMaps.num_speak, featMaps.num_nspeak, gazeObj.eyedata.shape[1]])
	
	wd = int(videoObj.vidWidth * d_rate / 100)		#new Width
	hd = int(videoObj.vidHeight * d_rate / 100)		#new Height
	tot_dim = wd*hd

	#For each subject
	for subj in range(n_subj):	

		print('\n\tSUBJECT NUMBER: ' + str(subj) + ' \n')

		#Gaze Sampler -----------------------------------------------------------------------------------------------
		gazeSampler = GazeSampler(videoObj.frame_rate)

		#Set up Kalman Filter ---------------------------------------------------------------------------------------
		initial_state_mean = np.zeros(5)
		initial_state_mean[1] = 1
		trans_cov = np.eye(5)
		vidHeight = int(featMaps.vidHeight * d_rate)
		vidWidth = int(featMaps.vidWidth * d_rate)
		n_dim_obs = vidHeight * vidWidth

		kf = KalmanFilter(n_dim_obs=tot_dim, n_dim_state=5,
		                  initial_state_mean=initial_state_mean,
		                  initial_state_covariance=np.ones((5, 5)),
		                  transition_matrices=np.eye(5),
		                  transition_covariance=trans_cov)

		filtered_state_means = np.zeros((nFrames, 5))
		filtered_state_covariances = np.zeros((nFrames, 5, 5))

		#For each video frame
		for iframe in range(nFrames):

			print('\nFrame number: ' + str(iframe))
			frame = videoObj.videoFrames[iframe]
			SampledPointsCoord = []

			featMaps.read_current_maps(gazeObj.eyedata, iframe, 1, d_rate=d_rate)
			y = np.squeeze(normalize(np.reshape(featMaps.eyeMap_train,[-1,1], order='F')))
			X = normalize(featMaps.X)

			if iframe == 0:
				regr_model = LinearRegression().fit(X,y)
				filtered_state_means[iframe] = regr_model.coef_
				filtered_state_covariances[iframe] = np.ones((5, 5))
			else:
				filtered_state_means[iframe], filtered_state_covariances[iframe] = kf.filter_update(filtered_state_mean=filtered_state_means[iframe-1], 
				                                                                                    filtered_state_covariance=filtered_state_covariances[iframe-1], 
				                                                                                    observation=y,
				                                                                                    observation_matrix=X)

			beta1 = softmax(filtered_state_means[iframe])
			beta2 = softmax(filtered_state_means[iframe]*4)

			#Center Bias proto map
			CBValue = beta1[1]
			CBExpVal = beta2[1]
			featMaps.cb.esSampleProtoParameters()
			featMaps.cb.define_patches(CBValue, CBExpVal)

			#Speaker(s) proto maps -------------------------------------------------------------------------
			speakerValue = beta1[3]
			speakerExpVal = beta2[3]
			featMaps.speaker.esSampleProtoParameters()
			featMaps.speaker.define_patches(speakerValue, speakerExpVal)

			#Non Speaker(s) proto maps ---------------------------------------------------------------------		
			nspeakerValue = beta1[4]
			nspeakerExpVal = beta2[4]
			featMaps.non_speaker.esSampleProtoParameters()
			featMaps.non_speaker.define_patches(nspeakerValue, nspeakerExpVal)
			
			#Low Level Saliency proto maps ---------------------------------------------------------------		
			stsValue = beta1[0]
			stsExpVal = beta2[0]
			featMaps.sts.esSampleProtoParameters()
			featMaps.sts.define_patches(stsValue, stsExpVal)

			#Patch Measures -------------------------------------------------------------------------------------------
			patches = compute_patch_measures(featMaps)
		
			#Gaze Sampling --------------------------------------------------------------------------------------------
			gazeSampler.sample(iframe=iframe, patches=patches, FOAsize=FOAsize//8, verbose=True)

			if display:
				#play results
				drawnow(draw_fig, stop_on_close=True)

			#At the end of the loop
			featMaps.release_fmaps()

		scanPath[subj] = np.concatenate(gazeSampler.sampled_gaze)

	np.save(fname, scanPath)
