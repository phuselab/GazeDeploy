#from __future__ import print_function
import cv2
#import os
import numpy as np
from gaze import Gaze
from video import Video
from feature_maps import Feature_maps
from sklearn.linear_model import LinearRegression
from utils import normalize, center, compute_density_image, softmax
import matplotlib.pyplot as plt
from pylab import *
#from drawnow import drawnow
from MVT_gaze_sampler import GazeSampler
from skimage.draw import circle
#from utils import compute_density_image
#import os.path
import imageio
from pykalman import KalmanFilter

def compute_patch_measures(featMaps):
	patches = []
	for fmap in featMaps.all_fmaps: #For every feature map
		if fmap.name == 'Uniform':
				continue
		for patch in fmap.patches:	#For every patch
			#patch.compute_expected_reward()
			patches.append(patch)
	return patches

def get_fix_from_scan(scan_dict, nFrames):
	generated_eyedata = np.zeros([2, nFrames, len(scan_dict)])
	fixations = np.zeros([2, nFrames])
	for i,k in enumerate(scan_dict.keys()):
		s = scan_dict[k]
		N = s.shape[0] // 10
		frames = np.split(s, N)
		#fixations = np.median(frames, axis=1)
		for j, f in enumerate(frames):
			if j < nFrames:
				med = np.median(f, axis=0)
				fixations[:,j] = np.median(f, axis=0)
		generated_eyedata[:,:,i] = fixations
	return generated_eyedata


#Directories
vidDir = 'data/videos/'
gazeDir = 'data/fix_data/'
dynmapDir = 'speaker_detect/data/ST_maps/'
facemapDir = 'speaker_detect/data/face_maps/'

gazeObj = Gaze(gazeDir)
videoObj = Video(vidDir)
featMaps = Feature_maps(dynmapDir, facemapDir)

curr_vid_name = '012.mp4'
scanPath = {}
d_rate = 1.5

#Gaze -------------------------------------------------------------------------------------------------------		
gazeObj.load_gaze_data(curr_vid_name)

#Video ------------------------------------------------------------------------------------------------------	
videoObj.load_video(curr_vid_name)
FOAsize = int(np.max(videoObj.size)/10)

#Feature Maps ----------------------------------------------------------------------------------------------- 
featMaps.load_feature_maps(curr_vid_name, videoObj.vidHeight, videoObj.vidWidth)

#Gaze Sampler -----------------------------------------------------------------------------------------------
gazeSampler = GazeSampler(videoObj.frame_rate) 

nFrames = min([len(videoObj.videoFrames), featMaps.num_sts, featMaps.num_speak, featMaps.num_nspeak])
wd = int(videoObj.vidWidth * d_rate / 100)
hd = int(videoObj.vidHeight * d_rate / 100)
tot_dim = wd*hd

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
betas = np.zeros((nFrames, 5))

#generated_scan = np.load('data/saved/gen_gaze/'+curr_vid_name[:-4]+'_DoubleValue_exp18_alpha65_18viewers.npy', allow_pickle=True).item()
generated_scan = np.load('data/gen_gaze/'+curr_vid_name[:-4]+'.npy', allow_pickle=True).item()
generated_eyedata = get_fix_from_scan(generated_scan, nFrames)

fig = plt.figure(figsize=(16, 10))
images = []
draw_fig = True
display = False
save_GIF = True

start_gif = 100
end_gif = 300

#For each video frame
for iframe in range(nFrames):

	if iframe < start_gif:
		continue
	if iframe > end_gif:
		break

	print('\nFrame number: ' + str(iframe))
	#Variables Initialization 
	frame = videoObj.videoFrames[iframe]
	SampledPointsCoord = []

	featMaps.read_current_maps(gazeObj.eyedata, iframe, 1, d_rate=d_rate)
	y = np.squeeze(normalize(np.reshape(featMaps.eyeMap_train,[-1,1], order='F')))
	X = normalize(featMaps.X)

	if iframe == start_gif:
		regr_model = LinearRegression().fit(X,y)
		filtered_state_means[iframe] = regr_model.coef_
		filtered_state_covariances[iframe] = np.ones((5, 5))
	else:
		filtered_state_means[iframe], filtered_state_covariances[iframe] = kf.filter_update(filtered_state_mean=filtered_state_means[iframe-1], 
		                                                                                    filtered_state_covariance=filtered_state_covariances[iframe-1], 
		                                                                                    observation=y,
		                                                                                    observation_matrix=X)

	beta1 = softmax(filtered_state_means[iframe])
	beta2 = softmax(filtered_state_means[iframe]*2.5)

	CBValue = beta1[1]
	CBExpVal = beta2[1]
	featMaps.cb.esSampleProtoParameters()
	featMaps.cb.define_patches(CBValue, CBExpVal)

	#Speaker saliency and proto maps -------------------------------------------------------------------------
	speakerValue = beta1[3]
	speakerExpVal = beta2[3]
	featMaps.speaker.esSampleProtoParameters()
	featMaps.speaker.define_patches(speakerValue, speakerExpVal)

	#Non Speaker saliency and proto maps ---------------------------------------------------------------------		
	nspeakerValue = beta1[4]
	nspeakerExpVal = beta2[4]
	featMaps.non_speaker.esSampleProtoParameters()
	featMaps.non_speaker.define_patches(nspeakerValue, nspeakerExpVal)
	
	#Low Level Saliency saliency and proto maps ---------------------------------------------------------------		
	stsValue = beta1[0]
	stsExpVal = beta2[0]
	featMaps.sts.esSampleProtoParameters()
	featMaps.sts.define_patches(stsValue, stsExpVal)

	betas[iframe-start_gif, :] = beta1

	#Patch Measures -------------------------------------------------------------------------------------------
	patches = compute_patch_measures(featMaps)

	#Gaze Sampling --------------------------------------------------------------------------------------------
	gazeSampler.sample(iframe=iframe-start_gif, patches=patches, FOAsize=FOAsize//12)

	curr_fix = generated_eyedata[:,iframe,:].T
	gen_saliency = compute_density_image(curr_fix, [videoObj.vidWidth, videoObj.vidHeight])

	if draw_fig:
		nRows = 2
		nCols = 3

		fig.clf()

		numfig=1
		plt.subplot(nRows,nCols,numfig)
		plt.imshow(frame)
		plt.title('Original Frame')
		
		numfig+=1
		plt.subplot(nRows,nCols,numfig)
		plt.imshow(featMaps.original_eyeMap)
		plt.title('"Real" Fixation Map')
		
		sp = betas[:iframe,3]
		nsp = betas[:iframe,4]
		cb = betas[:iframe,1]
		sts = betas[:iframe,0]
		uni = betas[:iframe,2]
		npoints = len(sp)
		numfig+=1
		plt.subplot(nRows,nCols,numfig)
		plt.plot(sp, label='speaker')
		plt.plot(nsp, label='Non speaker')
		plt.plot(cb, label='CB')
		plt.plot(sts, label='STS')
		plt.plot(uni, label='Uniform')
		plt.legend()
		plt.grid()
		plt.ylim(0, 0.4)
		plt.title('Value')

		numfig+=1
		finalFOA = gazeSampler.allFOA[-1].astype(int)
		plt.subplot(nRows,nCols,numfig)
		BW = np.zeros(videoObj.size)
		rr,cc = circle(finalFOA[1], finalFOA[0], FOAsize)
		rr[rr>=BW.shape[0]] = BW.shape[0]-1
		cc[cc>=BW.shape[1]] = BW.shape[1]-1
		BW[rr,cc] = 1
		FOAimg = cv2.bitwise_and(cv2.convertScaleAbs(frame),cv2.convertScaleAbs(frame),mask=cv2.convertScaleAbs(BW))
		plt.imshow(FOAimg)
		plt.title('Focus Of Attention (FOA)')

		#Heat Map
		numfig+=1
		plt.subplot(nRows,nCols,numfig)
		plt.imshow(gen_saliency)
		plt.title('Generated Fixation Map')

		#Scan Path
		numfig+=1
		plt.subplot(nRows,nCols,numfig)
		plt.imshow(frame)
		sampled = np.concatenate(gazeSampler.sampled_gaze)
		plt.plot(sampled[:,0], sampled[:,1], '-x')
		plt.title('Generated Gaze data')

		fig.canvas.draw()
		image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
		image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
		if iframe >= 100:
			if save_GIF:
				images.append(image)
		if iframe > 300:
			break

		if display:
			plt.pause(1/25.)

	#At the end of the loop
	featMaps.release_fmaps()


#kwargs_write = {'fps':10.0, 'quantizer':'nq'}
if save_GIF:
	imageio.mimsave('simulation_' + curr_vid_name[:-4] + '.gif', images, fps=10)
