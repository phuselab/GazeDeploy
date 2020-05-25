from __future__ import print_function
import numpy as np
import scipy.io as sio


class Gaze(object):
	"""Class to handle the datasets"""
 
	def __init__(self, gazeDir=None):
        
		self.gazeDir = gazeDir

	def load_gaze_data(self, video_name, data='find'):

		if data == 'coutrot':

			vName = video_name[6:-4]

			file = open(self.gazeDir+'Original_'+vName+'.txt', "r") 
			eyedata = file.read()
			eyedata = np.reshape(np.array(eyedata.split()),(36,-1))
			eyedata = np.reshape(eyedata, (2, 18, -1), order='F')
			eyedata = np.moveaxis(eyedata, (0,1,2), (0,2,1))

			self.video_name = video_name[:-4]
			self.eyedata = eyedata

		elif data == 'find':

			'''
			OLD!!!!!
			mat = sio.loadmat(self.gazeDir+video_name[:-4])
			eyedata = mat['fix_data']
			eyedata = np.moveaxis(eyedata, (0,1,2), (0,2,1))
			
			self.video_name = video_name[:-4]
			self.eyedata = eyedata'''

			mat = sio.loadmat(self.gazeDir+video_name[:-4])
			#eyedata = mat['fix_data']
			eyedata_all = mat['curr_v_all_s']

			n_subs = eyedata_all.shape[0]
			s_frames = eyedata_all[0][0].shape[0]

			eyedata = np.zeros([2,n_subs,s_frames])

			for s in range(n_subs):
				eyedata[:,s,:] = eyedata_all[s][0].T

			eyedata = np.moveaxis(eyedata, (0,1,2), (0,2,1))
			
			self.video_name = video_name[:-4]
			self.eyedata = eyedata
		else:

			raise ValueError("data should be 'coutrot' or 'find'")

		return eyedata