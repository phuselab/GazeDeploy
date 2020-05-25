#from __future__ import print_function
import cv2
import skvideo.io

class Video(object):
	"""Class to handle the datasets"""
 
	def __init__(self, vidDir=None):
        
		self.vidDir = vidDir

	def load_video(self, video_name, vidLib='skvideo'):

		videoFrames = {}

		if vidLib == 'ocv':
			cap = cv2.VideoCapture(self.vidDir+video_name)
			# Check if camera opened successfully
			if (cap.isOpened()== False): 
				print("\nError opening video stream or file!!!!!\n")

			# Read until video is completed
			frameNum = 0
			while(cap.isOpened()):
				# Capture frame-by-frame
				ret, frame = cap.read()
				if ret == True:

					videoFrames[frameNum] = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
					frameNum += 1
				else:
					break
			# When everything done, release the video capture object
			cap.release()

			vidHeight = videoFrames[0].shape[1]
			vidWidth = videoFrames[0].shape[0]

			self.videoFrames = videoFrames
			self.vidHeight = vidHeight
			self.vidWidth = vidWidth
			self.video_name = video_name
			self.size = [vidWidth, vidHeight]

		elif vidLib == 'skvideo':
			videogen = skvideo.io.vreader(self.vidDir+video_name)

			for frameNum, frame in enumerate(videogen):			
				videoFrames[frameNum] = frame

			vidHeight = videoFrames[0].shape[1]
			vidWidth = videoFrames[0].shape[0]

			videometadata = skvideo.io.ffprobe(self.vidDir+video_name)
			self.frame_rate = eval(videometadata['video']['@avg_frame_rate'])

			self.videoFrames = videoFrames
			self.vidHeight = vidHeight
			self.vidWidth = vidWidth
			self.video_name = video_name
			self.size = [vidWidth, vidHeight]

		return videoFrames, vidHeight, vidWidth