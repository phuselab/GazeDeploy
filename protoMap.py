import cv2
import numpy as np
import matplotlib.pyplot as plt
#from pinky import Pinky
from utils import map_in_range

class ProtoMap(object):
	"""Class to handle the datasets"""
 
	def __init__(self, protoMap=None, name=None):
        
		self.protoMap = protoMap.T
		self.name = name
		self.extent = np.array([0, protoMap.shape[0]-1, 0, protoMap.shape[1]-1])

	@property
	def shape(self):
		return self.protoMap.shape

	@property
	def value(self):
		return self.protoMap

	'''def InterestPoint_Sampling(self, value):

		NUM_MAX_IP = 500
		
		max_points = int(np.round(NUM_MAX_IP * value))

		if max_points > 0:
			p = Pinky(P=self.protoMap, extent=self.extent)
			sampled_points = p.sample(max_points)
			sampled_points = np.flip(sampled_points,1).astype(int)
		else:
			sampled_points = np.array([])

		SampledPointsCoordPMValue = np.ones(len(sampled_points))*value
		for s in range(len(sampled_points)):
			SampledPointsCoordPMValue[s] = SampledPointsCoordPMValue[s] * self.protoMap[sampled_points[s][0],sampled_points[s][1]]

		self.SampledPointsCoordPM = sampled_points
		self.SampledPointsCoordPMValue = SampledPointsCoordPMValue'''

