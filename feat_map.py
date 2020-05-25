import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import mkGaussian
from skimage import measure
from protoMap import ProtoMap
from patch import Patch
from skimage.draw import ellipse_perimeter, polygon
from operator import itemgetter

class Feat_map(object):
 
	def __init__(self, feat_map=None, name=None):
        
		self.feat_map = feat_map
		self.name = name

	@property
	def shape(self):
		return self.feat_map.shape

	@property
	def value(self):
		return self.feat_map

	def esSampleNbestProto(self, protoMap_raw, nBest, win):

		contours = measure.find_contours(protoMap_raw, 0.5)

		size_c = []
		for i,c in enumerate(contours):
			size_c.append(c.shape[0])
		sort_idx = np.argsort(np.array(size_c))[::-1]

		M_tMap = np.zeros(protoMap_raw.shape)
		nBest = min(nBest, len(contours))

		for i in range(nBest):
			img = np.zeros(protoMap_raw.shape)
			r = contours[sort_idx[i]][:,0]
			c = contours[sort_idx[i]][:,1]
			rr, cc = polygon(r, c)
			img[rr, cc] = 255
			img = img/np.max(img)	
			M_tMap = M_tMap + img

		return M_tMap

	def esSampleProtoMap(self, nBestProto):

		protoMap_raw = np.zeros(self.feat_map.shape)
		normSal = self.feat_map
		maxsal = np.max(normSal)
		minsal = np.min(normSal)
		normsal = np.divide((normSal-minsal),(maxsal-minsal))
		normSal = normSal*100

		ind = np.stack(np.where(normsal >= np.percentile(normsal,95)), axis=-1)

		protoMap_raw[ind[:,0], ind[:,1]] = 1
		
		openingwindow=15
		
		M_tMap = self.esSampleNbestProto(protoMap_raw,nBestProto,openingwindow)

		protoMap = np.logical_not(M_tMap)

		return M_tMap, protoMap
		
	def esSampleProtoParameters(self):

		if self.name == 'STS':
			nBestProto = 1
			M_tMap, protoMap = self.esSampleProtoMap(nBestProto)
			feat_map_img = M_tMap*255
		else:
			if np.max(self.feat_map) != 0:
				feat_map_img = (self.feat_map/np.max(self.feat_map))*255
			else:
				feat_map_img = self.feat_map

		_,thresh = cv2.threshold(cv2.convertScaleAbs(feat_map_img),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		B = measure.find_contours(thresh, 128)#0.5)

		if type(B).__module__ == np.__name__:
			B = [B]

		#Remove too tiny contours
		idx_to_keep = []
		for i,b in enumerate(B):
			if b.shape[0] >= 40:
				idx_to_keep.append(i)

		if len(B) > 1:
			B = itemgetter(*idx_to_keep)(B)

		if type(B).__module__ == np.__name__:
			B = [B]

		numproto = len(B)

		a = {}
		r1 = {}
		r2 = {}
		cx = {}
		cy = {}
		theta = {}
		boundaries = {}

		discarded = 0
		for p in range(numproto):
			boundary = np.flip(np.array(B[p]),1).astype(int)
			boundaries[p] = boundary

			ellipse = measure.EllipseModel()
			try:
				ellipse.estimate(boundary)
			except:
				boundary = np.unique(boundary, axis=0)
				ellipse.estimate(boundary)
			
			a[p] = ellipse.params

			r1[p] = a[p][2]
			r2[p] = a[p][3]
			cx[p] = a[p][0]
			cy[p] = a[p][1]
			theta[p] = a[p][4]

		self.numproto = numproto - discarded
		self.boundaries = boundaries
		self.ellipse = a
		self.radius1 = r1
		self.radius2 = r2
		self.centerx = cx
		self.centery = cy
		self.theta = theta

	def define_patches(self, value, expVal):

		num_patches = self.numproto
		patches = []
		protoMap = np.zeros(self.feat_map.shape)
		diag_size = np.sqrt(self.feat_map.shape[0]**2 + self.feat_map.shape[1]**2)

		for p in range(num_patches):

			patch = Patch(self.name, p, value, expVal, self.ellipse[p], [self.centerx[p], self.centery[p]], 
						 [self.radius1[p], self.radius2[p]], self.theta[p], np.pi * (self.radius1[p]/2) * (self.radius2[p]/2), 
						 self.boundaries[p], diag_size)

			patches.append(patch)

			axes = [self.radius1[p]/2, self.radius2[p]/2]
			m = [self.centerx[p], self.centery[p]]
			
			res = mkGaussian(m, axes, self.theta[p], self.feat_map.shape[1], self.feat_map.shape[0])
			protoMap = protoMap + res

		self.patches = patches
		if np.sum(protoMap) != 0:
			self.protoMap = ProtoMap(protoMap=protoMap/np.sum(protoMap), name=self.name)
		else:
			self.protoMap = ProtoMap(protoMap=protoMap, name=self.name)
