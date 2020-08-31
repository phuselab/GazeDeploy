import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

class Patch(object):
	
	def __init__(self, label, patch_num, value, expVal, a, center, axes, angle, area, boundaries, diag_size, tot_area):
        
		self.label = label
		self.patch_num = patch_num
		self.value = value
		self.a = a
		self.center = center
		self.axes = axes
		self.angle = angle
		self.area = area
		self.boundaries = boundaries
		self.diag_size = diag_size
		self.expVal = expVal
		self.tot_area = tot_area

	'''def compute_expected_reward(self):
		self.expect = self.value'''

	def compute_rho(self, last_patch_mu, choice, k):

		#k = 18
		distance = np.linalg.norm(last_patch_mu - self.center) / self.diag_size
		area = 1
		if choice:
			value = self.expVal
			#area = 1
		else:
			value = self.value
			#area = self.area/(4*self.diag_size)
		
		rho = value * area * np.exp(-k*distance)


		'''print('\n\tCOMPUTE RHO in ' + self.label)
		print('Current Patch center: ' + str(last_patch_mu))
		print(self.label + ' center: ' + str(self.center))
		print('Distance1: ' + str(distance))
		print('Value: ' + str(self.value))
		print('ExpValue: ' + str(self.expVal))
		print('Area: ' + str(area))
		print('exp(-distance2): ' + str(np.exp(-k*distance)))
		print('RHO: ' + str(rho))'''

		'''import code
		code.interact(local=locals())'''

		return rho