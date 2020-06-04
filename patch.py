import numpy as np
import matplotlib.pyplot as plt

class Patch(object):
	
	def __init__(self, label, patch_num, value, expVal, a, center, axes, angle, area, boundaries, diag_size):
        
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

	def compute_rho(self, last_patch_mu, choice):

		kappa = 15
		distance = np.linalg.norm(last_patch_mu - self.center) / self.diag_size
		area = 1
		if choice:
			value = self.expVal
		else:
			value = self.value
		
		rho = value * area * np.exp(-kappa*distance)

		return rho