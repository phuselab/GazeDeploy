import numpy as np
import matplotlib.pyplot as plt

def logistic(beta, dist):
	return 1/(1+np.exp(-beta*dist))

def sample_OU_trajectory(numOfpoints, alpha, mu, startxy, dt, sigma):

	Sigma1 = np.array([[1, np.exp(-alpha * dt / 2)],[np.exp(-alpha * dt / 2), 1]])
	Sigma2 = Sigma1

	x = np.zeros(numOfpoints)
	y = np.zeros(numOfpoints)
	mu1 = np.zeros(numOfpoints)
	mu2 = np.zeros(numOfpoints)
	x[0] = startxy[1]
	y[0] = startxy[0]
	mu1[0] = mu[1]
	mu2[0] = mu[0]
	for i in range(numOfpoints-1):
		r1 = np.random.randn() * sigma
		r2 = np.random.randn() * sigma

		x[i+1] = mu1[i]+(x[i]-mu1[i])*(Sigma1[0,1]/Sigma1[0,0])+np.sqrt(Sigma1[1,1])-(((Sigma1[0,1]**2)/Sigma1[0,0]))*r1
		y[i+1] = mu2[i]+(y[i]-mu2[i])*(Sigma2[0,1]/Sigma2[0,0])+np.sqrt(Sigma2[1,1])-(((Sigma2[0,1]**2)/Sigma2[0,0]))*r2
		mu1[i+1] = mu1[i]
		mu2[i+1] = mu2[i]

	return np.stack([y,x], axis=1)



class GazeSampler(object):
 
	def __init__(self, video_fr):
        		
		self.allFOA = []
		self.sampled_gaze = []
		self.fly = False
		self.eps = np.finfo(np.float32).eps
		self.beta = 35		#beta for he logistic function
		self.alpha = 0.65  #alpha < 1 for longer fixation times
		self.video_fr = video_fr

# ------------------------------------------------ GAZE SAMPLING WITH FORAGING STUFF -------------------------------------------------------------------------------


	def sample(self, iframe, patches, FOAsize, seed=None, verbose=False):

		if seed is not None:
			np.random.seed(seed)

		self.patch = patches
		self.FOAsize = FOAsize

		numPatch = len(self.patch)
		if iframe == 0:				#first frame
			
			self.s_ampl = []
			self.s_dir = []
			self.fix_dur = []
			self.fix_pos = []

			z = 0			#start from the CB
			self.z_prev = 0
			m = self.patch[0].center
			s = np.asarray(self.patch[0].axes)
			S = np.eye(2) * s
			arrivingFOA = np.random.multivariate_normal(m, S)
			
			self.allFOA.append(arrivingFOA)
			self.newFOA = arrivingFOA
			prevFOA = arrivingFOA
			self.t = 0

		else:
			if self.z_prev >= numPatch:
				self.z_prev = np.random.choice(numPatch, 1, p=np.ones(numPatch)/numPatch)[0]
			
			curr_value = self.patch[self.z_prev].value
			other_values = [patch.value for i,patch in enumerate(self.patch) if i!=self.z_prev]
			Q = np.mean(other_values)				#mean value of all the other patches (Eq. 28)
			curr_rho = self.patch[self.z_prev].compute_rho(np.array(self.patch[self.z_prev].center), False)				

			A = self.alpha / (curr_value+self.eps)
			gp = curr_rho * np.exp(-A*self.t)						#Eq. 26
			dist = gp - Q											#distance of the current gut-f value from the average patch value
			jump_prob = 1 - logistic(self.beta, dist)				#Eq. 29
			
			rand_num = np.random.rand()								#for the random choice, evaluate a new jump or not (Eq. 30)

			if rand_num < jump_prob:
				#possible saccade
				rho = np.array([patch.compute_rho(np.array(self.patch[self.z_prev].center), True) for patch in self.patch])
				pi_prob = rho / np.sum(rho)		
				z = np.random.choice(numPatch, 1, p=pi_prob)[0]		#choose a new patch (Eq. 31)

				rand_num2 = np.random.rand()		#for the random choice (small saccade of keep fixating)

				if z != self.z_prev or rand_num2 > 0.6:		#increase the value on the rhs to decrease the number of small saccades
					# if here, then a new patch has been choosen, OR a small saccade will take place
					self.z_prev = z
					m = self.patch[z].center
					s = np.asarray(self.patch[z].axes) * 3
					S = np.eye(2) * s
					self.newFOA = np.random.multivariate_normal(m, S)	#choose position inside the (new) patch
					prevFOA = self.allFOA[-1]
				else:
					# if here, will keep fixating the same patch
					z = self.z_prev 
					prevFOA = self.allFOA[-1]
					self.newFOA = prevFOA
			else:
				# keep fixating...
				z = self.z_prev 
				prevFOA = self.allFOA[-1]
				self.newFOA = prevFOA

		timeOfFlight = np.linalg.norm(self.newFOA - prevFOA)

		#Possible FLY --------------------------------------------------------------------------------------------------------------------------------------------
		if timeOfFlight > self.FOAsize:	
			if verbose:
				print('\nnew and prev: ', self.newFOA, prevFOA)
				#different patch from current
				print('\n\tFLY!!!! New patch: ' + self.patch[z].label)
			self.FLY = True
			self.SAME_PATCH = False
			self.allFOA.append(self.newFOA)

			self.s_ampl.append(timeOfFlight)
			curr_sdir = np.arctan2(self.newFOA[1]-prevFOA[1], self.newFOA[0]-prevFOA[0]) + np.pi
			self.s_dir.append(curr_sdir)
			self.fix_dur.append(self.t)
			self.fix_pos.append(prevFOA)
			self.t = 0
		else:
			self.FLY = False
			self.SAME_PATCH = True
			self.t += 1/self.video_fr

		#FEED ---------------------------------------------------------------------------------------------------------------------------------------------------
		if self.SAME_PATCH == True and iframe > 0:
			#if we are approximately within the same patch retrieve previous exploitation/feed state and continue the local walk... 
			if verbose:
				print('\n\tSame Patch, continuing exporation...')
			startxy = self.xylast
			alphap = self.alphaplast                          
			mup = self.newFOA
			sigmap = self.sigma_last

		elif self.SAME_PATCH == False or iframe == 0:
			#...init new exploitation/feed state for starting a new random walk   
			if verbose:        
				print('\n\tExploring new Patch...')
				
			if iframe == 0:
				startxy = self.newFOA
			else: 
				startxy = self.sampled_gaze[-1][-1,:]

			mup = self.newFOA

			if not self.FLY:
				alphap = np.max([self.patch[z].axes[0], self.patch[z].axes[1]])*10
				sigmap = 1
			else:
				alphap = timeOfFlight/5.
				sigmap = 15

			self.alphaplast = alphap
			self.sigma_last = sigmap
                      
		walk = sample_OU_trajectory(numOfpoints=10, alpha=alphap, mu=mup, startxy=startxy, dt=1/alphap, sigma=sigmap)
		
		arrivingFOA = np.round(walk[-1,:])
		self.xylast = arrivingFOA
		self.sampled_gaze.append(walk)