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
 
	def __init__(self, video_fr, phi, kappa):
        		
		self.allFOA = []
		self.sampled_gaze = []
		self.fly = False
		self.eps = np.finfo(np.float32).eps
		self.video_fr = video_fr
		self.phi = phi
		self.kappa = kappa

# ------------------------------------------------ GAZE SAMPLING WITH FORAGING STUFF -------------------------------------------------------------------------------


	def sample(self, iframe, patches, FOAsize, seed=None, verbose=False, debug=False):

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
			self.curr_fix_dur = 0

		else:
			if self.z_prev >= numPatch:
				self.z_prev = np.random.choice(numPatch, 1, p=np.ones(numPatch)/numPatch)[0]
			
			curr_value = self.patch[self.z_prev].value
			#other_values = [patch.value for i,patch in enumerate(self.patch) if i!=self.z_prev]
			others_rho = [patch.compute_rho(np.array(self.patch[self.z_prev].center), False, self.kappa) for i,patch in enumerate(self.patch) if i!=self.z_prev]
			#other_values = [patch.expect for i,patch in enumerate(self.patch)]
			#Q = np.mean(other_values)
			Q = np.mean(others_rho)		
			curr_rho = self.patch[self.z_prev].compute_rho(np.array(self.patch[self.z_prev].center), False, self.kappa)														#mean value of all the other patches

			A = self.phi / (curr_value+self.eps)
			gp = curr_rho * np.exp(-A*self.curr_fix_dur)
			dist = gp - Q																#distance of the current gut-f value from the average patch value
			jump_prob = 1 - logistic(20, dist)
			
			rand_num = np.random.rand()													#for the random choice (evaluate a new jump or not)
			
			if debug:
				print('\nCurrent Values for patches:')
				print([patch.value for patch in self.patch])
				print('\nInstantaneous Reward Rate: ' + str(gp))
				print('Current value for the present patch: ' + str(curr_value))
				print('Current mean value for other patches: ' + str(Q))
				print('Distance between threshold and GUT: ' + str(dist))
				print('Jump Probability: ' + str(jump_prob))
				print('Random Number: ' + str(rand_num) + '\n')
				
				import code
				code.interact(local=locals())

			if rand_num < jump_prob:
				#possible saccade
				rho = np.array([patch.compute_rho(np.array(self.patch[self.z_prev].center), True, self.kappa) for patch in self.patch])
				pi_prob = rho / np.sum(rho)		
				z = np.random.choice(numPatch, 1, p=pi_prob)[0]	#choose a new patch

				if debug:
					print('\npi_prob: ')
					print(pi_prob)
					print('Choice:')
					print(z)

					import code
					code.interact(local=locals())

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

		#timeOfFlight = euclidean(self.newFOA, prevFOA)
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
			self.fix_dur.append(self.curr_fix_dur)
			self.fix_pos.append(prevFOA)
			self.curr_fix_dur = 0
		else:
			self.FLY = False
			self.SAME_PATCH = True
			self.curr_fix_dur += 1/self.video_fr

		#FEED ---------------------------------------------------------------------------------------------------------------------------------------------------
		if self.SAME_PATCH == True and iframe > 0:
			#if we are approximately within the same patch retrieve previous exploitation/feed state and continue the local walk... 
			if verbose:
				print('\n\tSame Patch, continuing exporation...')
			startxy = self.xylast
			#t_patch = self.t_patchlast
			alphap = self.alphaplast                          
			mup = self.newFOA
			#dtp = self.dtplast
			sigmap = self.sigma_last

		elif self.SAME_PATCH == False or iframe == 0:
			#...init new exploitation/feed state for starting a new random walk   
			if verbose:        
				print('\n\tExploring new Patch...')
			#t_patch = np.round(np.max([self.patch[z].axes[0], self.patch[z].axes[1]]))*2 #setting the length of the feeding proportional to major axis
			#self.t_patchlast = t_patch
			
			#dtp = 1/t_patch             #time step
			#self.dtplast = dtp
				
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
		#self.muplast = mup
		self.sampled_gaze.append(walk)
