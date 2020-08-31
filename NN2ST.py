#-----------------------------------------------------------------------------#
#
#  Two-Sample Test with  Nearest-Neighbors Density Ratio Estimation
#
#  MIT License
#  Copyright (c) 2018: Andrea De Simone, Thomas Jacques
#
#-----------------------------------------------------------------------------#

from __future__ import print_function
from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from random import randint
from scipy import stats
from sklearn.utils import shuffle
from sklearn.neighbors import NearestNeighbors

#-----------------------------------------------------------------------------#
# Nearest Neighbors class
class NearestNeighborsClass(object):
    """
    Nearest neighbor density ratio estimator.
    Used to compute the Test Statistic.
    """
    
    def __init__(self, n_neighbors=2):
        """
        Initialize the class instance.
        n_neighbors: number of nearest neighbors (must be >= 1)
        """
        self.n_neighbors = n_neighbors
    
    
    def fit(self, x_benchmark, x_trial):
        """
        Build kd-trees for both samples.
        x_benchmark: benchmark (control) sample
        x_trial: trial (test) sample
        """
        
        # build kd-trees for both samples
        self.nbrs_benchmark = NearestNeighbors(n_neighbors=self.n_neighbors,
                                               algorithm='kd_tree').fit(x_benchmark)
        self.nbrs_trial     = NearestNeighbors(n_neighbors=self.n_neighbors+1,
                                               algorithm='kd_tree').fit(x_trial)


    def compute_distances(self, x_q):
        """
        Compute distances of Kth-NN in B and T from query points x_q.
        x_q: sample of query data points at which we evaluate the density ratio.
        Return: array of distances of Kth-NN in Benchmark and Trial,
                and array of coordinates (X_q) for each data point in the query sample.
        """
        # get distances of nearest neighbors in Benchmark from Query points
        distances_B, _ = self.nbrs_benchmark.kneighbors(x_q)
        radii_B = distances_B[:,-1] # distances of Kth-NN
                    
        # get distances of nearest neighbors in Trial from Query points
        distances_T, _ = self.nbrs_trial.kneighbors(x_q)
        radii_T = distances_T[:,-1] # distances of Kth-NN
                            
        coords = x_q
                                
        return(radii_B, radii_T, coords)
#*****************************************************************************#


#-----------------------------------------------------------------------------#
# Hypothesis Test class
class HypothesisTest(object):
    """
    Class to perform hypothesis testing (two-sample test) with
    Nearest Neighbors Density Ratio Estimation of pdfs.
    """
    
    def __init__(self, x_benchmark, x_trial, K=5, n_perm=100):
        """
        Initialize the class instance.
        """
        self.x_benchmark = x_benchmark
        self.x_trial = x_trial
        self.K = K
        self.n_perm = n_perm
        
        # Number of points in the samples
        self.NB = self.x_benchmark.shape[0]
        self.NT = self.x_trial.shape[0]
        
        # Dimensionality of points
        self.D = self.x_benchmark.shape[1]
    
    
    def TestStatistic(self, x_b, x_t):
        """
        Compute the test statistic using nearest neighbor
        density ratio estimator.
        """
        # Instantiate NearestNeighbors class
        NN = NearestNeighborsClass(n_neighbors = self.K)
        
        # Build kd-trees with fixed K
        NN.fit(x_b, x_t)
        
        # Compute distances r_{j,B}, r_{j,T} of Kth-NN in B and T,
        # from x_j in Trial
        self.r_B, self.r_T, _ = NN.compute_distances(x_t)
        
        # Compute estimated density ratio on Trial points
        r_hat = np.power(np.divide(self.r_B, self.r_T), self.D) * (self.NB/float(self.NT-1)) + np.finfo(float).eps
        
        # Compute test statistic over Trial points
        TS =  np.mean( np.log(r_hat) )
        
        return(TS)
    
    
    def PermutationTest(self):
        """
        Permutation Test to reconstruct the distribution of the test statistic.
        Called in 'compute_pvalue' method.
        It stores in 'self.TS_tilde' the set of TS computed at each permutation.
        """
        # U = union of B and T
        union_sample = np.concatenate((self.x_benchmark, self.x_trial), axis=0)
        n_samples = self.NB + self.NT
        
        # Initialize array of test statistic values
        self.TS_tilde = np.zeros(self.n_perm, dtype=np.float)
        
        count=0
        print("Running {:d} Permutations... 0%".format(self.n_perm))
        
        # loop over different samplings
        for i in range(self.n_perm):
            
            # Print progress
            progress = int(round(((i+1)/self.n_perm)*100,0))
            progress_list = [25, 50, 75, 100]
            if count < len(progress_list) and progress == progress_list[count]:
                count+=1
                print("Running {:d} Permutations... {:d}%".format(self.n_perm, progress))
            
            # Random permutations of U (sampling without replacement)
            x_resampled = shuffle(union_sample)
            # Assign first NB elements to Benchmark
            B_resampled = x_resampled[:self.NB]
            # Assign remaning NT elements to Trial
            T_resampled = x_resampled[self.NB:]
    
            # Compute the test statistic
            self.TS_tilde[i]  = self.TestStatistic(B_resampled, T_resampled)


    def compute_pvalue(self):
        """
        Perform permutation test and compute the (two-sided) p-value
        of the observed value of the test statistic.
        If p-value from permutations is 0, compute the p-value
        from gaussian fit to distribution (pvalue_gaussian).
        """
        # Run permutation test
        self.PermutationTest()
        # TS obtained from the original B,T samples
        self.TS_obs = self.TestStatistic(self.x_benchmark, self.x_trial)
                    
        # Mean and std of the TS distribution
        self.mu = np.mean(self.TS_tilde)
        self.sigma = np.std(self.TS_tilde)
                
        # Standardized test statistic (zero mean, unit variance)
        self.TS_prime = (self.TS_tilde - self.mu)/self.sigma
        self.TS_prime_obs = (self.TS_obs - self.mu)/self.sigma
                        
        # Two-sided p-value from TS' distribution
        self.p_value = 2*(1 - 0.01 * stats.percentileofscore(self.TS_prime,
                                                            abs(self.TS_prime_obs)) )
                            
        # if 0, compute it from standard normal
        if self.p_value == 0.0:
            self.p_value = self.pvalue_gaussian()
                
        print("")
        print("p-value  = {:e}".format(self.p_value))


    def pvalue_gaussian(self):
        """
        Two-sided p-value from stanard normal distribution (zero-mean unit-variance)
        of the standardized TS disitribution from permutations.
        The 'compute_pvalue' method needs to be evaluated first.
        """
            
        pv = 2 * stats.norm.sf(abs(self.TS_prime_obs), loc=0, scale=1)
        return(pv)


    def pvalue_test(self, alpha=0.01):
        """
        Compare p-value with alpha.
        The 'compute_pvalue' method needs to be evaluated first.
        """
        CL = int((1-alpha)*100)  # confidence level
                
        if self.p_value < alpha:
            print("Null hypothesis rejected at {:d}%CL => distributions are different".format(CL))
        else:
            print("Null hypothesis NOT rejected => distributions are the same")
                        

    def significance(self):
        """
        Convert two-sided p-value to equivalent significance Z 
        of a standard normal distribution: Z=InverseCDF(1-p/2),
        or equivalently Z=-InvserCDF(p/2).
        The 'compute_pvalue' method needs to be evaluated first.
        """
            
        Z = -1*stats.norm.ppf(0.5*self.p_value)
        print("Z = {:e} sigma".format(Z))
#*****************************************************************************#

#-----------------------------------------------------------------------------#
# Gaussian Data class
class GaussianData(object):
    """
    Class to generate 2-dimensional Benchmark and Trial
    samples from gaussian distributions
    """
    
    def __init__(self, mean_b, cov_b, mean_t, cov_t, n_points=1000):
        """
        Initialize the class instance.
        Input: mean vectors and covariance matrices for B and T samples.
        n_points: number of sample points to generate.
        """
        self.mean_b = mean_b
        self.cov_b  = cov_b
        self.mean_t = mean_t
        self.cov_t  = cov_t
        self.n_points = n_points
    
    
    def generate_data(self):
        """
        Generate toy data from two-dim Gaussian distributions
        """
        np.random.seed(0)
        
        L_b = np.linalg.cholesky(self.cov_b)
        L_t = np.linalg.cholesky(self.cov_t)
        
        self.x_benchmark = np.dot(L_b,(np.random.randn(self.n_points,2) + self.mean_b).T).T
        self.x_trial = np.dot(L_t,(np.random.randn(self.n_points,2) + self.mean_t).T).T
    
    
    def scatter_plot(self):
        """
        Generate plots of benchmark/trial samples in 2D feature space
        """
        sns.set_style('whitegrid')
        
        fig, ax = plt.subplots()
        cmap = sns.cubehelix_palette(8, start=.5, rot=-.75, as_cmap=True)
        
        
        plt.title('Benchmark and Trial Samples', fontsize=16)
        
        ax.xaxis.set_tick_params(labelsize=16, direction='inout', length=6, width=1, color='gray')
        ax.yaxis.set_tick_params(labelsize=16, direction='inout', length=6, width=1, color='gray')
        
        ax.scatter(self.x_benchmark[:,0], self.x_benchmark[:,1], c='magenta',
                   alpha=0.5, marker='x',label='B sample')
        ax.scatter(self.x_trial[:,0],self.x_trial[:,1], c='blue',
                   alpha=0.2, marker='s',label='T sample')
                   
        plt.grid(True)
        plt.legend(loc='upper left', fontsize=14)
        plt.show()
#*****************************************************************************#

#-----------------------------------------------------------------------------#
def main():

    # Generate gaussian random samples
    # Benchmark sample
    mean_b = np.array([1.0,1.0])
    cov_b  = np.array([[1,0.],[0.,1]])
    # Trial sample
    mean_t = np.array([1.15,1.15])
    cov_t  = np.array([[1.0,0.0],[0.0,1.0]])

    gdata = GaussianData(mean_b, cov_b, mean_t, cov_t, n_points=2000)
    gdata.generate_data()
    #gdata.scatter_plot()

    X_benchmark = gdata.x_benchmark
    X_trial    = gdata.x_trial

    print("Benchmark sample: {:d} points".format(X_benchmark.shape[0]))
    print("Trial sample: {:d} points\n".format(X_trial.shape[0]))
    
    
    
    
    # Compute Test Statistic and p-value
    K=2
    Nperm=100

    HT = HypothesisTest(X_benchmark, X_trial, K=K, n_perm=Nperm)

    print("TS_obs (K={:d})  = {:.5f}\n".format(K, HT.TestStatistic(X_benchmark, X_trial)) )


    HT.compute_pvalue()

    HT.significance()
    
    HT.pvalue_test(alpha=0.05)
    
#*****************************************************************************#



#-----------------------------------------------------------------------------#
if __name__ == '__main__':
    main()
    print("")
#*****************************************************************************#