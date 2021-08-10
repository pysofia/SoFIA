import numpy as np
from numpy import random

class Sobol:

    def sampling_sequence (self, N, d, dist, seed):

        """

        This function computes the different matrices needed
        for the computation of sample-based Sobol indices.

        INPUTS: Number of evaluations on which to base the indices, dimensions
        list of strings for the inputs distributions are considered by numpy.random

        OUTPUTS: Canonical evaluation matrix to be fed to the black box model

        """
        ## Sampling matrices and seed ##

        if seed != None:
            random.seed(seed)

        A = np.zeros((N,d))
        B = np.zeros((N,d))

        ## Assigning samples from canonical distributions to random matrices A/B ##

        for i in range(d):
            if dist[i]=='uniform':
                for j in range(N):
                    A[j][i] += np.random.random()
                    B[j][i] += np.random.random()
            elif dist[i]=='normal':
                for j in range(N):
                    A[j][i] += np.random.randn()
                    B[j][i] += np.random.randn()

        ## Generating the AB matrices ##

        A_B = np.zeros((d*N,d))
        for i in range(d):
            for j in range(d):
                for k in range(N):
                    A_B[k+(N*i)][j] = A[k][j]
                    A_B[k+(N*j)][j] = B[k][j]

        ## Generating the sampling matrix ## [A/B/A_B]

        samples = np.zeros((N*(d+2),d))
        matrices = [A, B]
        for i in range(2):
            for j in range(N):
                samples[j+(N*i)] = matrices[i][j]

        for i in range(d):
            for j in range(N):
                samples[j+N*(i+2)] = A_B[j+(N*i)]

        return samples

    def indices (self, eval, N, d):

        """

        This function computes the first order and total Sobol indices.

        INPUTS: The complete matrix of evaluations corresponding to A, B, ABi, 
        the number of evaluations N, input dimension d

        OUTPUTS: [First order, total indices]

        """
        ## Expectation E(f) and variance V(f) ##
        
        E = 0.
        for i in range(N):
            E += (1/N)*eval[i][0]

        V = 0.
        for i in range(N):
            V += (1/(N-1))*np.power(eval[i][0],2)
        V = V - np.power(E,2)

        ## Estimators for first and total orders ##

        Vx = [0.]*d
        for i in range(d):
            for j in range(N):
                Vx[i] += (1/N)*eval[j+N][0]*(eval[j+(2+i)*N][0]-eval[j][0])

        Vtx = [0.]*d
        for i in range(d):
            for j in range(N):
                Vtx[i] += (1/(2*N))*np.power((eval[j][0]-eval[j+(2+i)*N][0]),2)

        sobol_first = [Vx[i]/V for i in range(d)]
        sobol_total = [Vtx[i]/V for i in range(d)]

        return [sobol_first, sobol_total]