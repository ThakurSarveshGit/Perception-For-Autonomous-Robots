##################################################################
########################### Part 2 ###############################
##################################################################

import math
import numpy as np
import matplotlib.pyplot as plt

class EM_Algorithm:
 
    def em_calc(self, varience):
    	# Implements the EM algorithm to find the converged gaussians parameters.

        prob = [1. / self.k] * self.k #initialize the probability
        n, p = file.shape
        #weight
        w = np.zeros((n, k))
        log_likelihoods = []
        mean = file[np.random.choice(n, self.k, False), :]
        #mean = np.array([[[6.67052548]], [[6.23213941]], [[6.25556869]]]) #uncomment to check the result in the report
        sigmas = np.array([np.eye(p)] * self.k)
       
        if varience == 0:
            #sigmas = np.array([[[ 7.67788792]] ,[[ 7.67788792]], [[ 7.67788792]]])#uncomment to check the result in the report
            sigmas *= np.random.uniform(1, 10)
        
        print("Initial Mean:",mean.reshape(1,-1))
        print("Initial Variance:",sigmas.reshape(1,-1))
        
        for i in range(iteration):
            #E step
            for q in range(k):
                w[:, q] = prob[q] * calculate_gaussian(mean[q], sigmas[q])
            
            log_likelihood = np.sum(np.log(np.sum(w, axis=1)))
            log_likelihoods.append(log_likelihood)
            w = (w.T / np.sum(w, axis=1)).T
            sum_w = np.sum(w, axis=0)

            # M Step
            prob = np.zeros(k)
            for j in range(k):
                 prob[j] = 1. / n * sum_w[j]

            
            for j in range(k):#mean update
                for q in range(n):
                   mean=update_mean(mean,j,q,w)
                mean[j] /= sum_w[j]
                
            
            if varience == 0:#variance update
                for j in range(k):
                    for q in range(n):
                        update_var(sigmas,w,j,q,np.reshape(file[q] - mean[j], (p, 1)))
                    sigmas[j] /= sum_w[j]

            if(check_convergence(log_likelihoods,log_likelihood)):
                break

        return  prob, mean, sigmas, log_likelihoods

def calculate_gaussian(meu,s):
    A=file-meu
    B=np.dot(np.linalg.inv(s), (file - meu).T)
    C=(A * B.T).sum(axis=1)
    a=np.linalg.det(s) ** -.5 ** (2 * np.pi) ** (-file.shape[1] / 2.)# ** to get the value in integer 
    d=a * np.exp(-0.5 * C)
    return d

def update_mean(mean,j,q,ws):
    mean[j] += ws[q, j] * file[q]
    return mean

def update_var(sigmas,ws,j,q,s):
    sigmas[j] += ws[q, j] * np.dot(s, s.T)
    return sigmas
    
def check_convergence(l,l_1):
    if(len(l)>2 and np.abs(l_1- l[-2]) < e):
        return 1
    else:
        return 0
        
    
if __name__ == "__main__":

    datafile = "em_data.txt"
    file = np.loadtxt(datafile, delimiter=',')
   
    iteration=500
    e = 0.01
    k = 3#number of clusters
    file = file.reshape(-1, 1)
    
    varience = 0
    gausian =  EM_Algorithm()
    gausian.k=k
    gausian.e=e;
    prob, mean, sigmas, log_likelihoods = gausian.em_calc(varience)
    print("Final Mean:",mean.reshape(1,-1))
    print("Final Variance 0:",sigmas.reshape(1,-1))
    print("Number of iterations:",len(log_likelihoods))
    print("Log likelihood list : ",log_likelihoods)

    plt.plot(np.linspace(1,len(log_likelihoods),len(log_likelihoods)),log_likelihoods)
    plt.show()
