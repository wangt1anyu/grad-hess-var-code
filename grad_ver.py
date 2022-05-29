import numpy as np 
from scipy import optimize
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize
import os

from tqdm import * 
import pickle


def func(x): 
    
    return np.sum( np.sin(x) ) + np.exp((x[0]-1) * (x[1]+2))  


def sample_stiefel(n,k): 
    
    A = np.random.normal( 0, 1, (n, k) ) 
    
    def normalize(v): 
        return v / np.sqrt(v.dot(v)) 

    n = A.shape[1] 

    A[:, 0] = normalize(A[:, 0])  

    for i in range(1, n): 
        Ai = A[:, i] 
        for j in range(0, i): 
            Aj = A[:, j] 
            t = Ai.dot(Aj) 
            Ai = Ai - t * Aj 
        A[:, i] = normalize(Ai) 
        
    return A 



###### 
def stiefel( x, delta, k, n = 500 ): 

    V = sample_stiefel(n,k) 

    gs = [ n/delta/2 * ( func( x + delta * V[:,i]) - func( x - delta * V[:,i]) ) * V[:,i] for i in range(k) ] 

    g = np.mean( np.array( gs ) , axis = 0 ) 
    
    return g





# delta = 0.01; 
n = 500; rep = 10

xs = [np.array([0]*500), np.array([np.pi/4.]*500) ] 

deltas = [ 0.1, 0.01, 0.001 ] 

for i in range(len(xs)):
    
    x = xs[i]
    
    truth = (np.array( np.cos(x) ) + np.array( [(x[1] + 2)* np.exp( (x[0]-1)*(x[1]+2) ) 
                                                 , (x[0] - 1)* np.exp( (x[0]-1)*(x[1]+2) ) ] + [0]*(n-2) ) ) 
    
    for delta in deltas: 

        res = [] 

        print(delta)
        
        for _ in range(rep): 
            
            res_inner = []

            for k in range(1,n+1): 

                V = sample_stiefel(n,k) 

                gs = [ n/delta/2 * ( func( x + delta * V[:,i]) - func( x - delta * V[:,i]) ) * V[:,i] for i in range(k) ] 

                g = np.mean( np.array( gs ) , axis = 0 ) 
                
                error = np.linalg.norm( g - truth )

                res_inner.append(error) 
                
            res.append(res_inner) 
            
        pickle.dump( res, 
                            open('./raw_results/for_trend_x{0}_delta{1}'.format(i,delta), 'wb' ) ) 