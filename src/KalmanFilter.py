# Kalman filter library containing the one-step filter and the looping function
# Cormac Tredoux 2025
# External libraries of numpy, scipy and os called

import numpy as np
from scipy.stats import norm
import os

# Kalman() defines one filter step at time t

def Kalman(x,y,state,P,Q,R,d):
	
 
	###Setup###
	#Building observation row: 1xd
	H = np.array([x])
	#Making scalar y into "vector" obeject: 1x1
	z = np.array([[y]])
 
 
	###Prediction Step###
	
	#Updating state cov in prediction step
	P+=Q
 
	#S = cov of pred. distb. of yt|t-1 
    #  -> S = Ht @ Pt|t-1 @ t(Ht) + R
	S = np.matmul(H,np.matmul(P,np.transpose(H)))+R
 
	###Update Step###
 
	#Innovation as z - t(H)%*%state -> note state is the estimated state here
	innov = err = z-np.matmul(H,state)
 
	#Kalman Gain 
	#  -> Kt = Pt|t-1 @ t(Ht)
	#  -> Kt = Kt @ S^-1
	K = np.matmul(P,np.transpose(H))
	K = np.matmul(K,np.linalg.inv(S))
 
	#Update state with Kalman gain and *unfiltered* error
	state = state+np.matmul(K,err)
	#Update -> Pt|t = (I - Kt @ Ht) @ Pt|t-1
	P = np.matmul((np.eye(d)-np.matmul(K,H)),P)
 
	#Filtered error
	err = z-np.matmul(H,state)
	return (state,P,err,S, innov)

# Kalman_run() loops the one step filter in Kalman()

def Kalman_run(xs,ys,state_0,P_0,Q,R,d):
    
    #Makes a python list whos first element is state_0 & 0 respectively.
	states = [state_0]
	errs = [0]
	state = state_0
	P = P_0
	Ps = [P]
	Ss = []
	innovs = []
 
	#Starting Likelihood off at 0
	loglik = 0.0
 
	for k in range(0,len(xs)):
     
		#kth row of 2-D NumPy array xs
		x = xs[k]
		#kth element in y
		y = ys[k]
  
		(state,P,err,S, innov) = Kalman(x,y,state,P,Q,R,d)
		Ps += [P]
  
		#Updating log-likelihood
		loglik += -0.5*(np.log(float(S)) +(float(innov)**2)/S +np.log(2*np.pi))
		states.append(state)
	
		errs+=[err.flatten()[0]]
		innovs.append(float(innov))
		Ss.append(float(S))

    
	return (loglik,states,errs, Ps,Ss, innovs)

def set_params(params,d):
	state_0 = np.reshape(params[0:d],(d,1))
	k = d
	P_0 = np.zeros((d,d))
	for ii in range(0,d):
		for jj in range(0,d):
			if ii>=jj:
				P_0[ii,jj] = params[k]
				k+=1
	P_0 = np.matmul(P_0,np.transpose(P_0))
	Q = np.zeros((d,d))
	for ii in range(0,d):
		for jj in range(0,d):
			if ii>=jj:
				Q[ii,jj] = params[k]
				k+=1
	Q = np.matmul(Q,np.transpose(Q))
	R = np.array([[params[k]**2]])

	return (state_0,P_0,Q,R)


def generate_initial_states(NumProxies, scale=1.0, overwrite=True):
   

    state0 = np.zeros((NumProxies, 1))
    P0 = np.identity(NumProxies) * scale
    Q = np.identity(NumProxies) * scale
    R = np.array([[0.01]])

    if overwrite:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        save_dir = os.path.join(parent_dir, "Saved_States")
        os.makedirs(save_dir, exist_ok=True)

        np.savetxt(os.path.join(save_dir, "state0.txt"), state0)
        np.savetxt(os.path.join(save_dir, "P0.txt"), P0)
        np.savetxt(os.path.join(save_dir, "Q.txt"), Q)
        np.savetxt(os.path.join(save_dir, "R.txt"), R)

    return state0, P0, Q, R
