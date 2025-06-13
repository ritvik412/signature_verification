import numpy as np

def delta(arr):
	return arr[1:]-arr[:-1]

# Differentiate a 2d array with respect to time.
def diff(arr,time):
	da = delta(arr)
	dt = delta(time)
	return da/dt

# Take the l2 norm of a 2d array.
def l2(arr):
	return np.sqrt(arr[:,0]**2+arr[:,1]**2)