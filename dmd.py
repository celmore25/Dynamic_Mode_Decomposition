import numpy as np
from scipy import linalg as la
from cmath import exp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class visualize:
	'''
	This class holds all of the functions needed for visualizing of DMD
	results and the input data into DMD.
	'''
	def surface_data(F,x,t,bounds_on=False,bounds=[[0,1],[0,1],[0,1]]):
		'''
		This function will create a surface plot of given a set of data
		for f(x),x,t. f(x) must be given in matrix format with evenly 
		spaced x and t corresponding the A matrix.

		inputs:
		f - spacial-temporal data
		x - spacial vector
		t - time vector

		outputs:
		surf - object of the 3D plot

		options:
		bounds_on - boolean to indicate bounds wanted
		bounds - Optional array that contains the bounds desired to put on
				 the axes. Sample input: [[0,1],[0,1],[0,1]] for f(x),x,t.
		'''
		
		# first make a meshgrid with the t and x vector. 
		# we first define the x values as the rows and t as the columns
		# in order to be consistent with general DMD structure. 
		x_len = np.size(x)
		t_len = np.size(t)
		X, T = np.meshgrid(x, t)

		# Create 3D figure
		fig = plt.figure()
		ax = fig.gca(projection='3d')

		# Plot f(x)
		surf = ax.plot_surface(X, T, F, linewidth=0,cmap=cm.coolwarm,antialiased=True)
		# surf  = ax.plot_trisurf(x,t,F)
		return surf

	def surface_function(f,x,t,bounds_on=False,bounds=[[0,1],[0,1],[0,1]]):
		'''
		This function will create a surface plot of given a set of data
		for f(x),x,t. 

		inputs:
		f - input function
		x - spacial vector
		t - time vector

		outputs:
		surf - object for the figure that is created

		options:
		bounds_on - boolean to indicate bounds wanted
		bounds - Optional array that contains the bounds desired to put on
				 the axes. Sample input: [[0,1],[0,1],[0,1]] for f(x),x,t.
		'''

		# first make a meshgrid with the t and x vector. 
		# we first define the x values as the rows and t as the columns
		# in order to be consistent with general DMD structure. 
		x_len = np.size(x)
		t_len = np.size(t)
		X, T = np.meshgrid(x, t)
		dt = t[0]-t[1]

		# now evaluate the function.
		F = np.zeros((t_len,x_len))
		for i,x_val in enumerate(x):
			for j,t_val in enumerate(t):
				F[j,i] = f(x_val,t_val)

		# Create 3D figure
		fig = plt.figure()
		ax = fig.gca(projection='3d')

		# Plot f(x)
		surf = ax.plot_surface(X, T, F, linewidth=0,cmap=cm.coolwarm,antialiased=True)
		
		return surf

class manipulate:

	'''
	This class contains helpful functions for the specific matrix 
	manipulation needed for Dynamic Mode Decomposition.
	'''

	def two_2_three(A,n,verbose = False):
		'''
		This function will take a matrix that has three dimensional data
		expressed as a 2D matrix (columns as full vectors) and return a 3D
		matrix output where the columns are transformed back into 2D matrices
		and then placed in a numpy list of matricies. 

		This is useful for DMD
		computation because the general input to DMD is a 2D matrix, but the
		output may need to be converted back to 3D in order to visualize 
		results. 

		This function is also useful for the visualization of dynamic
		modes that are computed by DMD.
		
		inputs: 
		A - 2D matrix where columns represent (mxn) matricies
		n - # of columns in each smaller matrix

		outputs:
		A_3 - 3D numpy array containing each indiviual mxn matrix

		Options:
		verbose - prints matricies in order to see the transformation 
				  that takes place
		'''
		if verbose:
			print('Entering 2D to 3D conversion:\n')

		# make sure the input is a numpy array
		A = np.array(A)
		rows, cols = np.shape(A)

		# calculate the number of columns in the matrix and quit if 
		# the numbers do not match
		m = int(rows/n)
		if rows%m  != 0:
			print('Invalid column input. \nFunction Exited.')
			return None

		# initalize A_3, the matrix 
		A_3 = np.zeros((cols,m,n))

		# now loop over the columns construct the matrices and place them
		# in the A_3 matrix
		a_hold = np.zeros((m,n))
		for i in range(cols):
			# grab the column that holds the vector
			col_hold = A[:,i]

			# convert the column into the holder matrix shape
			for j in range(n):
				a_hold[:,j] = col_hold[j*m:(j+1)*m]

			# place the matrix in the new 3D holder
			A_3[i] = a_hold

		# print the transformation that occured
		if verbose:
			print('A =\n',A,'\n')
			print('A_3 =\n',A_3,'\n')

		# the program is finished
		return A_3

	def three_2_two(A,verbose = False):
		'''
		This function will take a matrix that has three dimensional data
		expressed as a 3D matrix and return a 2D matrix output where the 
		columns represent the matricies in the original 3D matrix

		This is neccessary for DMD computation because spacial-temporal 
		data has to be converted into one single matrix in order to perform
		the neccessary singluar value decomposition.
		
		inputs: 
		A - 3D matrix where columns represent (mxn) matricies

		outputs:
		A_2 - 2D numpy array containing each indiviual mxn matrix as a 
			  column

		Options:
		verbose - prints matricies in order to see the transformation 
				  that takes place
		'''

		if verbose:
			print('Entering 3D to 2D conversion:\n')

		# make sure the input is a numpy array and store shape
		A = np.array(A)
		length, rows, cols = np.shape(A)
		n = length # number of columns for A_2

		# calculate the length of each column needed (number of rows)
		m = rows*cols

		# initalize A_2, the matrix to be returned
		A_2 = np.zeros((m,n))

		# now loop over the matricies, construct the columns and place them
		# in the A_2 matrix
		vec = np.zeros(m)
		A_2 = A_2.transpose()
		for i in range(n):
			# grab the matrix
			matrix = A[i]

			# loop through the columns and store them in a_hold
			for j in range(cols):
				vec[j*rows:(j+1)*rows] = matrix[:,j]

			# place the matrix in the new 3D holder
			A_2[i] = vec
		A_2 = A_2.transpose()
		# print the transformation that occured
		if verbose:
			print('A =\n',A,'\n')
			print('A_2 =\n',A_2,'\n')

		# the program is finished
		return A_2

	def split(Xf,verbose = False):
		'''
		This function will perform a crutical manipulation for DMD
		which is the splitting of a spacial-temporal matrix (Xf) into 
		two matrices (X and Xp). The X matrix is the time series for 
		1 to n-1 and Xp is the time series of 2 to n where n is the 
		number of time intervals (columns of the original Xf).  

		input: 
		Xf - matix of full spacial-temporal data

		output:
		X - matix for times 1 to n-1
		Xp - matix for times 2 to n

		options:
		verbose - boolean for visualization of splitting
		'''

		if verbose:
			print('Entering the matrix splitting function:')

		if verbose:
			print('Xf =\n',Xf,'\n')

		X = Xf[:,:-1]
		Xp = Xf[:,1:]

		if verbose:
			print('X =\n',X,'\n')
			print('Xp =\n',Xp,'\n')
		return X,Xp

class dmd:
	'''
	This class contains the functions needed for performing a full DMD
	on any given matrix. Depending on functions being used, different 
	outputs can be achived.
	This class also contains functions useful to the analysis of DMD
	results and intermediates.
	'''
	def decomp(Xf,time,verbose=False,rank_cut=True,esp=1e-2,svd_cut=False,
				num_svd=1):
		'''
		This function performs the basic DMD on a given matrix A.
		The general outline of the algorithm is as follows...
		1)  Break up the input X matrix into time series for 1 to n-1 (X)
			and 2 to n (X) where n is the number of time intervals (X_p)
			(columns). This uses the manipulate class's function "split".
		2)  Compute the Singular Value Decomposition of X. X = (U)(S)(Vh)
		3)  Compute the A_t matrix. This is related to the A matrix which
			gives A = X * Xp. However, the At matrix has been projected 
			onto the POD modes of X. Therefore At = U'*A*U. (' denomates 
			a matrix transpose)
		4)  Compute the eigendecomposition of the At matrix. At*W=W*L
		5)  Compute the DMD modes of A by SVD reconstruction. Finally, the
			DMD modes are given by the columns of Phi.
			Phi = (Xp)(V)(S^-1)(W)
		6)  Compute the discrete and continuous time eigenvalues 
			lam (discrete) is the diagonal matrix of eigenvalues of At.
			omg (continuous) = ln(lam)/dt
		7) 	Compute the amplitude of each DMD mode (b). This is a vector 
			which applies to this system: Phi(b)=X_1 Where X_1 is the first 
			column of the input vector X. This requires a linear equation
			solver via scipy. 
		8)  Reconstruct the matrix X from the DMD modes (Xdmd). 
		inputs:
		X - (mxn) Spacial Temporal Matrix
		time - (nx1) Time vector
		outputs:
		(1) Phi - DMD modes
		(2) omg - discrete time eigenvalues
		(3) lam - continuous time eigenvalues
		(4) b - amplitudes of DMD modes
		(5) Xdmd - reconstructed X matrix from DMD modes
		(6) rank - the rank used in calculations
		*** all contained in a class ***
		*** see ### (10) ### below   ***
		options:
		verbose - boolean for more information
		svd_cut - boolean for truncation of SVD values of X
		esp - value to truncate singular values lower than
		rank_cut - truncate the SVD of X to the rank of X
		num_svd - number of singular values to use
		'''
		if verbose:
			print('Entering Dynamic Mode Decomposition:\n')

		### (1) ####
		# split the Xf matrix 
		X,Xp = manipulate.split(Xf)
		if verbose:
			print('X = \n',X,'\n')
			print('X` = \n',Xp,'\n')

		### (2) ###
		# perform a singular value decompostion on X
		U,S,Vh = la.svd(X)
		if verbose:
			print('Singular value decomposition:')
			print('U: \n',U)
			print('S: \n',S)
			print('Vh: \n',Vh)
			print('Reconstruction:')
			S_m = np.zeros(np.shape(X))
			for i in range(len(list(S))):
				S_m[i,i] = S[i]
			recon = np.dot(np.dot(U,S_m),Vh)
			print('X =\n',recon)

		# perfom desired trunctions of X
		if svd_cut:
			rank_cut = False
		if rank_cut: # this is the default truncation
			rank = 0
			for i in S:
				if i > esp:
					rank += 1
			if verbose:
				print('Singular Values of X:','\n',S,'\n')
				print('Reducing Rank of System...\n')
			Ur = U[:,0:rank]
			Sr = S[0:rank]
			Vhr = Vh[0:rank,:]
			if verbose:
				recon = np.dot(np.dot(Ur,np.diag(Sr)),Vhr)
				print('Rank Reduced reconstruction:\n','X =\n',recon)
		elif svd_cut:
			rank = num_svd
			if verbose:
				print('Singular Values of X:','\n',S,'\n')
				print('Reducing Rank of System to n =',num_svd,'...\n')
			Ur = U[:,0:rank]
			Sr = S[0:rank]
			Vhr = Vh[0:rank,:]
			if verbose:
				recon = np.dot(np.dot(Ur,np.diag(Sr)),Vhr)
				print('Rank Reduced reconstruction:\n','X =\n',recon)


		# return the condition number to view singularity
		condition = max(Sr)/min(Sr)
		smallest_svd = min(Sr)
		svd_used = np.size(Sr)
		if verbose:	
			condition = max(Sr)/min(Sr)
			print('Condition of Rank Converted Matrix X:'\
											,'\nK =',condition,'\n')

		# make the singular values a matrix and take the inverse
		Sr_inv = np.diag([i**-1 for i in Sr])
		Sr = np.diag(Sr)

		### (3) ###
		# now compute the A_t matrix 
		Vr = Vhr.conj().T
		At = Ur.conj().T.dot(Xp)
		At = At.dot(Vr)
		At = At.dot(la.inv(Sr))
		if verbose:
			print('A~ = \n',At,'\n')

		### (4) ###
		# perform the eigen decomposition of At
		L,W = la.eig(At)

		### (5) ### 
		# compute the DMD modes
		phi = np.dot(Xp,Vhr.conj().T)
		phi = np.dot(phi,Sr_inv)
		phi = np.dot(phi,W)

		if verbose:
			print('Normal Mode Matrix:','\nPhi =',phi,'\n')

		### (6) ### 
		# compute the continuous and discrete eigenvalues
		dt = time[1] - time[0]
		lam = L
		omg = np.log(lam)/dt
		if verbose:
			print('Discrete time eigenvalues:\n','Lambda =',L,'\n')
			print('Continuous time eigenvalues:\n','Omega =',np.log(L)/dt,'\n')

		### (7) ### 
		# compute the applitude vector b by solving the linear system described
		# note that a least squares solver has to be used in order to approximate
		# the solution to the overdefined problem
		x1 = X[:,0]
		b = la.lstsq(phi,x1)
		b = b[0]
		if verbose:
			print('b =\n',b,'\n')

		### (8) ### 
		# finally reconstruct the data matrix from the DMD modes
		length = np.size(time) # number of time measurements
		dynamics = np.zeros((rank,length),dtype=np.complex_) # initialize the time dynamics
		for t in range(length):
			omg_p = np.array([exp(i*time[t]) for i in omg])
			dynamics[:,t] = b*omg_p
			
		if verbose:
			print('Time dynamics:\n',dynamics,'\n')
		Xdmd = np.dot(phi,dynamics)
		if verbose:
			print('Reconstruction:\n',np.real(Xdmd),'\n')
			print('Original:\n',np.real(Xf),'\n')

		### (9) ### 
		# calculate some residual value
		res = np.real(Xf - Xdmd)
		error = la.norm(res)/la.norm(Xf)
		if verbose:
			print('Reconstruction Error:',round(error*100,2),'%')

		### (10) ###
		# returns a class with all of the results
		class results():
			def __init__(self):
				self.phi = phi
				self.omg = omg
				self.lam = lam
				self.b = b
				self.Xdmd = Xdmd
				self.error = error * 100
				self.rank = rank
				self.svd_used = svd_used
				self.condition = condition
				self.smallest_svd = smallest_svd
		final = results()

		return final

	def predict(dmd,t):
		'''
		This function will take a DMD decomposition output 
		result and a desired time incremint prediction and 
		produce a prediction of the system at the given time.

		inputs:
		dmd - class that comes from the function "decomp"
		t - future time for prediction

		outputs:
		x - prediction vector (real part only)
		'''

		# finally reconstruct the data matrix from the DMD modes
		dynamics = np.zeros((dmd.rank,1),dtype=np.complex_)
		omg_p = np.array([exp(i*t) for i in dmd.omg]) 
		dynamics = dmd.b*omg_p
		x = np.real(np.dot(dmd.phi,dynamics))

		return x

class energy:
	'''
	This class will hold all of the necessary function for manipulating 
	the energy price data in this project along with the DMD results
	'''
	def imp_prices(name):
		'''
		This is a simple function that will import price data as a 
		matrix in numpy and return the resulting matrix.

		inputs:
		name - string with name of the csv file

		outputs:
		X - numpy array with price data over time
		'''
		X = np.genfromtxt(name, delimiter=',')
		X = X[1:]
		return X

	def imp_locations(name):
		'''
		This is a simple function that will import location data as a 
		matrix in numpy and return the resulting matrix.

		inputs:
		name - string with name of the csv file

		outputs:
		numpy array with price data over time
		'''

		X = np.genfromtxt(name, delimiter=',')
		X = X[1:]
		return X

	def plot_energy(data,indecies,start,num_vals):
		'''
		This function will plot the energy prices for a given vector of 
		indicies corresponding to location in a data matrix. The number of
		time measurements and where to start can also be specified. 

		inputs:
		data - energy price data
		indecies - array of integers corresponding to locations
		start - time measurement to start at
		num_vals - number of time measurements to plot

		output:
		fig - figure containing the plot
		'''
		fig, ax = plt.subplots()
		indecies = list(indecies)
		for i in indecies:
			ax.plot(data[i,start:start+num_vals])
		title = 'Energy Price Visualization\n'
		title = title + str(len(indecies)) + ' Locations Shown'
		ax.set(xlabel='time (days)', ylabel='price ($)',\
        		title='Energy Price Visualization')
		return fig

if __name__ == '__main__':
	print('DMD testing environment entered.\n')

	print('\n\n ---- Function Test ---- \n\n')
	# testing function from book
	# im = 0+1j
	# sech = lambda x: 1/np.cosh(x)
	# f = lambda x,t: sech(x + 3)*exp(2.3*im*t) + 2*sech(x)*np.tanh(x)*exp(im*2.8*t)
	# points = 3
	# x = np.linspace(-10,10,points)
	# t = np.linspace(0,4*np.pi,points)

	# # test decomposition of the function given above
	# F = np.zeros((np.size(x),np.size(t)),dtype=np.complex_)
	# for i,x_val in enumerate(x):
	# 	for j,t_val in enumerate(t):
	# 		F[i,j] = f(x_val,t_val)
	# results = dmd.decomp(F,t,verbose = True,num_svd=2,svd_cut=True)


	print('\n\n ---- Energy Test ---- \n\n')
	data = energy.imp_prices('prices.csv')
	locs = np.shape(data)[0]
	point = 330
	start = 0
	X = data[0:locs]
	X = X.T
	X = X[start:start+point]
	X = X.T

	t = np.arange(int(np.size(data[0])))
	error = []
	num_svd = []
	condition = []
	smallest_svd = []

	esp_vec = np.linspace(1,1e-2,20)
	for i in esp_vec:
		print('esp',i)
		results = dmd.decomp(data,t,verbose = False,num_svd=330,svd_cut=False,esp = i)
		error.append(results.error)
		num_svd.append(results.svd_used)
		condition.append(results.condition)
		smallest_svd.append(results.smallest_svd)
	
	plt.plot(condition,error)
	plt.show()
	














	# decomposition of data matrix
	# starts = np.arange(20)
	# points = np.arange(4,20)
	# locs = 3380

	# error = []
	# end_error = []
	# first_error = []
	# for point in points:
	# 	error_hold = []
	# 	end_error_hold = []
	# 	first_error_hold = []

	# 	for start in starts:
	# 		X = data[0:locs]
	# 		X = X.T
	# 		X = X[start:start+point]
	# 		X = X.T

	# 		t = np.arange(point)
	# 		results = dmd.decomp(X,t,verbose = False,num_svd=3,svd_cut=False)
	# 		error_hold.append(results.error)

	# 		# predcition testing on the last training point
	# 		t_test = t[-1]
	# 		act_vec = data[0:locs].T[t_test]
	# 		pred_vec = dmd.predict(results,t_test)
	# 		end_error_inst = la.norm(act_vec - pred_vec) / la.norm(act_vec)
	# 		end_error_hold.append(end_error_inst*100)

	# 		# prediction on the first time step input
	# 		t_test += 1
	# 		act_vec = data[0:locs].T[t_test]
	# 		pred_vec = dmd.predict(results,t_test)
	# 		first_error_inst = la.norm(act_vec - pred_vec) / la.norm(act_vec)
	# 		print(point,start,first_error_inst)
	# 		first_error_hold.append(first_error_inst*100)
	# 	error.append(min([np.mean(error_hold),100]))
	# 	end_error.append(min([np.mean(end_error_hold),100]))
	# 	first_error.append(min([np.mean(first_error_hold),100]))

	# plt.plot(points,error)
	# plt.plot(points,end_error)
	# plt.plot(points,first_error)
	# plt.legend(['Reconstruction','Last Column','First Prediction'])
	# plt.show()


	# prediction testing outside training
	# steps = np.arange(0,3)
	# print(steps)
	# t_new = max(t)
	# dt = t[1] - t[0]
	# all_error = []
	# for step in steps:
	# 	t_new += dt
	# 	t = np.append(t,t_new)
	# 	act_vec = data[start:locs].T[t_new]
	# 	pred_vec = dmd.predict(results,t_new)
	# 	error = la.norm(act_vec - pred_vec) / la.norm(act_vec)
	# 	all_error = np.append(all_error,error)
	# 	print('time:',t_new)
	# 	print('prediction:\n',pred_vec)
	# 	print('actual:\n',act_vec)
	# # plt.plot(t,all_error)
	# # plt.show()
	# print(t)

	

	# plotting testing
	# surf = visualize.surface_data(np.real(F),t,x)
	# surf2 = visualize.surface_data(np.real(results.Xdmd),t,x)
	# plt.show()

	# predcition testing inside training
	# num = 10
	# t_test = t[num]
	# act_vec = np.real(F[:,num])
	# pred_vec = dmd.predict(results,t_test)
	# error = la.norm(act_vec - pred_vec) / la.norm(act_vec)
	# # print('Training Prediction Error:',error,'\n')

	# # prediction testing outside training
	# steps = np.arange(1,1e4,10)
	# t_new = max(t)
	# dt = t[1] - t[0]
	# all_error = []
	# t = []
	# for step in steps:
	# 	t_new += dt*step
	# 	t.append(t_new)
	# 	act_vec = np.real(f(x,t_new))
	# 	pred_vec = dmd.predict(results,t_new)
	# 	error = la.norm(act_vec - pred_vec) / la.norm(act_vec)
	# 	all_error.append(error)
	# # plt.plot(t,all_error)
	# # plt.show()

	








