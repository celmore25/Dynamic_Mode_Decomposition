import numpy as np
from scipy import linalg as la
from math import floor
from cmath import exp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from usa import *


class manipulate:

	'''
	This class contains helpful functions for the specific matrix 
	manipulation needed for Dynamic Mode Decomposition.
	'''

	def two_2_three(A, n, verbose=False):
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
		m = int(rows / n)
		if rows % m != 0:
			print('Invalid column input. \nFunction Exited.')
			return None

		# initalize A_3, the matrix
		A_3 = np.zeros((cols, m, n))

		# now loop over the columns construct the matrices and place them
		# in the A_3 matrix
		a_hold = np.zeros((m, n))
		for i in range(cols):
			# grab the column that holds the vector
			col_hold = A[:, i]

			# convert the column into the holder matrix shape
			for j in range(n):
				a_hold[:, j] = col_hold[j * m:(j + 1) * m]

			# place the matrix in the new 3D holder
			A_3[i] = a_hold

		# print the transformation that occured
		if verbose:
			print('A =\n', A, '\n')
			print('A_3 =\n', A_3, '\n')

		# the program is finished
		return A_3

	def three_2_two(A, verbose=False):
		
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
		n = length  # number of columns for A_2

		# calculate the length of each column needed (number of rows)
		m = rows * cols

		# initalize A_2, the matrix to be returned
		A_2 = np.zeros((m, n))

		# now loop over the matricies, construct the columns and place them
		# in the A_2 matrix
		vec = np.zeros(m)
		A_2 = A_2.transpose()
		for i in range(n):
			# grab the matrix
			matrix = A[i]

			# loop through the columns and store them in a_hold
			for j in range(cols):
				vec[j * rows:(j + 1) * rows] = matrix[:, j]

			# place the matrix in the new 3D holder
			A_2[i] = vec
		A_2 = A_2.transpose()
		# print the transformation that occured
		if verbose:
			print('A =\n', A, '\n')
			print('A_2 =\n', A_2, '\n')

		# the program is finished
		return A_2

	def split(Xf, verbose=False):
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
			print('Xf =\n', Xf, '\n')

		X = Xf[:, :-1]
		Xp = Xf[:, 1:]

		if verbose:
			print('X =\n', X, '\n')
			print('Xp =\n', Xp, '\n')
		return X, Xp


class cluster:

	'''
	This will hold the functions needed to cluster the data into different
	geographic sections based on error from the first sensitivity analysis.
	'''

	def KKN(result):

		'''
		'''

		sets = []
		return sets

	def bayes(result):

		'''
		'''
		
		sets = []
		return sets

	def neural(result):

		'''
		'''
		
		sets = []
		return sets

	def logistic(result):

		'''
		'''
		
		sets = []
		return sets

		
class write:

	'''
	This class holds the functions that are used for writing parallel
	tasks to be used by the CRC. This can easily be modified to fit any
	supercomputing system. 

	Also, this class contains writing function for how to easily make 
	delimited text files from analysises created by testing classes.
	'''

	def sv_dependence_results(name,results,data,N = 1,description = ''):

		'''
		This will write the results obtained from a singular value 
		dependence test.

		input:
		results - object returned from the singular dependence test
		description - description of the test being run
		N - number of % error values wanted to be reported
		data - data used for the results

		output(not returned):
		file - tab delimited file containing all important results
		'''

		name = name + '.txt'

		with open(name,'w') as output:
			# description 
			output.write('Test Description:\n')
			output.write(description+'\n\n')

			# full results
			output.write('Full Reconstruction Results\n\n')
			output.write('error\tnumber of singular values\tcondition number\tsmallest singular value\n')
			for ind,val in enumerate(results.error):
				output.write(str(results.error[ind])+'\t')
				output.write(str(results.num_svd[ind])+'\t')
				output.write(str(results.condition[ind])+'\t')
				output.write(str(results.smallest_svd[ind])+'\n')

			output.write('\nLast t Measurement Errors\n')

		# all residuals
		test_num = N
		end_val_vec = np.array(range(test_num)) + 1
		for end_val in end_val_vec:
			end_error = energy.calc_end_error(end_val,results.res_matrix,data,verbose = False)

			with open(name,'ab') as output:
				np.savetxt(output,matrix,delimiter = '\t')

		return True


class examples:

	'''
	This class will hold functions that will give very simple examples of 
	how DMD works in this library of functions. There will be theoretical
	examples as well as data driven examples in this class once the class 
	is out of development
	'''

	def kutz():

		'''
		This is a simple example of how DMD can be used to reconstruct
		a complex example from Kutz's book on DMD.
		'''

		print('To show how DMD can be performed using the class given')
		print('let us take a look at an example from Kutz\'s book on DMD\n')

		print('We will look at a complex, periodic function given below:\n')

		print('f(x,t) = sech(x+3)exp(2.3it) + 2sech(x)tanh(x)exp(2.8it)\n')

		print('Now, the 3D function will be plotted on a surface plot as well as its')
		print('DMD reconstruction based on rank reduction at 1,2, and 3 singular values.\n')

		print('It can be shown that this function only has rank = 2, so notice how the DMD')
		print('reconstruction at rank = 3 is pretty much identical to the rank = 2 surface.\n')


		# testing function from book
		im = 0+1j
		sech = lambda x: 1/np.cosh(x)
		f = lambda x,t: sech(x + 3)*exp(2.3*im*t) + 2*sech(x)*np.tanh(x)*exp(im*2.8*t)
		points = 100
		x = np.linspace(-10,10,points)
		t = np.linspace(0,4*np.pi,points)

		# test decomposition of the function given above
		F = np.zeros((np.size(x),np.size(t)),dtype=np.complex_)
		for i,x_val in enumerate(x):
			for j,t_val in enumerate(t):
				F[i,j] = f(x_val,t_val)
		results1 = dmd.decomp(F,t,verbose = False,num_svd=1,svd_cut=True)
		results2 = dmd.decomp(F,t,verbose = False,num_svd=2,svd_cut=True)
		results3 = dmd.decomp(F,t,verbose = False,num_svd=3,svd_cut=True)

		# plotting 

		# make the figure
		fig = plt.figure(figsize = (10,7))
		surf_real_ax = fig.add_subplot(2, 2, 1, projection='3d')
		surf1_ax = fig.add_subplot(2, 2, 2, projection='3d')
		surf2_ax = fig.add_subplot(2, 2, 3, projection='3d')
		surf3_ax = fig.add_subplot(2, 2, 4, projection='3d')

		surf_real_ax = visualize.surface_data(np.real(F),t,x\
								,provide_axis = True, axis = surf_real_ax)
		surf1_ax = visualize.surface_data(np.real(results1.Xdmd),t,x\
								,provide_axis = True, axis = surf1_ax)
		surf2_ax = visualize.surface_data(np.real(results2.Xdmd),t,x\
								,provide_axis = True, axis = surf2_ax)
		surf3_ax = visualize.surface_data(np.real(results3.Xdmd),t,x\
								,provide_axis = True, axis = surf3_ax)
		
		surf_real_ax.set_xlabel('t')
		surf_real_ax.set_ylabel('x')
		surf_real_ax.set_zlabel('f(x,t)')
		surf_real_ax.set_title('Original function')

		surf1_ax.set_xlabel('t')
		surf1_ax.set_ylabel('x')
		surf1_ax.set_zlabel('f(x,t)')
		surf1_ax.set_title('1 Normal Mode')

		surf2_ax.set_xlabel('t')
		surf2_ax.set_ylabel('x')
		surf2_ax.set_zlabel('f(x,t)')
		surf2_ax.set_title('2 Normal Modes')

		surf3_ax.set_xlabel('t')
		surf3_ax.set_ylabel('x')
		surf3_ax.set_zlabel('f(x,t)')
		surf3_ax.set_title('3 Normal Modes')

		plt.show()

		return fig


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
		name - string with name of the csv filei
		outputs:
		numpy array with price data over time
		'''

		X = np.genfromtxt(name, delimiter=',')
		X = X[1:]
		return X

	def data_wrangle(data,start_loc,end_loc,start_time,end_time):

		''' 
		This function will just return a numpy array with the given
		specifications of where you want to start the time and the 
		location examples.
		'''

		X = data[start_loc:end_loc]
		X = X.T
		X = X[start_time:end_time]
		X = X.T

		return X

	def calc_end_error(end_val,error,data,verbose = False):

		''' 
		This function calculates the 2-norm error of a matrix of 
		residual values based on the last "end_val" times.
		The input is a list of residual matricies like the 
		"sv_dependence" function returns. 
		'''

		# determine the 2-norm of the data in the last time measurements
		time_len = np.size(data[0])
		data = data.T
		data = data[time_len - end_val:]
		data = data.T
		data_norm = la.norm(data)

		# initalize a list for the error values
		end_error = []

		# loop through to find the error for each sv
		for test_ind, res_matrix in enumerate(error):
			

			# grab the last end_vals
			res_matrix = res_matrix.T
			res_matrix = res_matrix[time_len - end_val:]
			res_matrix = res_matrix.T

			# calculate the error
			error = la.norm(res_matrix) / data_norm * 100

			# append the error
			end_error.append(error)

			if verbose:
				print('------------------------------------')
				print('Test #'+str(test_ind))
				print()
				print(res_matrix)
				print(la.norm(res_matrix))
				print()
				print(data)
				print(data_norm)
				print('Error:',error)
				print('------------------------------------')
				print()

		return end_error

	def calc_opt_sv_cut(results,data,N = 24,verbose = False):
		'''
		From a singular value sensitivity test class, this will calculate
		the optimal rank reduction given a time period to test on.

		inputs:
		results - class returned from sv_sensitivity
		N - time period on which to test

		outputs:
		opt_sv - optimal singular value to cut on
		end_error - array that has the error for each rank reduction
		'''

		# determine the optimal number of singular values
		end_error = energy.calc_end_error(N,results.res_matrix,data,verbose = False)
		opt_sv = end_error.index(min(end_error)) + 1
		if verbose:
			print('For a time period of',N,'hours...')
			print('\nOptimal Singular Value Reduction Identified:',opt_sv)
			print('Percentage:',opt_sv/np.size(results.num_svd)*100,'%')

		return opt_sv, end_error


class dmd:
	
	'''
	This class contains the functions needed for performing a full DMD
	on any given matrix. Depending on functions being used, different 
	outputs can be achived.
	This class also contains functions useful to the analysis of DMD
	results and intermediates.
	'''

	def decomp(Xf, time, verbose=False, rank_cut=True, esp=1e-2, svd_cut=False,
			   num_svd=1, do_SVD=True, given_svd=False):

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
		***  see ### (10) ### below  ***

		options:
		verbose - boolean for more information
		svd_cut - boolean for truncation of SVD values of X
		esp - value to truncate singular values lower than
		rank_cut - truncate the SVD of X to the rank of X
		num_svd - number of singular values to use
		do_SVD - tells the program if the svd is provided to it or not
		'''

		if verbose:
			print('Entering Dynamic Mode Decomposition:\n')

		# --- (1) --- #
		# split the Xf matrix
		X, Xp = manipulate.split(Xf)
		if verbose:
			print('X = \n', X, '\n')
			print('X` = \n', Xp, '\n')

		### (2) ###
		# perform a singular value decompostion on X
		if do_SVD:
			if verbose:
				'Performing singular value decompostion...\n'
			U, S, Vh = la.svd(X)
		else:
			if verbose:
				'Singular value decompostion provided...\n'
			U, S, Vh = given_svd

		if verbose:
			print('Singular value decomposition:')
			print('U: \n', U)
			print('S: \n', S)
			print('Vh: \n', Vh)
			print('Reconstruction:')
			S_m = np.zeros(np.shape(X))
			for i in range(len(list(S))):
				S_m[i, i] = S[i]
			recon = np.dot(np.dot(U, S_m), Vh)
			print('X =\n', recon)

		# perfom desired truncations of X
		if svd_cut:
			rank_cut = False
		if rank_cut:  # this is the default truncation
			rank = 0
			for i in S:
				if i > esp:
					rank += 1
			if verbose:
				print('Singular Values of X:', '\n', S, '\n')
				print('Reducing Rank of System...\n')
			Ur = U[:, 0:rank]
			Sr = S[0:rank]
			Vhr = Vh[0:rank, :]
			if verbose:
				recon = np.dot(np.dot(Ur, np.diag(Sr)), Vhr)
				print('Rank Reduced reconstruction:\n', 'X =\n', recon)
		elif svd_cut:
			rank = num_svd
			if verbose:
				print('Singular Values of X:', '\n', S, '\n')
				print('Reducing Rank of System to n =', num_svd, '...\n')
			Ur = U[:, 0:rank]
			Sr = S[0:rank]
			Vhr = Vh[0:rank, :]
			if verbose:
				recon = np.dot(np.dot(Ur, np.diag(Sr)), Vhr)
				print('Rank Reduced reconstruction:\n', 'X =\n', recon)

		# return the condition number to view singularity
		condition = max(Sr) / min(Sr)
		smallest_svd = min(Sr)
		svd_used = np.size(Sr)
		if verbose:
			condition = max(Sr) / min(Sr)
			print('Condition of Rank Converted Matrix X:' \
					  , '\nK =', condition, '\n')

		# make the singular values a matrix and take the inverse
		Sr_inv = np.diag([i ** -1 for i in Sr])
		Sr = np.diag(Sr)

		### (3) ###
		# now compute the A_t matrix
		Vr = Vhr.conj().T
		At = Ur.conj().T.dot(Xp)
		At = At.dot(Vr)
		At = At.dot(la.inv(Sr))
		if verbose:
			print('A~ = \n', At, '\n')

		### (4) ###
		# perform the eigen decomposition of At
		L, W = la.eig(At)

		### (5) ###
		# compute the DMD modes
		phi = np.dot(Xp, Vhr.conj().T)
		phi = np.dot(phi, Sr_inv)
		phi = np.dot(phi, W)

		if verbose:
			print('Normal Mode Matrix:', '\nPhi =', phi, '\n')

		### (6) ###
		# compute the continuous and discrete eigenvalues
		dt = time[1] - time[0]
		lam = L
		omg = np.log(lam) / dt
		if verbose:
			print('Discrete time eigenvalues:\n', 'Lambda =', L, '\n')
			print('Continuous time eigenvalues:\n', 'Omega =', np.log(L) / dt, '\n')

		### (7) ###
		# compute the amplitude vector b by solving the linear system described
		# note that a least squares solver has to be used in order to approximate
		# the solution to the overdefined problem
		x1 = X[:, 0]
		b = la.lstsq(phi, x1)
		b = b[0]
		if verbose:
			print('b =\n', b, '\n')

		### (8) ###
		# finally reconstruct the data matrix from the DMD modes
		length = np.size(time)  # number of time measurements
		dynamics = np.zeros((rank, length), dtype=np.complex_)  # initialize the time dynamics
		for t in range(length):
			omg_p = np.array([exp(i * time[t]) for i in omg])
			dynamics[:, t] = b * omg_p

		if verbose:
			print('Time dynamics:\n', dynamics, '\n')
		Xdmd = np.dot(phi, dynamics)
		if verbose:
			print('Reconstruction:\n', np.real(Xdmd), '\n')
			print('Original:\n', np.real(Xf), '\n')

		### (9) ###
		# calculate some residual value
		res = np.real(Xf - Xdmd)
		error = la.norm(res) / la.norm(Xf)
		if verbose:
			print('Reconstruction Error:', round(error * 100, 2), '%')

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

		return results()

	def predict(dmd, t):

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
		dynamics = np.zeros((dmd.rank, 1), dtype=np.complex_)
		omg_p = np.array([exp(i * t) for i in dmd.omg])
		dynamics = dmd.b * omg_p
		x = np.real(np.dot(dmd.phi, dynamics))

		return x

	def dmd_specific_svd(Xf):

		''' 
		This is a helper function which will split the data and
		perform a singular value decomposition based on whatever the
		input data is and return the outputs for scipy. 
		'''

		X, Xp = manipulate.split(Xf)
		result = la.svd(X)

		return result


class tests:

	'''
	This class holds functions for various tests to perform 
	on dmd predictions and reconstructions of any data set
	'''

	def sv_dependence(data,verbose = False):

		'''
		This function will take a set of input data and return
		an array of dependence of the data on the rank truncation
		that is performed. 

		Required inputs:
		data - Data matrix to be decomposed

		outputs:
		num_svd - array of the number of number of singular values used
		condition - array of the condition number
		smallest_svd - array of the smallest sv used
		error - array of the error of the reconstruction
		res_matrix - array of the residual of data and the reconstruction
		opt_sv_full - rank reduction to use to minimize all the full error
		'''

		if verbose:
			print('---- Singular Value Dependence Test ---- \n')

		# we are assuming the input data is wrangled and will just decompose
		# whatever is given to the function

		# make an arbitrary time vector
		t = np.arange(int(np.size(data[0])))

		# intialize the arrays for each of the things you will want to capture
		error = []
		res_matrix = []
		num_svd = []
		condition = []
		smallest_svd = []

		# determine the max rank
		max_rank = min( [len(t) , int(np.size(data.T[0]))] )

		# make the vector for the number of svd's to use
		sv_vec = np.arange(max_rank - 1) + 1

		# perform the singular value decomposition for speed of the algorithm
		svd_calc = dmd.dmd_specific_svd(data)

		# loop through each rank trunction and do the error calculation
		for i in sv_vec:
			
			# give the user some idea of where they are in the test
			if verbose:
				print('singular value: ',i,' of ',max_rank - 1)

			# perform the decomposition
			results = dmd.decomp(data, t, num_svd=i, svd_cut=True , verbose = False,
									given_svd = svd_calc, do_SVD = False)

			# store the desired values
			error.append(results.error)
			num_svd.append(results.svd_used)
			condition.append(results.condition)
			smallest_svd.append(results.smallest_svd)

			# also calculate the full residual matrices
			matrix = np.real(data - results.Xdmd)
			res_matrix.append(matrix)

		# calculate the best rank reconstruction 
		opt_sv_full = error.index(min(error)) + 1

		class results:
			def __init__(self):
				self.error = np.array(error)
				self.res_matrix = np.array(res_matrix)
				self.num_svd = np.array(num_svd)
				self.condition = np.array(condition)
				self.smallest_svd = np.array(smallest_svd)
				self.opt_sv_full = opt_sv_full

		if verbose:
			print('Sensitivity Test Finished\n\n')

		return results()


	def sv_test_run_and_plot(start_loc,end_loc,start_time,end_time,data,N = 24,full = False,\
		verbose = False, condition = False, smallest_svd = False):

		''' 
		This will run and plot a singular value sensitivity test of any section of 
		data from a given dataset.

		inputs:
		start_loc - starting location coordinate
		end_loc - ending location coordinate
		start_time - starting time
		end_time - ending time
		data - price data
		N - optimal singular cut testing point (set to one period)
		full - plotting the whole reconstruction or not
		condition - graph of condition number
		smallest_svd - graph of smallest singular value used

		outputs:
		results - results from test 
		figure - optimal figure for N testing
		*** note: the "optimal_svd" for the N value is now also returned as "opt_sv" ***
		***		  also, the value can be retrived for any N measurements with    ***
		***		  the new function "energy.calc_opt_sv_cut" from the results     ***
		*** 	  class that is given to this function.

		'''

		# get the matrix we want
		data_new = energy.data_wrangle(data,start_loc,end_loc,start_time,end_time)

		# perform the test
		results = tests.sv_dependence(data_new,verbose = verbose)

		# determine the optimal number of singular values
		opt_sv, end_error = energy.calc_opt_sv_cut(results,data_new, N = N, verbose = verbose)

		# plot the optimal point speicified by the user
		plt.figure()
		plt.xlabel('number of singular values (%)')
		plt.ylabel('error for last '+str(N)+' hours')
		plt.grid('on')
		plt.ylim([0,100])
		plt.plot(results.num_svd/np.size(results.num_svd)*100, end_error)
		plt.plot(opt_sv/np.size(results.num_svd)*100, end_error[opt_sv - 1],'r.')
		plt.title('Optimal Point: (rank = '+str(opt_sv)+' , error = '+\
			str(round(end_error[opt_sv - 1],1))+'%)')
		figure = plt.gcf()

		# plotting results of test only for the full reconstructions
		if full:
			plt.figure()
			plt.xlabel('number of singular values (%)')
			plt.ylabel('full error')
			plt.plot(results.num_svd/np.size(results.num_svd)*100, results.error)
			plt.grid('on')
			plt.ylim([0,100])
			opt_sv_full = results.opt_sv_full
			plt.plot(opt_sv_full/np.size(results.num_svd)*100, results.error[opt_sv_full - 1],'r.')
			plt.title('Optimal Point: (rank = '+str(opt_sv_full)+' , error = '+\
				str(round(results.error[opt_sv_full - 1],1))+'%)')

			# plot condition numbers 
			if condition:
				plt.figure()
				plt.xlabel('log10 condition number')
				plt.ylabel('full error')
				plt.plot(np.log10(results.condition), results.error)
				plt.grid('on')
				plt.ylim([0,100])

			# plot smallest singular values
			if smallest_svd:
				plt.figure()
				plt.xlabel('log10 condition number')
				plt.ylabel('full error')
				plt.plot(np.log10(results.smallest_svd), results.error)
				plt.grid('on')
				plt.ylim([0,100])

		# LEGACY CODE MAY BE USEFUL LATER 
		# all_mins = []
		# end_val_vec = np.array(range(test_num)) + 1
		# for end_val in end_val_vec:
			# all_mins.append(min_ind)
		# all_mins = np.array(all_mins)
		# rank = int(floor(np.mean(all_mins)))
		# plt.legend(end_val_vec)

		return results, opt_sv, figure


	def sv_test_plot(data_new,results,N = 6,full = False):

		'''
		This makes some simple plots for the singular value dependency test.

		input:
		results - object from the testing results
		data_new - data used
		N - number for plotting at ending times and optimal SV
		full - boolean for wanting to plot the full reconstruction or not
		'''

		if full:
			# plotting results of full test
			plt.figure()
			plt.xlabel('number of singular values (%)')
			plt.ylabel('full error')
			plt.plot(results.num_svd/np.size(results.num_svd)*100, results.error)
			plt.grid('on')
			plt.ylim([0,100])

			plt.figure()
			plt.xlabel('log10 condition number')
			plt.ylabel('full error')
			plt.plot(np.log10(results.condition), results.error)
			plt.grid('on')
			plt.ylim([0,100])


		# plotting for last N values
		plt.figure()
		plt.xlabel('number of singular values (%)')
		plt.ylabel('error in ending segment')
		plt.grid('on')
		plt.ylim([0,100])
		all_mins = []
		test_num = N
		end_val_vec = np.array(range(test_num)) + 1
		for end_val in end_val_vec:
			end_error = energy.calc_end_error(end_val,results.res_matrix,data_new,verbose = False)
			plt.plot(results.num_svd/np.size(results.num_svd)*100, end_error)
			min_ind = end_error.index(min(end_error))
			all_mins.append(min_ind)
		all_mins = np.array(all_mins)
		rank = int(floor(np.mean(all_mins)))
		print('\nOptimal Singular Value Reduction Identified:',rank)
		print('Percentage:',rank/np.size(results.num_svd)*100,'%')
		plt.legend(end_val_vec)
		figure = plt.gcf()

		return figure


	def location_analysis(start_loc,end_loc,start_time,end_time,data,rank_cut,
		N = 6,verbose = False):

		'''
		This will perform a single reconstruction specified by the user and return
		error information about each of the locations that were reconstructed. 

		inputs:
		data - price matrix
		rank_cut - the number of singular values to use
		N - number of time incremints before the end to return error for

		outputs:
		results - (class that holds the following information)
		error_mat - full error matrix
		full_error - average error for each location
		end_error - list of average errors for each last N segments for each location
					(starts at 1 and goes to N)
		test_index - the index used to the end of the data to calculate error

		'''

		# get the data in the correct form
		data_new = energy.data_wrangle(data,start_loc,end_loc,start_time,end_time)

		# perform the singular value decomposition for speed of the algorithm in possible
		# future applications
		svd_calc = dmd.dmd_specific_svd(data_new)

		# make an arbitrary time vector
		t = np.arange(int(np.size(data_new[0])))

		# perform the decomposition
		dmd_result = dmd.decomp(data_new, t, num_svd = rank_cut, svd_cut = True , verbose = False,\
								given_svd = svd_calc, do_SVD = False)

		# calculate the error (normalized on all points)
		error_mat = 100 * np.absolute(np.real(data_new - dmd_result.Xdmd) / data_new)

		# calculate the average % error for each location
		full_error = []
		for loc_ind,loc_error in enumerate(error_mat):
			full_error.append(np.mean(loc_error))
		full_error = np.array(full_error)

		# now calculate the error for the last N point described
		end_error = []
		end_val_vec = (np.array(range(N)) + 1) * (-1) + np.size(t)
		for ind in end_val_vec:
			# intialize a holder array
			loc_holder = []

			# get the data we actually want at the end
			end_mat = error_mat.T[ind:].T

			# calculate based on each location
			for loc_error in end_mat:
				loc_holder.append(np.mean(loc_error))

			# append the holder
			end_error.append(loc_holder)

		# convert the ending list and transpose
		end_error = np.array(end_error).T

		# class for the returning of all results
		class results:
			def __init__(self):
				self.error_mat = error_mat
				self.full_error = full_error
				self.end_error = end_error
				self.test_index = end_val_vec

		return results()


	def prediction_test(data,train_start,train_end,num_predict,
		start_loc,end_loc,rank,verbose = False, plot = False):

		'''
		This will run a very basic prediction test where you supply the price dataset,
		the number of training points, and the number of points to go into the future.

		inputs:
		data - price dataset
		train_start - point to start training set
		train_end - point to start training set
		num_predict - time incremints to predict to
		start_loc,end_loc - location coordinates
		rank - sv's to use
		plot - desired plots to accompany the analysis or not

		outputs:
		error_mat - matrix of % error
		price_mat - matrix of prediction prices
		'''

		# get the data in the correct form for training and prediction
		data_train = energy.data_wrangle(data,start_loc,end_loc,train_start,train_end)
		data_pred = energy.data_wrangle(data,start_loc,end_loc,train_end,train_end + num_predict)

		# make an arbitrary time vector for training and prediction
		t_train = np.arange(int(np.size(data_train[0])))
		t_pred = np.arange(num_predict) + int(np.size(data_train[0]))

		# perform the singular value decomposition for speed of the algorithm in possible
		# future applications
		svd_calc = dmd.dmd_specific_svd(data_train)

		# perform the decomposition on the training set
		dmd_result = dmd.decomp(data_train, t_train, num_svd = rank, svd_cut = True , verbose = False,\
								given_svd = svd_calc, do_SVD = False)

		# make the prediction matrix for the rest of the time vector
		price_pred = np.zeros((end_loc - start_loc , num_predict))

		# and now make the prediction at each time
		for ind,t in enumerate(t_pred):
			price_pred.T[ind] = dmd.predict(dmd_result,t)

		# an finally create the matricies that will be returned
		price_dmd = np.concatenate((np.real(dmd_result.Xdmd), np.real(price_pred)), axis=1)
		price_real = np.concatenate((data_train, data_pred), axis=1)
		error_mat = np.absolute((price_dmd - price_real) / price_real) * 100

		class results:
			def __init__(self):
				self.price_dmd = price_dmd
				self.error_mat = error_mat
				self.price_real = price_real
				self.num_predict = num_predict

		if verbose:
			print('-------------------------------------------------------------------------------')
			print('training data:\n',data_train)
			print('prediction data:\n',data_pred,'\n')
			print('training time:\n',t_train)
			print('prediction time:\n',t_pred,'\n')
			print('X_dmd\n',np.real(dmd_result.Xdmd),'\n')
			print('Price Prediction:\n',price_pred,'\n')
			print('Price Matrix:\n',price_dmd,'\n')
			print('Error Matrix:\n',error_mat,'\n')
			print('-------------------------------------------------------------------------------')

		return results()


	def plot_prediction(results,locations,average = False,
		both=False,error_only=False,price_only = True):

		''' 
		This will make some simple plots of given locations or the average for a system
		given the prediciton test above.

		inputs:
		results - result from prediciton_test which has error and price for dmd and data
		locations - list of indicies that are desried to be plotted

		outputs:
		error_fig - figure of error
		price_fig - figure for price
		'''

		# cast locations
		locations = np.array(locations)

		# make a time vector:
		t = np.arange(int(np.size(results.price_real[0])))

		# create figures and axes
		price_fig = plt.figure()
		price_ax = price_fig.gca()
		error_fig = plt.figure()
		error_ax = error_fig.gca()
		
		# loop through the locations and plot the results (only if there are any)
		colors = cm.jet(np.linspace(0, 1, np.size(locations)))
		if locations.any():
			for loc,c in zip(locations,colors):
				dmd_vec = results.price_dmd[loc]
				data_vec = results.price_real[loc]
				error_vec = results.error_mat[loc]
				price_ax.plot(t,dmd_vec,color=c,label=str(loc))
				price_ax.plot(t,data_vec,'--',color=c)
				error_ax.plot(t,error_vec,color=c,label=str(loc))

		# plot the average error and prices as well
		if average:
			mean_dmd_vec = []
			mean_data_vec = []
			mean_error_vec = []
			for ind in t:
				mean_dmd_vec.append(np.mean(results.price_dmd.T[ind]))
				mean_data_vec.append(np.mean(results.price_real.T[ind]))
				mean_error_vec.append(np.mean(results.error_mat.T[ind]))
			price_ax.plot(t,mean_dmd_vec,'k',label='mean')
			price_ax.plot(t,mean_data_vec,'k--')
			error_ax.plot(t,mean_error_vec,'k',label='mean')

		# make a line for the prediction time
		x = [np.size(t) - results.num_predict,np.size(t) - results.num_predict]
		y = [0,100]
		price_ax.plot(x,y,'k--')
		error_ax.plot(x,y,'k--')


		# make labels and such
		error_ax.legend()
		price_ax.legend()
		error_ax.set_ylabel('Error (%)')
		price_ax.set_ylabel('Price ($)')
		error_ax.set_xlabel('Time (hours)')
		price_ax.set_xlabel('Time (hours)')
		price_ax.grid('on')
		error_ax.grid('on')
		error_ax.set_title('Error')
		error_ax.axis([0, np.size(t), 0, 100])
		price_ax.axis([0, np.size(t), 0, 80])

		if both:
			return error_fig, price_fig
		if price_only:
			return price_fig
		if error_only:
			return error_fig


if __name__ == '__main__':
	# print('DMD testing environment entered.\n')

	# import data
	data = energy.imp_prices('prices.csv')

	# define the number of locations and time
	start_loc = 0
	end_loc = 3380
	train_start = 0
	train_end = 40
	results,opt_sv,figure = tests.sv_test_run_and_plot(start_loc,end_loc,train_start,train_end\
	                                 ,data,N = 24,full = True,verbose = True)

	# define the number of locations and time
	# start_loc = 0
	# end_loc = 3380
	# train_start = 0
	# train_end = 312
	# num_predict = 24
	# rank = 294
	# results = tests.prediction_test(data,train_start,train_end,num_predict,
	#     start_loc,end_loc,rank,verbose = False)
	# locations = [3]
	# error, price = tests.plot_prediction(results,locations, average = False)
	# plt.show(price)

	# store the results in a file
	# write.sv_dependence_results('test1',results,data_new, N = 5)

	# fig = examples.kutz()







