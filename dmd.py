import numpy as np
from scipy import linalg as la


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
	def decomp(Xf,verbose=False,rank_cut=True,truncate=False,esp=1e-8):
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

		inputs:
		X - (mxn) Matrix

		outputs:
		Phi - DMD modes

		options:
		verbose - boolean for more information
		truncate - boolean for truncation of SVD values of X
		esp - value to truncate singular values lower than
		rank_cut - truncate the SVD of X to the rank of X
		'''
		if verbose:
			print('Entering Dynamic Mode Decomposition:\n')

		### (1) ####
		# split the Xf matrix 
		X,Xp = manipulate.split(Xf)

		### (2) ###
		# perform a singular value decompostion on X
		U,S,Vh = la.svd(X)

		# truncate the SVD to the rank of X
		rank = np.count_nonzero(S)
		Ur = U[:,0:rank]
		Sr = S[0:rank]
		Vhr = Vh[0:rank,:]

		# return the condition number to view singularity
		if verbose:
			print('Rank of X matrix:',rank,'\n')
			cond = max(Sr)/min(Sr)
			print('Condition of Rank Converted Matrix X:'\
											,'\nK =',cond,'\n')

		# make the singular values a matrix and take the inverse
		Sr_inv = np.diag([i**-1 for i in Sr])
		Sr = np.diag(Sr)

		### (3) ###
		# now compute the A_t matrix 
		At = np.dot(Ur.transpose(),Xp)
		At = np.dot(At,Vhr.transpose())
		At = np.dot(At,Sr_inv)

		### (4) ###
		# perform the eigen decomposition of At
		L,W = la.eig(At)

		### (5) ### 
		# compute the DMD modes
		phi = np.dot(Xp,Vhr.transpose())
		phi = np.dot(phi,Sr_inv)
		phi = np.dot(phi,W)

		if verbose:
			print('Normal Mode Matrix:','\nPhi =',phi)

		return phi

if __name__ == '__main__':
	print('DMD environment entered.\n')


	A = np.array([[1,2,3,4],[5,6,7,8],[1,2,-3,4],[-5,-6,-7,-8]])
	A_3 = manipulate.two_2_three(A,2,verbose = True)
	A = manipulate.three_2_two(A_3,verbose = True)
	X,Xp = manipulate.split(A,verbose = True)
	dmd.decomp(A,verbose=True)






