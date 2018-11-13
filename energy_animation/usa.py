import numpy as np
from scipy import linalg as la
from cmath import exp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation

from dmd import *
import time

class cal_plot:
	'''
	This class will hold all necessary function for the plotting
	of any energy stuff on a California map
	'''
	def plot_price(data,locations,time = 0):
		'''
		This will be the basic plotting function to allow you to plot
		energy prices on a colormap on california.

		Inputs:
		data - (mxn) matrix where columns are times and row are locations
		locations - standard 
		time - integer for time you want to plot
		'''
		fig,ax = plt.subplots()
		lat = locations[:,0]
		lon = locations[:,1]
		price = data[:,time]

		max_price = 70
		min_price = 20

		data_plot = ax.scatter(lon,lat,alpha = 0.5,c=price, cmap = cm.jet, s=8\
			,vmin = min_price, vmax = max_price)
		plt.colorbar(data_plot,label = 'Price ($)')
		ax.axis([-126.30, -104, 32.45, 46.73])
		map_img = mpimg.imread('map.png')
		plt.imshow(map_img, extent=[-126.30, -104, 32.45, 46.73])
		ax.axis("off")
		title = 'Time: '+str(time)
		ax.set_title(title)

		return fig

	def animate_price(data,locations,time = 100,speed = 150,label = 'Price'):
		'''
		This will be the basic plotting function to allow you to plot
		energy prices on a colormap on california.

		Inputs:
		data - (mxn) matrix where columns are times and row are locations
		locations - standard 
		time - integer for time you want to plot
		speed - input into the animation object for frame update speed
		'''
		fig,ax = plt.subplots()
		lat = locations[:,0]
		lon = locations[:,1]
		price = data[:,0]

		max_price = 60
		min_price = 10

		data_plot = ax.scatter(lon,lat,alpha = 0.5,c=price, cmap = cm.jet, s=8\
					,vmin = min_price, vmax = max_price)
		plt.colorbar(data_plot,label = 'Price ($)')
		ax.axis([-126.30, -104, 32.45, 46.73])
		map_img = mpimg.imread('map.png')
		plt.imshow(map_img, extent=[-126.30, -104, 32.45, 46.73])
		ax.axis("off")
		label = label + '\n'
		
		def update(t):
			price = data[:,t]
			title = 'Time: '+str(t)
			ax.set_title(label+title)
			data_plot.set_array(price)

		animation = FuncAnimation(fig, update, interval = speed, frames = time)

		return animation
		
	def animate_error(error,locations,time = 100,speed = 150):
		'''
		This will be the basic plotting function to allow you to plot
		energy prices on a colormap on california.

		Inputs:
		data - (mxn) matrix where columns are times and row are locations (0,1)
		locations - standard 
		time - integer for time you want to plot
		speed - input into the animation object for frame update speed
		'''
		fig,ax = plt.subplots()
		lat = locations[:,0]
		lon = locations[:,1]
		price = error[:,0]

		max_error = 10
		min_error = 0

		data_plot = ax.scatter(lon,lat,alpha = 0.5,c=price, cmap = cm.jet, s=8\
					,vmin = min_error, vmax = max_error)
		plt.colorbar(data_plot,label = 'absolute error')
		ax.axis([-126.30, -104, 32.45, 46.73])
		map_img = mpimg.imread('map.png')
		plt.imshow(map_img, extent=[-126.30, -104, 32.45, 46.73])
		ax.axis("off")
		
		def update(t):
			price = error[:,t]
			title = 'Time: '+str(t)
			ax.set_title(title)
			data_plot.set_array(price)

		animation = FuncAnimation(fig, update, interval = speed, frames = time)
		plt.show()

	def animate_all(data,dmd_data,error_data,locations,time = 100,speed = 150):
		'''
		This will be the basic plotting function to allow you to plot
		energy prices on a colormap on california.

		Inputs:
		data - (mxn) matrix where columns are times and row are locations
		dmd_data - (mxn) reconstruction matrix of data via DMD
		locations - standard lat and longs
		time - integer for time interval you want to plot (n)
		speed - input into the animation object for frame update speed
		'''

		# set the first things to plot
		lat = locations[:,0]
		lon = locations[:,1]
		price = data[:,0]
		dmd = dmd_data[:,0]
		error = error_data[:,0]

		# define the colorbar maxs and mins
		max_price = 60
		min_price = 10
		max_error = 10
		min_error = 0

		# create the figure
		fig, ax = plt.subplots(3, sharex=True)

		# plot the original data, dmd, and error
		data_plot = ax[0].scatter(lon,lat,alpha = 0.5,c=price, cmap = cm.jet, s=8\
					,vmin = min_price, vmax = max_price)
		plt.colorbar(data_plot,label = 'Price ($)',ax=ax[0])

		dmd_plot = ax[1].scatter(lon,lat,alpha = 0.5,c=dmd, cmap = cm.jet, s=8\
					,vmin = min_price, vmax = max_price)
		plt.colorbar(dmd_plot,label = 'Price ($)',ax=ax[1])

		error_plot = ax[1].scatter(lon,lat,alpha = 0.5,c=error, cmap = cm.jet, s=8\
					,vmin = min_error, vmax = max_error)
		plt.colorbar(error_plot,label = 'Price ($)',ax=ax[2])

		# plot the map for each subplot
		map_img = mpimg.imread('map.png')
		for i in range(3):
			ax[i].axis([-126.30, -104, 32.45, 46.73])
			ax[i].imshow(map_img, extent=[-126.30, -104, 32.45, 46.73])
			ax[i].axis("off")

		# create labels for each plot
		labels = ['Prices\n','DMD\n','Error\n']
		
		# this is the function that will be updated for each animation interation
		def update(t):
			# get new data
			price = data[:,t]
			dmd = dmd_data[:,t]
			error = error_data[:,t]

			# make a new title
			for i in range(3):
				title = 'Time '+str(t)
				ax[i].set_title(labels[i]+title)

			# update the data
			data_plot.set_array(price)
			dmd_plot.set_array(dmd)
			error_plot.set_array(error)

		animation = FuncAnimation(fig, update, interval = speed, frames = time)
		plt.show()
		return animation

if __name__ == '__main__':
	data = energy.imp_prices('prices.csv')
	locations = energy.imp_locations('coordinates.csv')
	price_ani = cal_plot.animate_price(data,locations,time = 336)
	 

	t = np.arange(int(np.size(data[0])))
	results = dmd.decomp(data,t,verbose = False,num_svd=330,svd_cut=False,esp = 2e-1)

	dmd_ani = cal_plot.animate_price(np.real(results.Xdmd),locations,time = 336,label = 'DMD')

	error_matrix = np.abs(results.Xdmd - data)
	error_ani = cal_plot.animate_error(error_matrix,locations,time = 336)

	# all_ani = cal_plot.animate_all(error_matrix,data,np.real(results.Xdmd),locations,time = 336)

	plt.show()

