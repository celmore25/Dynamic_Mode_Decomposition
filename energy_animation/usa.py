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

	def animate_price(data,locations,time = 100):
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
		

		def update(t):
			price = data[:,t]
			title = 'Time: '+str(t)
			ax.set_title(title)
			data_plot.set_array(price)

		animation = FuncAnimation(fig, update, interval=100, frames = time)
		plt.show()

if __name__ == '__main__':
	data = energy.imp_prices('prices.csv')
	locations = energy.imp_locations('coordinates.csv')
	cal_plot.animate_price(data,locations,time = 336)

