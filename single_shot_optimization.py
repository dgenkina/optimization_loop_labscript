from __future__ import division
import numpy as np
<<<<<<< HEAD
import h5py
import matplotlib.pyplot as plt
import lyse
#import gradient_descent_functions as optimization_protocol
import pandas as pd
=======

import matplotlib.pyplot as plt
import lyse
import h5py
#import gradient_descent_functions as optimization_protocol
import pandas as pd
from scipy import optimize
>>>>>>> e635434002d746ebd0fcaf5776a459bd20c57355
#from optimization_classes import *

camera = 'imagesXZ0_Flea3'
orientation0 = 'xy0'
orientation1 = 'xy'
orientationXZ = 'xz'

#camera = 'imagesFlea3'
<<<<<<< HEAD

try:
    lyse.path
except NameError:
    import sys
    path = sys.argv[1]
=======
#try:
#   lyse.path
#except NameError:
#    import sys
#    path = sys.argv[1]
path = lyse.path
if path is None:
    path = "C:/Users/rubidium/Downloads/2018_11_19_20180906_quadTrap_toOptimize_00.h5"
>>>>>>> e635434002d746ebd0fcaf5776a459bd20c57355

with h5py.File(lyse.path) as h5_file:
    camera_name = camera 
    #atoms, probe, dark = dummy_h5_label['data/'+camera_name]['Raw'][:]
    atoms = np.float64(h5_file['images/' + orientationXZ +'/shot1/']['atoms'])
    probe = np.float64(h5_file['images/' + orientationXZ +'/shot2/']['probe'])
    dark = np.float64(h5_file['images/' + orientationXZ +'/shot3/']['background'])




<<<<<<< HEAD
# with h5py.File(lyse.path) as dummy_h5_label:
#     camera_name = camera 
#     atoms, probe, dark = dummy_h5_label['data/'+camera_name]['Raw'][:]
=======
#with h5py.File(path) as dummy_h5_label:
#    camera_name = camera 
#    atoms, probe, dark = dummy_h5_label['data/'+camera_name]['Raw'][:]
>>>>>>> e635434002d746ebd0fcaf5776a459bd20c57355

df = lyse.data()
#df = df.sort_values('run time')
# dummy1 = df['dummy1'][-1]
# dummy2 = df['dummy2'][-1]

xmin = 0
xmax = 400
ymin = 0
ymax = 480


absorbed_fraction = (atoms - dark)/(probe - dark)
naive_OD = -np.log(absorbed_fraction)
negative_ind = [absorbed_fraction <= 0.0]
naive_OD[negative_ind] = 0.0


_fig_ = plt.figure(frameon=True)

ax1= plt.subplot(231)
im1 = ax1.imshow(atoms, interpolation='None',aspect='equal',cmap = 'jet')

ax2 = plt.subplot(232)
im2 = ax2.imshow(probe, interpolation='None',aspect='equal',cmap = 'jet')

ax3 = plt.subplot(233)
im3 = ax3.imshow(dark, interpolation='None',aspect='equal',cmap = 'jet')
plt.colorbar(im3)

ax4 = plt.subplot(234)
im4 = ax4.imshow(naive_OD, vmin=-0.1, vmax=1.5, interpolation='None',aspect='equal',cmap = 'jet')
plt.colorbar(im4)


OD_ROI = naive_OD[ymin:ymax, xmin:xmax]
<<<<<<< HEAD
print('OD_ROI shape')
print(OD_ROI.shape)
=======
>>>>>>> e635434002d746ebd0fcaf5776a459bd20c57355
ax5 = plt.subplot(235)
im5 = ax5.imshow(OD_ROI, vmin=-0.1, vmax=1.5, interpolation='None',aspect='equal',cmap = 'jet')
plt.colorbar(im5)

<<<<<<< HEAD

run_instance = lyse.Run(lyse.path)
print(lyse.path)
=======
run_instance = lyse.Run(path)
print(path)

def gaussian(x_list, x0, A, sigma, offset):
    y_list = A*np.exp(-(x_list-x0)**2.0/(2.0*sigma**2.0)) + offset
    return y_list

>>>>>>> e635434002d746ebd0fcaf5776a459bd20c57355

sigma0 = (3/2/np.pi) * 780e-9**2


atom_number = np.nansum(OD_ROI*(5.6e-6/6.0)**2/sigma0)
<<<<<<< HEAD
# x = np.array([dummy1,dummy2])
fitness = -atom_number
=======

OD_3 = np.ones([112, 74])*3
atom3 = np.nansum(OD_3*(5.6e-6/6.0)**2/sigma0)

fig = plt.figure()

max_ind = np.unravel_index(np.argmax(naive_OD), naive_OD.shape)
print(max_ind)

y_data = plt.subplot(121)
y_slice = naive_OD[:, 315]
#y_slice = naive_OD[:, max_ind[1]]
y_data.plot(y_slice, '.')
Y = np.arange(y_slice.size)
y = np.sum(Y*y_slice)/np.sum(y_slice)
y_width = np.sqrt(np.abs(np.sum((Y-y)**2*y_slice)/np.sum(y_slice)))
y_max = y_slice.max()
y_max_ind = np.where(y_slice == y_max)[0][0]
popt, pcov = popt, pcov = optimize.curve_fit(gaussian, Y, y_slice, p0=(y_max_ind, y_max,y_width,0.0))
y_data.plot(Y, gaussian(Y, *popt), '-')
y_stdev = popt[2]

x_data = plt.subplot(122)
x_slice = naive_OD[290, :]
#x_slice = naive_OD[max_ind[0], :]
x_data.plot(x_slice, '.')
X = np.arange(x_slice.size)
x = np.sum(X*x_slice)/np.sum(x_slice)
x_width = np.sqrt(np.abs(np.sum((X-x)**2*x_slice)/np.sum(x_slice)))
x_max = x_slice.max()
x_max_ind = np.where(x_slice==x_max)[0][0]
popt, pcov = optimize.curve_fit(gaussian, X, x_slice, p0=(x_max_ind, x_max,x_width,0.0))
x_data.plot(X,gaussian(X,*popt), '-')
x_stdev = popt[2]
plt.show
#print(y_stdev, x_stdev)

# x = np.array([dummy1,dummy2])

#fitness = -atom_number + 1E4 * ((y_stdev - 80) + (x_stdev - 75))

def fitness_funct(y_stdev, x_stdev, atom_number): 
    fitness = -atom_number
    if y_stdev >= 85:
        fitness += 5E3 * (y_stdev - 80) 
    if x_stdev >= 80:
        fitness += 5E3 * (x_stdev - 75)
    return fitness

fitness = fitness_funct(y_stdev, x_stdev, atom_number)
>>>>>>> e635434002d746ebd0fcaf5776a459bd20c57355


# run_instance.save_result('average_counts', atom_avg)
run_instance.save_result('fitness', fitness)

print('Saved Cost value of ' + str(fitness))






	