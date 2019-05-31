from __future__ import division
import numpy as np
import h5py
import matplotlib.pyplot as plt
import lyse
#import gradient_descent_functions as optimization_protocol
import pandas as pd
#from optimization_classes import *

camera = 'imagesXZ0_Flea3'
orientation0 = 'xy0'
orientation1 = 'xy'
orientationXZ = 'xz'

#camera = 'imagesFlea3'

try:
    lyse.path
except NameError:
    import sys
    path = sys.argv[1]

with h5py.File(lyse.path) as h5_file:
    camera_name = camera 
    #atoms, probe, dark = dummy_h5_label['data/'+camera_name]['Raw'][:]
    atoms = np.float64(h5_file['images/' + orientationXZ +'/shot1/']['atoms'])
    probe = np.float64(h5_file['images/' + orientationXZ +'/shot2/']['probe'])
    dark = np.float64(h5_file['images/' + orientationXZ +'/shot3/']['background'])




# with h5py.File(lyse.path) as dummy_h5_label:
#     camera_name = camera 
#     atoms, probe, dark = dummy_h5_label['data/'+camera_name]['Raw'][:]

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
print('OD_ROI shape')
print(OD_ROI.shape)
ax5 = plt.subplot(235)
im5 = ax5.imshow(OD_ROI, vmin=-0.1, vmax=1.5, interpolation='None',aspect='equal',cmap = 'jet')
plt.colorbar(im5)


run_instance = lyse.Run(lyse.path)
print(lyse.path)

sigma0 = (3/2/np.pi) * 780e-9**2


atom_number = np.nansum(OD_ROI*(5.6e-6/6.0)**2/sigma0)
# x = np.array([dummy1,dummy2])
fitness = -atom_number


# run_instance.save_result('average_counts', atom_avg)
run_instance.save_result('fitness', fitness)

print('Saved Cost value of ' + str(fitness))






	