# %%
import sys
sys.path.append('./Util')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Mapping import *
from PCA import *
from nn import model
import pickle
import itertools
import random
%load_ext autoreload
%autoreload 2
# %%
V5 = pickle.load(open('V5.p','rb'))
cp_list = V5["cp_list"]
active_L_table_slide_DOA = V5["active_L_table_slide_DOA"]
active_L_table_slide_matrix = V5["active_L_table_slide_matrix"]
active_long_table_slide_DOA = V5["active_long_table_slide_DOA"]
active_long_table_slide_matrix = V5["active_long_table_slide_matrix"]
# %%
DOA_LIST = cp_list
ROOM_COORDINATES = ROOM_COORDINATES
TABLE_CP_IND = [0,1,2,3,4,5]
CHAIR_CP_IND = [6,7,8,9,10]
ALL_CP_IND   = [0,1,2,3,4,5,6,7,8,9,10]
L_TABLE_CP_IND = [0,1,2,3]
LONG_TABLE_CP_IND = [4,5]
DATA_IND = [TABLE_CP_IND,CHAIR_CP_IND,ALL_CP_IND]
# %%
# use cp1 to calculate displacement for L table slide
R_1 = ROOM_COORDINATES[0,:2].T.reshape(-1,1)
D_1 = np.median(DOA_LIST[0], axis=0).reshape(-1,1)

# use cp6 to calculate displacement for long table slide
R_6 = ROOM_COORDINATES[5,:2].T.reshape(-1,1)
D_6 = np.median(DOA_LIST[5], axis=0).reshape(-1,1)

R_LIST = [R_1, R_6]
D_LIST = [D_1, D_6]
# %%
EVENT_DOA = [active_L_table_slide_DOA, active_long_table_slide_DOA]
EVENT_DOA = [active_L_table_slide_DOA]
EVENT_LABEL = ['L Table Slide', 'Long Table slide']
B_MATRIX_NAME = ['Table','Chair','All']
COLOR_LIST = ['r','b','g']
MARKER_LIST = ["$1$","$2$","$3$","$4$","$5$","$6$","$7$","$8$","$9$","$10$","$11$"]
# %%
# %%
def generate_points():
    mapped_points = []
    for ii in range(len(EVENT_DOA)):
        for jj in range(len(DATA_IND)):
            DOA_points = [DOA_LIST[IND] for IND in DATA_IND[jj]]
            room_coordinates = ROOM_COORDINATES[DATA_IND[jj],:]
            B = generate_linear_transform_matrix(DOA_points, room_coordinates, 2)[0]
            R_0 = R_LIST[ii]-B @ D_LIST[ii]
            r = R_0 +B @ EVENT_DOA[ii].T
            mapped_points.append(np.array([r[0,:], r[1,:]]))
    return np.hstack(mapped_points).T
# %%
fig = plt.figure(figsize = [16,12])
plt.rcParams['font.size'] = '16'
ax = fig.add_subplot(1,1,1)

for ii in range(len(EVENT_DOA)):
    for jj in range(len(DATA_IND)):
        DOA_points = [DOA_LIST[IND] for IND in DATA_IND[jj]]
        room_coordinates = ROOM_COORDINATES[DATA_IND[jj],:]
        B,R_mean,D_mean,D = generate_linear_transform_matrix(DOA_points, room_coordinates, 2) 
        R_0 = R_LIST[ii]-B @ D_LIST[ii]
        r = R_0 +B @ EVENT_DOA[ii].T
        ax.scatter(r[0,:], r[1,:], c=COLOR_LIST[jj], s=2, label=B_MATRIX_NAME[jj])    

# rect_side_table = matplotlib.patches.Rectangle((0,1.71), 0.92, (3.54-1.71), alpha = 0.3, color = '0.7')
rect_main_table_1 = matplotlib.patches.Rectangle((2.08,1.81), (4.4-0.2-2.08), (2.57-1.81), alpha = 0.3, color = '0.7')
ax.add_patch(rect_main_table_1)
ax.set_xlabel("X (m)", fontsize = 21)
ax.set_ylabel("Y (m)", fontsize = 21)
ax.set_aspect('equal')
ax.set(xlim=(0,4.385), ylim=(0,3.918))
ax.set(xlim=(0,4.385), ylim=(1.4,3.65))#ylim=(1.4,3.918))
plt.xticks([0, 1, 2, 3, 4])  
plt.yticks([1.5, 2,2.5, 3, 3.5])  
ax.scatter(ROOM_COORDINATES[:,0],ROOM_COORDINATES[:,1], c='k', s=30)
ax.tick_params(axis='y', labelsize = 21, width = 2, length = 8)
ax.tick_params(axis='x',labelsize = 21, width = 2, length = 8)

for kk in range(ROOM_COORDINATES.shape[0]):
    ax.scatter(ROOM_COORDINATES[kk,0]+0.2, ROOM_COORDINATES[kk,1], marker=MARKER_LIST[kk], s=200, c='k')
ax.legend(markerscale=5,fontsize=15)
plt.show()
# fig.savefig('Mappingtables.pdf', bbox_inches='tight', pad_inches=0)
# %%
def filter_coords(mappings, table_coords, tol=0.1):
    w1, w2 = set(table_coords[:, 0])
    h1, h2 = set(table_coords[:, 1])

    filtered_coords = []
    for coord in mappings:
        x, y = list(coord)
        if np.isclose(x, w1, rtol=0, atol=tol) or np.isclose(x, w2, rtol=0, atol=tol):
            if y >= (h1 - tol) and y <= (h2 + tol):
                filtered_coords.append(coord)
        elif np.isclose(y, h1, rtol=0, atol=tol) or np.isclose(y, h2, rtol=0, atol=tol):
            if x >= (w1 - tol) and x <= (w2 + tol):
                filtered_coords.append(coord)
    return np.array(filtered_coords)
# %%
mapped_points = generate_points()
table_coords = np.array([room[:2] for room in ROOM_COORDINATES[L_TABLE_CP_IND]])
filtered = filter_coords(mapped_points, table_coords, tol=0.10)
r = filtered.T
# r = mapped_points.T

fig = plt.figure(figsize = [16,12])
plt.rcParams['font.size'] = '16'
ax = fig.add_subplot(1,1,1)
ax.scatter(r[0,:], r[1,:], c='b', s=2, label=B_MATRIX_NAME[jj])    
rect_main_table_1 = matplotlib.patches.Rectangle((2.08,1.81), (4.4-0.2-2.08), (2.57-1.81), alpha = 0.3, color = '0.7')
ax.add_patch(rect_main_table_1)
ax.set_xlabel("X (m)", fontsize = 21)
ax.set_ylabel("Y (m)", fontsize = 21)
ax.set_aspect('equal')
ax.set(xlim=(0,4.385), ylim=(0,3.918))
ax.set(xlim=(0,4.385), ylim=(1.4,3.65))#ylim=(1.4,3.918))
plt.xticks([0, 1, 2, 3, 4])  
plt.yticks([1.5, 2,2.5, 3, 3.5])  
ax.scatter(ROOM_COORDINATES[:,0],ROOM_COORDINATES[:,1], c='k', s=30)
ax.tick_params(axis='y', labelsize = 21, width = 2, length = 8)
ax.tick_params(axis='x',labelsize = 21, width = 2, length = 8)

for kk in range(ROOM_COORDINATES.shape[0]):
    ax.scatter(ROOM_COORDINATES[kk,0]+0.2, ROOM_COORDINATES[kk,1], marker=MARKER_LIST[kk], s=200, c='k')
ax.legend(markerscale=5,fontsize=15)
plt.show()
print(f"lenght: {filtered.shape[0]}")
# %%
