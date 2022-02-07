# %%
import sys
sys.path.append('../src')
import pickle
import matplotlib.pyplot as plt
from Models import *
# %%
V5 = pickle.load(open('../data/V5.p','rb'))
cp_list = V5["cp_list"]
active_L_table_slide_DOA = V5["active_L_table_slide_DOA"]
active_L_table_slide_matrix = V5["active_L_table_slide_matrix"]
active_long_table_slide_DOA = V5["active_long_table_slide_DOA"]
active_long_table_slide_matrix = V5["active_long_table_slide_matrix"]

control_points = cp_list[:4]
coordinates = [np.array([i[0], i[1]]) for i in ROOM_COORDINATES[:4]]
X = np.vstack([c for c in cp_list[:4]])
y = np.vstack([np.full([p.shape[0], len(c)], c) for p, c in zip(control_points, coordinates)])
# %%
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# %%
fig, axes = plt.subplots(2, 2)
axes = axes.flatten()
# %%
models = {'linear regression': LinearRegression, 'svr': SVR, 'decision tree': DecisionTreeRegressor, 'random forest': RandomForestRegressor}
for m, ax in zip(models.items(), axes):
	regr = MultiOutputRegressor(m[1]())
	regr.fit(X, y)
	ax.scatter(*regr.predict(X).T)
	ax.scatter(*regr.predict(active_L_table_slide_DOA).T, label=m[0])
# %%
fig