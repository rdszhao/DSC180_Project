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
from sklearn.svm import SVR
# %%
regr = MultiOutputRegressor(SVR(kernel='rbf'))
regr.fit(X, y)
# %%
plt.scatter(*regr.predict(X).T)
plt.scatter(*regr.predict(active_L_table_slide_DOA).T)
# %%
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform
# %%
param_distributions = {'estimator__C': uniform(1, 10), 'estimator__gamma': reciprocal(0.001, 0.1)}
rscv = RandomizedSearchCV(regr, param_distributions, n_iter=10)
rscv.fit(X, y)
# %%
plt.scatter(*rscv.predict(X).T)
plt.scatter(*rscv.predict(active_L_table_slide_DOA).T)