# modules
from BraTS2017.load import *
from tensorly.regression.kruskal_regression import KruskalRegressor
import pandas as pd

# Paths
StartDir = '/Volumes/Data-Julio'
ListDir = './data/survival_data.csv'
rows = 240
cols = 240
cuts = 155

# list
Patients = get_patient_list(ListDir)
num_patients = len(Patients)

# Preallocate
Xtensor_flair = np.zeros((num_patients,rows,cols,cuts))
Xtensor_t1 = np.zeros((num_patients,rows,cols,cuts))
Xtensor_t2 = np.zeros((num_patients,rows,cols,cuts))

for idx, patient in enumerate(Patients):
    I = load_data(StartDir, Patients[0], element='flair')
    Xtensor_flair[idx,...] = I


# KruskalRegressor
Regressor= KruskalRegressor(2, tol=1e-06, reg_W=1, n_iter_max=100, random_state= 99, verbose=1)
Regressor.fit(Xtensor_flair, y)
