import numpy as np
import subprocess
import os
from sklearn.metrics import mean_squared_error

from config_utils import multi_config_to_params_dict
from gen_data import gen_multi_group_data

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

from rpy2.robjects.functions import SignatureTranslatedFunction
STM = SignatureTranslatedFunction
fuser = importr('fuser')
fuser.fusedLassoProximal = STM(fuser.fusedLassoProximal, init_prm_translate={'lambda': 'lambda_'})
robjects.r('Sys.setenv(LANG = "en")')


class JointLasso:

    def __init__(self, proj_home_path=''):
        self.data_path = os.path.join(proj_home_path, 'data/r_scratch')
        self.proj_home_path = proj_home_path
        self.estimated_betas = None

    def fit(self, x, y, z, gamma, lamb=0.0, iters=2000, l1=True):
        nr, nc = x.shape
        xvec = robjects.FloatVector(x.transpose().reshape((x.size)))
        xr = robjects.r.matrix(xvec, nrow=nr, ncol=nc)
        yr = robjects.FloatVector(y)
        zr = robjects.FloatVector(z + 1)

        k = len(set(z))
        G = np.ones((k, k))
        gvec = robjects.FloatVector(G.transpose().reshape((G.size)))
        gr = robjects.r.matrix(gvec, nrow=k, ncol=k)

        coefs = fuser.fusedLassoProximal(xr, yr, zr, gamma=gamma, G=gr, tol=1e-6, num_it=iters,
                                         intercept=False, scaling=True, lambda_=lamb)

        self.estimated_betas = np.asarray(coefs)

    def predict(self, X, z):
        y_pred = np.zeros(len(z))
        for z_i in np.arange(len(set(z))):
            y_pred[z == z_i] = np.dot(X[z == z_i], self.estimated_betas[:, z_i])
        return y_pred


if __name__ == '__main__':

    jl = JointLasso(proj_home_path='../')
    # X, y, z, perm = gen_data.gen_multi_group_data()
    print('hi')

    params_file = 'params.ini'
    gen_data_params, model_params, results_params = multi_config_to_params_dict(params_file, config_dir='../config/')
    X, X_test, y, y_test, z, z_test, zero_coefs = gen_multi_group_data(gen_data_params,
                                                                       test_group_sizes=1000, seed=0)

    # jl.fit(X, y, z, gamma=1e-8)
    # y_pred = jl.predict(X, z)
    # print(mean_squared_error(y, y_pred))
    # minority_z = max(z)
    # print(mean_squared_error(y[z == 1], y_pred[z == 1]))
    # print(mean_squared_error(y[z == 0], y_pred[z == 0]))

    jl.fit(X, y, z, gamma=0.1, lamb=0.1, iters=3000)
    y_pred = jl.predict(X, z)
    print(mean_squared_error(y, y_pred))
    minority_z = max(z)
    print(mean_squared_error(y[z == 1], y_pred[z == 1]))
    print(mean_squared_error(y[z == 0], y_pred[z == 0]))


