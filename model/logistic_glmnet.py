
import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score



import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.functions import SignatureTranslatedFunction

base = importr('base')
glmnet = importr('glmnet')
stats = importr('stats')
STM = SignatureTranslatedFunction
glmnet.cv_glmnet = STM(glmnet.cv_glmnet, init_prm_translate={'lambda': 'lambda_'})
robjects.r('Sys.setenv(LANG = "en")')


class FairElasticGlmNet():

    def __init__(self, proj_home_path=''):
        self.proj_home_path = proj_home_path
        self.estimated_betas = None
        self.lambda_min = None

    def fit(self, x, y, curr_n_k=None, penalty_factor=None, logistic=False, lambdas=None, l1=True, alpha=0.5):
        nr, nc = x.shape
        xvec = robjects.FloatVector(x.transpose().reshape((x.size)))
        xr = robjects.r.matrix(xvec, nrow=nr, ncol=nc)
        yr = robjects.FloatVector(y)
        # family will be binomial
        family = 'binomial' if logistic else 'gaussian'
        type_measure = 'auc' if logistic else 'mse'

        curr_n_k = curr_n_k if curr_n_k is not None else np.ones(nr)
        curr_n_k_r = robjects.FloatVector(curr_n_k)
        penalty_factor = penalty_factor if penalty_factor is not None else np.ones(nc)
        penalty_factor_r = robjects.FloatVector(penalty_factor)

        if lambdas is None:
            # xr: This is the input feature matrix. It should be a 2D array where each row represents an observation and each column represents a feature.
            # yr: This is the response vector. It should be a 1D array where each element corresponds to the response variable for the respective observation in xr.
            # alpha: This parameter controls the elastic net mixing parameter. It ranges from 0 to 1:
                # alpha=1 corresponds to Lasso regression (L1 penalty).
                # alpha=0 corresponds to Ridge regression (L2 penalty).
                # Values between 0 and 1 provide a mix of L1 and L2 penalties.
            # weights: This is an array of observation weights. It allows you to give different weights to different observations, which can be useful if some observations are more important or reliable than others.

            # penalty_factor: This is an array that applies different penalty factors to different coefficients. It allows you to penalize some coefficients more than others.
           
            # intercept: This boolean parameter indicates whether to fit an intercept term. If False, no intercept will be included in the model.

            # family: This specifies the type of model to be fit. Common values include:
                # 'gaussian' for linear regression.
                # 'binomial' for logistic regression.
            # Other values for different types of generalized linear models.
            # type_measure: This specifies the type of measure to use for cross-validation. Common values include:
                # 'mse' for mean squared error.
                # 'mae' for mean absolute error.
                # 'deviance' for deviance.
            # nfolds: This specifies the number of folds to use in cross-validation. A typical value is 4, but it can be adjusted based on the size of your dataset and the desired robustness of the cross-validation.
            r = robjects.r
            print("xr shape:", tuple(r['dim'](xr)))
            print("curr_n_k shape:", len(curr_n_k_r))
            print("yr shape:", len(yr))
            print("penalty_factor_r:", len(penalty_factor_r))

            #offset1 = robjects.FloatVector(np.ones(x.shape[0]))
            
            
            res = glmnet.cv_glmnet(xr, yr, alpha=alpha, weights=curr_n_k_r, penalty_factor=penalty_factor_r,
                                   intercept=False, family=family, type_measure=type_measure, nfolds=20)
        else:
            lambdas_r = robjects.FloatVector(lambdas)
            res = glmnet.cv_glmnet(xr, yr, alpha=alpha, weights=curr_n_k_r, penalty_factor=penalty_factor_r,
                                   intercept=False, family=family, type_measure=type_measure, nfolds=20, lambda_=lambdas_r)

        coef = np.asarray(base.as_matrix(stats.coef(res, s="lambda.min")))
        self.estimated_betas = coef.reshape(coef.size)[1:]
        self.lambda_min = res[res.names.index('lambda.min')][0]


    def predict(self, x):
        if self.estimated_betas is None:
            raise ValueError("Model is not fitted yet. Please call the fit method first.")
        
        linear_predictor = np.dot(x, self.estimated_betas)
        probabilities = 1 / (1 + np.exp(-linear_predictor))
        return (probabilities >= 0.5).astype(int)
    
    def predict_proba(self, x):
        if self.estimated_betas is None:
            raise ValueError("Model is not fitted yet. Please call the fit method first.")  
        
        linear_predictor = np.dot(x, self.estimated_betas)
        probabilities = 1 / (1 + np.exp(-linear_predictor))
        return probabilities


