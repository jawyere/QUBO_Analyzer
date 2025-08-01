import pandas as pd
from .CovarianceModel import CovarianceModel

class Covariance(CovarianceModel):

    def get_covariance_matrix(self, stock_df):
        return stock_df.cov().values