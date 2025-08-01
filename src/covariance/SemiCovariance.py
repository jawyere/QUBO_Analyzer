import pandas as pd
from .CovarianceModel import CovarianceModel

class SemiCovariance(CovarianceModel):
    def get_covariance_matrix(self, stock_df):
        