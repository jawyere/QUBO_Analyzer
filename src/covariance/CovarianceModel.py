from abc import ABC, abstractmethod

class CovarianceModel(ABC):

    @abstractmethod
    def get_covariance_matrix(self, stock_df):
        pass