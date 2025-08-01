import numpy as np
import pandas as pd
import dimod
from neal import SimulatedAnnealingSampler
import numpy as np
from .Returns.MeanReturn import MeanReturn
from .Returns.ReturnModel import ReturnModel


class QUBO:

    """
    Declares an uninitialized QUBO with all values set to 0:

    Parameters:
        stock_data - Pandas df with date rows and ticker columns
        max_stocks_per_ticker - max amount of stocks that can be bought in a single company
    """
    def __init__(self, stock_data, bits_per_ticker, theta, M_initial, R, returns_method: ReturnModel):

        self.tickers = stock_data.columns
        self.bits_per_ticker = bits_per_ticker
        self.num_tickers = len(self.tickers)
        self.qubo_dim = int(bits_per_ticker * self.num_tickers)
        self.qubo = np.zeros((self.qubo_dim, self.qubo_dim))

        """
        R - desired Return
        M - return coefficient
        theta - risk coefficient
        """
        self.R = R
        self.M_initial = M_initial
        self.theta = theta


        self.cov_matrix = stock_data.cov().values

        #creates a matrix that maps KN-dim input vector to N-dim asset weight vector
        P_matrix = self._create_P_matrix()

        return_series = returns_method.get_return(stock_data)
        self.return_vector = return_series.values.reshape(-1,1)
        print(self.return_vector)

       

        #Calculate Qubo Matrix Based on Paper Definition
        #Creation of the quadratic term of the qubo has both a risk part and a return related part
        #since qubo is for minimization, we are minimizing risk while maximizing returns (minimizing -returns)
        risk = theta * P_matrix.T @ self.cov_matrix @ P_matrix
        returns = M_initial * P_matrix.T @ self.return_vector @ self.return_vector.T @ P_matrix
        quadratic = risk - returns

        linear = 2 * self.R * P_matrix.T @ self.return_vector
        

        


       
    
    
    """
    Initializes values of QUBO correctly:
    
    Parameters:
    
   """
    
    




    def solve(self):
        pass

    #creates a matrix that maps kn-dim input vector to n-dim asset weight vector
    def _create_P_matrix(self):

        n = self.num_tickers
        k = self.bits_per_ticker
        P = np.zeros((n, k*n))

        start = 0
        for i in range(n):
            for j in range(k):
                
                P[i][j + start] = .5**(j+1)
            start += k



        return P


a = pd.DataFrame({"stock1":[1,2,3,4,5,6], "stock2":[4,5,6,5,7,9], "stock3":[1,2,3,3,2,1], "stock4":[6,5,4,3,2,1]})

np.set_printoptions(linewidth=200)

method = MeanReturn()
b = QUBO(a,4, 1, 1, 1, method)

print(b._create_P_matrix())

