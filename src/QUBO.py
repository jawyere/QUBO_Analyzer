import numpy as np
import pandas as pd
import dimod
from neal import SimulatedAnnealingSampler
import numpy as np
from .returns.MeanReturn import MeanReturn
from .returns.ReturnModel import ReturnModel
from .covariance.CovarianceModel import CovarianceModel
from .covariance.Covariance import Covariance

class QUBO:

    """
    Declares an uninitialized QUBO with all values set to 0:

    Parameters:
        stock_data - Pandas df with date rows and ticker columns
        max_stocks_per_ticker - max amount of stocks that can be bought in a single company
    """
    def __init__(self, stock_data, bits_per_ticker, theta, M_initial, returns_method: ReturnModel, covar: CovarianceModel,  R = 0):

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

        #creates covariance matrix into numpy form
        self.cov_matrix = covar.get_covariance_matrix(stock_data)

        #creates a matrix that maps KN-dim input vector to N-dim asset weight vector
        P_matrix = self._create_P_matrix()

        return_series = returns_method.get_return(stock_data)
        
        self.return_vector = return_series.values.reshape(-1,1)
        


        #Calculate Qubo Matrix Based on Paper Definition
        #Creation of the quadratic term of the qubo has both a risk part and a return related part
        #since qubo is for minimization, we are minimizing risk while maximizing returns (minimizing -returns)
        risk = theta * P_matrix.T @ self.cov_matrix @ P_matrix
        returns = M_initial * P_matrix.T @ self.return_vector @ self.return_vector.T @ P_matrix
        quadratic = risk - returns

        linear = 2 * self.M_initial * self.R * P_matrix.T @ self.return_vector

        if len(linear) != quadratic.shape[0] or quadratic.shape[0] !=  quadratic.shape[1]:
            raise ValueError("linear dimension does not match qubo or quadratic is not square matrix")
        
        #Initialize qubo
        self.qubo = quadratic
        for i in range(len(linear)):
            self.qubo[i][i] += linear[i][0]
   

    def solve(self, to_num = True):

        Q = {(i,j): self.qubo[i][j] 
            for i in range(self.qubo.shape[0])
            for j in range(i,self.qubo.shape[1])
            }
        
        bqm = dimod.BinaryQuadraticModel.from_qubo(Q)

        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm, num_reads = 100)

        best_sample = sampleset.first.sample
        best_energy = sampleset.first.energy


        
        #iterates through array of bits and forms a dictionary of ticker_names -> number stocks
        if(to_num):
            out = {}
            n = self.num_tickers
            k = self.bits_per_ticker

            for i in range(n):

                value = 0
                for j in range(k):
                    #if the current bit is on, add its value
                    if best_sample[i*k + j] == 1:
                        value += 2**(k-j-1)

                out[self.tickers[i]] = value
            
            return out
                
        
        #return optimal list of bits
        else:
            out = [int(best_sample[num]) for num in sorted(best_sample)]
            return out

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


a = pd.DataFrame({"stock1":[1,2,3,4,5,6], "stock2":[4,5,6,5,7,9], "stock3":[100, 50, 20, 8, 5, 2], "stock4":[100,120,130,135,137,139]})
ex = pd.DataFrame({
    "stockA": [14, 14, 14, 14.2, 14.3, 14.5],
    "stockB": [20, 18, 19, 21, 22, 24],
    "stockC": [30, 31, 30, 29, 30, 30],
    "stockD": [5, 5.5, 6, 6.2, 6.5, 7]
})
ex2 = pd.DataFrame({"stock1":[1,2,3,4,5,6], "stock2":[6,5,4,3,2,1], "stock3":[1,2,3,4,5, 6]})
np.set_printoptions(linewidth=200)

method = MeanReturn()
covariance = Covariance()

b = QUBO(ex2,40, .01, 10, method, covariance)
d = b.solve()
print(b.return_vector)

print(d)

