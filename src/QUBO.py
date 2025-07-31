import numpy as np
import pandas as pd
import dimod
from neal import SimulatedAnnealingSampler
import numpy as np
import 

#binary vector ex: (0, 1, 1), most sig bit on right
def get_val_from_binaries(bin_vec):

    length = len(bin_vec)
    granularity = 1.0 / 2**(length)
    sum = 0

    for k in range(1,length + 1):
        sum += granularity * 2**(k-1) * bin_vec[k-1]


class QUBO:

    """
    Declares an uninitialized QUBO with all values set to 0:

    Parameters:
        stock_data - Pandas df with date rows and ticker columns
        max_stocks_per_ticker - max amount of stocks that can be bought in a single company
    """
    def __init__(self, stock_data, bits_per_ticker, theta, M_initial):

        tickers = stock_data.columns
        num_tickers = len(tickers)
        # bits_per_ticker = np.ceil(np.log2(max_stocks_per_ticker + 1.0))
        qubo_dim = int(bits_per_ticker * num_tickers)
        qubo = np.zeros((qubo_dim, qubo_dim))


        cov_matrix = stock_data.cov()
        cov_dim = len(cov_matrix)

        #creates a matrix that maps KN-dim input vector to N-dim asset weight vector
        #P_matrix = _create_P_matrix()

        return_vector = None

        print()
       

                    


       
    
    
    """
    Initializes values of QUBO correctly:
    
    Parameters:
    
   """
    
    




    def solve(self):
        pass

    


a = pd.DataFrame({"stock1":[1,2,3,4,5,6], "stock2":[4,5,6,5,7,9], "stock3":[1,2,3,3,2,1], "stock4":[6,5,4,3,2,1]})

QUBO(a,1, 1, 1)



# print(df,"\n", df.cov())