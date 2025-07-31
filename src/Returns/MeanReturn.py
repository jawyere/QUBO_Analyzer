import pandas as pd
from ReturnModel import ReturnModel

class MeanReturn(ReturnModel):

    def get_return(self, stock_df, throw_at = 1):

        #gets return_i = (price_i - price_i-1)/price_i-1
        returns_df = stock_df.astype('float64').pct_change(fill_method=None)

        # #remove initial NaN created due to Return Def
        returns_df = returns_df.drop(index = 0)

        #check that percent of invalid values < throw_at
        num_na = returns_df.isna().sum().sum()
        total_vals = returns_df.shape[0]*returns_df.shape[1]

        if num_na/float(total_vals) > throw_at:
            raise ValueError("Percent of NaN values > throw_at ratio")

        return returns_df
    


if __name__ == "__main__":

    df = pd.DataFrame({'a':[1, 2, 3], 'b':[4, 5, 6], 'c':[1, 4, 32]})
    mr = MeanReturn()
    print(mr.get_return(df, throw_at=.01))
