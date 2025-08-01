from abc import ABC, abstractmethod


class ReturnModel(ABC):
    
    @abstractmethod
    def get_return(stock_data_frame, throw_at = 1):
        pass