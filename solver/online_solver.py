import numpy as np


class OnlineSolver:
    def __init__(self) -> None:
        pass
    
    def initial_with_problem(self,T,Y_0,track_list):
        self.value_histories = np.zeros(T)
        for item in track_list:
            exec( 'self.'+item +' = np.zeros( (T+1,) + Y_0.shape ) ' )
        self.time = np.zeros(T)

    def calculate_aver_value(self):
        self.aver_value_histories = OnlineSolver.aver_array(self.value_histories)

    def sum_time(self):
        self.time_sum = OnlineSolver.sum_array(self.time)

    @staticmethod
    def aver_array (arr):
        aver_arr = np.copy(arr)
        T = len(arr)
        value = 0
        for t in range(T):
            value = t/(t+1) * value + 1/(t+1) * arr[t]
            aver_arr[t] = value
        return aver_arr
        
    @staticmethod
    def sum_array (arr):
        sum_arr = np.copy(arr)
        value = 0
        
        for t in range (len(arr)):
            value = value + arr[t]
            sum_arr[t] = value
        return sum_arr