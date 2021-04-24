import numpy as np 
import matplotlib.pyplot as plt
import math
import seaborn as sns

class TimeSeries(object):
    def __init__(self, arr):
        '''
        @params:
            - arr: 1-D np.array.
        '''
        assert(isinstance(arr, np.ndarray) == True), "arr must be np.ndarray." 
        assert(len(arr.shape) == 1), "arr must be 1-D array."
        self._vals = arr
        
    def __len__(self):
        return self._vals.size

    def __str__(self):
        return f'TimeSeries: len: {self.__len__()} | {self.describe()}'

    def describe(self):
        min_val, max_val, mean_val, median_val = np.min(self._vals), np.max(self._vals), np.mean(self._vals), np.median(self._vals)
        return ' | '.join([f'{m}: {str(i)}' for m, i in zip(['min', 'max', 'mean', 'median'], [min_val, max_val, mean_val, median_val])])

    def get_vals(self):
        return self._vals

    def plot(self, color='g', alpha=0.4, title='time-series'):
        series_len = len(self._vals)
        x = [i for i in range(series_len)]

        sns.set_style('darkgrid')
        plt.figure()
        plt.plot(x, self._vals, c=color, alpha=alpha)
        plt.title(title)
        plt.show()

    def down_sampling(self, ref_len, ref_n_ele):
        '''
        @params:
            - ref_len: int, greater or equal to timeseries len.
            - ref_n_ele: int, number of elements described reference timeseries
        @returns:
            - series: a TimeSeries object.
        '''
        def round_up(numb):
            return math.floor(numb+0.5) 

        def round_down(numb):
            return math.floor(numb)

        arr =self.get_vals()
        n_ele = self.__len__()
        
        bound_list = np.linspace(start=0, stop=ref_len, num=ref_n_ele+1)
        new_arr = list()
        for lower_bound, upper_bound in zip(bound_list[:-1], bound_list[1:]):
            start_idx = round_up(lower_bound)
            end_idx = round_down(upper_bound) + 1 # inclusively
            if (start_idx > n_ele - 1):
                break
            
            new_arr.append(np.mean(arr[start_idx:end_idx]))
        
        new_arr = np.array(new_arr)
        return TimeSeries(new_arr)

class TimeSeriesGenerator(object):
    def __init__(self):
        super().__init__()

    def _check_input(self, pattern, n_points, start_val, step, amplitude, has_seed, seed):
        def is_number(var):
            return isinstance(start_val, (int, float))

        assert len(pattern) > 0, "pattern must be non-empty string."
        count_UDF = sum([pattern.count(c) for c in ['U', 'D', 'F']])
        assert (count_UDF == len(pattern)), "pattern must contain one of the following characters: 'U', 'D', 'F'."
        assert ((isinstance(n_points, int) == True) and (n_points > 0)), "n_points must be a positive integer."
        assert (is_number(start_val) == True), "start_val must be a number."
        assert (is_number(amplitude) == True and amplitude > 0), "amplitude must be a positive number."
        assert (is_number(step) == True and step > 0), "step must be a positive number."
        assert (isinstance(has_seed, bool) == True), "has_seed must be a boolean."
        assert (isinstance(seed, int) == True), "seed must be an integer."
        return True

    def generate_timeseries(self, pattern='FFFFUUFFFDFD', n_points=5, start_val=0, amplitude=5, step=20, has_seed=True, seed=42):
        '''
        @params:
            - pattern: str, only accepts ['U', 'D', 'F'].
            - n_points: int, how many points per character in pattern.
            - start_val: float, starting value of the timeseries.
            - amplitude: float, value randomly sampled in range [val-amplitude, val+amplitude].
            - step: difference in value when go U(p) or D(own).
            - has_seed: boolean. To determine whether to use seed or not.
            - seed: int, to reduplicate result.
        @returns:
            - series: a TimeSeries object.
        '''
        is_good = self._check_input(pattern, n_points, start_val, step, amplitude, has_seed, seed)
        if (is_good == False):
            return None

        if (has_seed == True):
            np.random.seed(seed)

        series = []
        cur_val = start_val
        
        for c in pattern:
            if (c == 'U'):
                new_val = cur_val + step*n_points
                gen_arr = np.arange(start=cur_val+step, stop=new_val+1, step=step)

            elif (c == 'D'):
                new_val = cur_val - step*n_points
                gen_arr = np.arange(start=cur_val-step, stop=new_val-1, step=-step)

            else:
                new_val = cur_val
                gen_arr = np.full(shape=n_points, fill_value=new_val)
            
            noise_arr = np.random.uniform(low=-amplitude, high=amplitude, size=n_points)                            
            sampled_arr = gen_arr + noise_arr
            cur_val = new_val
            series.append(sampled_arr)

        series = np.concatenate(series)
        return TimeSeries(series)
        
def plot_time_series_pair(s1, s2, color_s1='limegreen', color_s2='cornflowerblue', name_1='Time-series s1', name_2='Time-series s2',
                         figsize=(12, 6), alpha=0.4, title='Time series pair'):
    s1, s2 = s1.get_vals(), s2.get_vals()
    
    sns.set_style('darkgrid')

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [2, 2]}) # 2 rows x 1 column 
    ax1, ax2 = axes

    ax1.plot(np.arange(start=0, stop=s1.size, step=1), s1, '.-', c=color_s1, alpha=alpha)
    ax1.set_title(name_1)

    ax2.plot(np.arange(start=0, stop=s2.size, step=1), s2, '.-', c=color_s2, alpha=alpha)        
    ax2.set_title(name_2)

    plt.suptitle(title)

    plt.show()

if (__name__ == '__main__'):
    timeseries_gen = TimeSeriesGenerator()
    s1 = timeseries_gen.generate_timeseries(pattern='FFFFUUFFFDFD',    
                                     n_points=10, 
                                     start_val=0, 
                                     amplitude=1, 
                                     step=20, 
                                     has_seed=True, 
                                     seed=42)
    s2 = timeseries_gen.generate_timeseries(pattern='FFUUFFFDFD', 
                                     n_points=10, 
                                     start_val=0, 
                                     amplitude=1, 
                                     step=20, 
                                     has_seed=False)
    plot_time_series_pair(s1, s2)
    
    # Test for downsampling
    s2 = s1.down_sampling(ref_len=120, ref_n_ele=20)
    plot_time_series_pair(s1, s2)