import numpy as np
from time_series import TimeSeriesGenerator
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.patches import ConnectionPatch

import seaborn as sns
import os

from time_series import TimeSeries

class DynamicTimeWarping(object):
    def __init__(self, s1, s2):
        '''
        @params:
            - s1: TimeSeries object.
            - s2: TimeSeries object.
        '''
        self._s1 = s1
        self._s2 = s2

    def compute(self, max_ele):
        ''' Main method to compute DTW matrix and time series mapping.
        @params:
            - max_ele: int, maximum elements per time series can be shown in the illustration.
        @returns:
            - DTW_dist: float.
            - matrix: 2-D np.array.
            - connection_list: list of list of int, mapping between 2 time series.
            - s1, s2: TimeSeries, rescaled data.
        '''

        # only downsampling if max_ele < duration of shorter time series
        s1, s2 = self._s1, self._s2
        len_s1, len_s2 = len(s1), len(s2)
        if (max_ele < min(len_s1, len_s2)):
            # downsampling time series maximum allowable length while maintain the length ratio in the pair
            s1, s2 = self._preprocess(s1=s1, 
                                    s2=s2,
                                    max_ele=max_ele,)

        # compute DTW_dist and mapping
        DTW_dist, matrix  = self._compute_DTW_matrix_and_distance(s1, s2)
        connection_list = self._compute_backtrack(s1, s2, matrix)
        return DTW_dist, matrix, connection_list, s1, s2

    def _preprocess(self, s1, s2, max_ele=20):
        len_limit = max([len(s1), len(s2)])
        new_s1 = s1.down_sampling(ref_len=len_limit, ref_n_ele=max_ele)
        new_s2 = s2.down_sampling(ref_len=len_limit, ref_n_ele=max_ele)

        return new_s1, new_s2
    
    def _compute_DTW_matrix_and_distance(self, s1, s2):
        s1, s2 = s1.get_vals(), s2.get_vals()
        n, m = len(s1), len(s2)
        matrix = np.full((n+1, m+1), np.inf)

        matrix[0, 0] = 0
        # NOTE: matrix[i, j] implies a connection between s1[i-1] and s2[j-1]

        # We start scanning the matrix, up to bottom, left to right, 
        for i in range(n):
            for j in range(m):
                # the cost to establish existing mapping
                cost = abs(s1[i] - s2[j])

                # look back at 3 possible past actions and select one that least costly
                additional_cost = min([
                    matrix[i, j+1], # s2 moves up one position and match
                    matrix[i+1, j], # s1 moves up one position and match
                    matrix[i, j],   # match
                ])

                # update the optimal cost up till this mapping
                matrix[i+1, j+1] = cost + additional_cost

        DTW_dist = matrix[n, m]

        return DTW_dist, np.array(matrix) 

    def _compute_backtrack(self, s1, s2, matrix):
        s1, s2 = s1.get_vals(), s2.get_vals()
        n, m = len(s1), len(s2)
        
        connection_list = []

        # We trace back from bottom right cell
        while ((n > 0) and (m > 0)):
            # possible taken actions are...
            cell_vals = [
                matrix[n-1, m],   # s2 moves up one position and match
                matrix[n-1, m-1], # match
                matrix[n, m-1],   # s1 moves up one position and match
            ]
        
            next_cell_mapping = [
                [n-1, m],   # s2 moves up one position and match
                [n-1, m-1], # match
                [n, m-1],   # s1 moves up one position and match
            ]

            # Select the cell has lowest cost
            min_val_idx = np.argmin(cell_vals)

            # the selected cell's position imply a mapping established between 2 time series
            point_in_s1 = [n-1, s1[n-1]]
            point_in_s2 = [m-1, s2[m-1]]
            connection_list.append([point_in_s1, point_in_s2])
            n, m = next_cell_mapping[min_val_idx]

        return connection_list[::-1]

    def plot_matrix(self, matrix, connection_list, s1, s2, mask=None, line_ratio=1, use_log=False, figsize=(8, 8), 
                    color_s1='limegreen', color_s2='cornflowerblue', show_numb=False, is_saved=False, img_name='DTW_matrix.png'):
        ''' 
        @params:
            - matrix: 2-D np.array.
            - connection_list: list of list of int, mapping between 2 time series.
            - s1, s2: TimeSeries.
            - mask: np.array, could be [0, 1] or boolean.  
            - line_ratio: float, [0.; 1.]. How much of backtracking line to be shown.
            - use_log: boolean. Transform matrix's cell value with natural log.
            - figsize: tuple, should be a square. 
            - color_s1, color_s2: str, a seaborn color.
            - show_numb: boolean, shows cost on cell.
            - is_saved: boolean, to save the plot to file or not. 
            - img_name: str, file name. Only applicable if is_saved=True.
        '''
        s1, s2 = s1.get_vals(), s2.get_vals()
        matrix = matrix[1:, 1:]

        if (use_log == True):
            matrix = np.log(matrix)
        matrix = matrix.astype(int)

        # Set up the axes with gridspec
        fig = plt.figure(figsize=figsize, frameon=False)
        grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
        plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)

        # Plot DTW matrix and colorbar
        main_ax = fig.add_subplot(grid[1:, 1:])
        y_ax = fig.add_subplot(grid[1:, 0], xticklabels=[], sharey=main_ax)
        x_ax = fig.add_subplot(grid[0, 1:], yticklabels=[], sharex=main_ax)
        cbax_ax = fig.add_axes([0.12, 0.75, 0.05, 0.1])

        if (mask is None):
            mask = np.zeros_like(matrix)
    
        sns.heatmap(matrix, mask=mask, cmap='icefire', annot=show_numb, fmt='d', linewidth=0.5, 
                    cbar=True, cbar_ax=cbax_ax, cbar_kws={"shrink": .70},
                    xticklabels=[], yticklabels=[], ax=main_ax)

        # Plot backtracking line on top
        x = [x2+0.5 for ((x1, y1), (x2, y2)) in connection_list]
        y = [x1+0.5 for ((x1, y1), (x2, y2)) in connection_list]
        n_points = len(x)
        n_points_shown = int(n_points*(1-line_ratio))
        main_ax.plot(x[n_points_shown:], y[n_points_shown:], '-', c='darkslategray')
        main_ax.set_xlabel('s2')
        main_ax.set_ylabel('s1')
        main_ax.xaxis.set_label_position('top')


        # Plot the 2 time series
        len_s1, len_s2 = matrix.shape
        y_ax.plot(s1, [i+0.5 for i in range(len_s1)], '.-', c=color_s1, alpha=0.5)
        y_ax.invert_xaxis()
        y_ax.set_xticks([])
        y_ax.set_xticklabels([])
        
        x_ax.plot([i+0.5 for i in range(len_s2)], s2, '.-', c=color_s2, alpha=0.5)
        x_ax.set_yticks([])
        x_ax.set_yticklabels([])

        
        plt.subplots_adjust(top=0.9)
        plt.suptitle('Dynamic Time Warping matrix')

        if (is_saved == True):
            plt.savefig(img_name)
        else:
            plt.show()
        plt.close()

    def plot_series_mapping(self, s1, s2, connection_list, figsize=(8, 8), 
                    color_s1='limegreen', color_s2='cornflowerblue', is_saved=False, img_name='series_mapping.png'):
        ''' 
        @params:
            - s1, s2: TimeSeries.
            - connection_list: list of list of int, mapping between 2 time series.
            - figsize: tuple. 
            - color_s1, color_s2: str, a seaborn color.
            - show_connection_val: boolean, shows value on each connection.
        '''

        # Shift s2 upwards
        s1, s2 = s1.get_vals(), s2.get_vals()
    
        sns.set_style('darkgrid')
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [1, 2, 2], 'hspace': 0.3}) # 3 rows x 1 column 
        ax1, ax2, ax3 = axes
            
        # Plot the 2 time series...
        ax2.plot(np.arange(start=0, stop=s1.size, step=1), s1, '.-', c=color_s1, alpha=0.5, label='s1')
        ax2.xaxis.set_label_position('top')
        ax2.set_xlabel('Time-series s1')
        ax3.plot(np.arange(start=0, stop=s2.size, step=1), s2, '.-', c=color_s2, alpha=0.5, label='s2')
        ax3.set_xlabel('Time-series s2')
        
        connection_val_list = [0]*s1.size
        # and its mappings (connection)
        for point_in_s1, point_in_s2 in connection_list:
            x1, y1 = point_in_s1
            _, y2 = point_in_s2
            val = abs(y1 - y2)
            connection_val_list[x1] += val 

            con = ConnectionPatch(xyA=point_in_s1, xyB=point_in_s2, coordsA="data", coordsB="data",
                        axesA=ax2, axesB=ax3, color="black")
            ax3.add_artist(con)

        # Plot the true absolute gap at each point in s1
        bar_data = {
            'x': np.arange(start=0, stop=s1.size, step=1),
            'y': connection_val_list,
        }


        ax1 = sns.barplot(x='x', y='y', data=bar_data, color='salmon', saturation=0.75, ax=ax1)        
        ax1.tick_params(axis='x', rotation=90, width=5)
        ax1.set_xlabel('Absolute value of the connections in s1')

        fig.tight_layout()
        
        if (is_saved == True):
            plt.savefig(img_name,bbox_inches='tight')
        else:
            plt.show()                  
        plt.close()

    def generate_algorithm_gif(self, matrix, connection_list, s1, s2, use_log=False, figsize=(10, 10), 
                    color_s1='limegreen', color_s2='cornflowerblue', show_numb=False, is_saved=False, gif_name='DTW_algo_animation.gif'):
        ''' 
        @params:
            - matrix: 2-D np.array.
            - connection_list: list of list of int, mapping between 2 time series.
            - s1, s2: TimeSeries.
            - use_log: boolean. Transform matrix's cell value with natural log.
            - figsize: tuple, should be same size. 
            - color_s1, color_s2: str, a seaborn color.
            - show_numb: boolean, shows cost on cell.
            - is_saved: boolean, to save the plot to file or not. 
            - gif_name: str, file name. Only applicable if is_saved=True.
        '''
        # generate image
        # matrix
        mask = np.ones_like(matrix[1:, 1:])
        n_steps_matrix_gen = mask.shape[0]
        img_list = list()

        for step_i in range(n_steps_matrix_gen):
            mask[step_i, :] = False
            self.plot_matrix(matrix, connection_list, s1, s2, mask=mask, line_ratio=0, use_log=use_log, figsize=figsize,
                             color_s1=color_s1, color_s2=color_s2, show_numb=show_numb, is_saved=True, img_name='temp.png')
            img_list.append(plt.imread('temp.png'))
            os.remove('temp.png')

        # backtracking line
        for line_ratio in [0.33, 0.66, 1.]:
            self.plot_matrix(matrix, connection_list, s1, s2, mask=mask, line_ratio=line_ratio, use_log=use_log, figsize=figsize,
                             color_s1=color_s1, color_s2=color_s2, show_numb=show_numb, is_saved=True, img_name='temp.png')
            img_list.append(plt.imread('temp.png'))
            os.remove('temp.png')


        # merge to gif file
        frames = [] # for storing the generated images
        fig = plt.figure(figsize=figsize)
        plt.axis('off')
        frames = [[plt.imshow(img, cmap=cm.Greys_r,animated=True)] for img in img_list]
        ani = animation.ArtistAnimation(fig, frames, interval=250, blit=True,
                                        repeat_delay=1000)
        
        if (is_saved == True):
            ani.save(gif_name)
            plt.close()
        else:
            plt.show()
            plt.close()


    def reshape_s1_to_s2(self, s1, s2, connection_list):
        s1, s2 = s1.get_vals(), s2.get_vals()
        new_s1 = list()
        connection_mapping_list = list()
        left_list = [0]
        right_list = [0]

        for point_in_s1, point_in_s2 in connection_list[1:]:
            s1_x, _ = point_in_s1
            s2_x, _ = point_in_s2
            
            # this is 1-to-1 mapping
            if ((s1_x not in left_list) and (s2_x not in right_list)):
                connection_mapping_list.append([left_list, right_list])
                left_list, right_list = [s1_x], [s2_x]

            # this is 1-to-n mapping for s1_x
            elif (s1_x in left_list):
                right_list.append(s2_x)

            # this is n-to-1 mapping for s1_x -> merge
            elif (s2_x in right_list):
                left_list.append(s1_x)

            else:
                raise ValueError(f'connection_mapping_list has a strange x_pair: {x_pair}.')

        else:
            # add the last mapping
            connection_mapping_list.append([left_list, right_list])

        for left_list, right_list in connection_mapping_list:
            n_points_left, n_points_right = len(left_list), len(right_list)
            if ((n_points_left == 1) and (n_points_right == 1)):
                idx = left_list[0]
                new_s1.append(s1[idx])
            
            elif ((n_points_left == 1) and (n_points_right > 1)):
                idx = left_list[0]
                new_s1.extend([s1[idx]]*n_points_right)

            elif ((n_points_left > 1) and (n_points_right == 1)):
                idx_list = left_list
                new_s1.append(np.mean(s1[idx_list]))

            else:
                raise ValueError(f'there is an unexpected n-to-n mapping: {left_list} vs {right_list}.')

        new_s1 = np.array(new_s1)
        return TimeSeries(new_s1)
            
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

    algo = DynamicTimeWarping(s1, s2)
    DTW_dist, matrix, connection_list, s1, s2 = algo.compute(max_ele=40)
    
    # Plot matrix
    algo.plot_matrix(matrix, connection_list, s1, s2)
    
    # Save matrix
    algo.plot_matrix(matrix, connection_list, s1, s2, is_saved=True)
    
    # Plot time series pair mapping
    algo.plot_series_mapping(s1, s2, connection_list)
    
    # Generate gif to illustrate the algorithm
    algo.generate_algorithm_gif(matrix, connection_list, s1, s2)
    
    # Assymetric DTW resampling
    s1 = algo.reshape_s1_to_s2(s1, s2, connection_list)
    
    # Recompute DTW after resampling s.t. time series pair has the same length
    algo = DynamicTimeWarping(s1, s2)
    DTW_dist, matrix, connection_list, s1, s2 = algo.compute(max_ele=40)
    algo.plot_series_mapping(s1, s2, connection_list)