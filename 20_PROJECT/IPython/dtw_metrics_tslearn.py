import numpy as np
import tslearn.metrics as tsm
from tslearn.preprocessing import TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance, TimeSeriesResampler

def row_wise_minmax_scaling(x):
    '''
    Takes a 2D array and scales each row to the range of [0.0, 1.0]
    '''
    scaler = TimeSeriesScalerMinMax(value_range=(0.0, 1.0))
    return (scaler.fit_transform(x).squeeze())

def ptp_scaler(x):
    x -= np.min(x)
    if np.ptp(x) != 0:        
        x /= np.ptp(x)    
    return x

def rms_difference(x, y):
    '''
    x and y - 1D arrays of same size
    '''
    assert(len(x) == len(y))
    summ = 0
    for x_i, y_i in zip(x, y):
        summ += (x_i - y_i)**2
    return summ / len(x)

def get_dtw_distance_mse(query, reference):
    '''
    Parameters:
    query     : 1D array
    reference : 1D array
    '''    
    if len(query) > len(reference):
        query, reference = reference, query
    try:
        path, distance = tsm.dtw_path(query, reference, global_constraint='sakoe_chiba')        
        warped_reference = np.zeros(len(query))
        for pair in path:
            warped_reference[pair[0]] = reference[pair[1]]        
        mse = rms_difference(query, warped_reference)
        return distance, mse
    except ValueError:
        return (1000, 100)

def get_row_wise_dtw_metrics(query, reference):
    '''
    Takes two 2D arrays with equal numbers of rows.
    Aligns rows with dynamic time wrapping algorythm.
    Returns mean DTW distance and mean MSE between aligned rows.
    '''
    assert(len(query)==len(reference))
    if len(query[0]) > len(reference[0]):
        query, reference = reference, query    
    
    distance_values = []
    mse_values = []    
    for i in range(len(query)):
        distance, mse = get_dtw_distance_mse(query[i], reference[i])
        distance_values.append(distance)        
        mse_values.append(mse)
    return np.mean(distance_values), np.mean(mse_values)

def get_column_wise_dtw_metrics(query, reference):
    '''
    Calculates dtw distances between columns.
    Transposes input arrays (columns become rows),
    calculates DTW distance using euclidean norm and MSE between aligned columns. 
    Parameters:
    query     : 2D array with shape n * m (n time frames, m features)
    reference : 2D array with shape k * m (k time frames, m features)    
    '''    
    assert(len(query)==len(reference))
    if len(query[0]) > len(reference[0]):
        query, reference = reference, query        
    query = query.T
    reference = reference.T
    path, distance = tsm.dtw_path(query, reference, global_constraint='sakoe_chiba')    
    warped_reference = np.zeros(query.shape)
    for pair in path:
        warped_reference[pair[0]] = reference[pair[1]]        
    mse = np.linalg.norm(query - warped_reference) / len(query)
    return distance, mse