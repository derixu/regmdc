import numpy as np

def get_unique_column_entries(X_design, method='tcmars', number_of_bins=None):
    '''
    Currently only supports tcmars
    
    X_design: n x d numPy array or pandas DataFrame containing n datapoints with d features
    number_of_bins: length d list or numPy array containing bin counts (ints) for each column of X_design
    
    More type restrictive than R version 
        - number_of_bins cannot be an integer, and incorrect lengths of number_of_bins is not handled
        - if an element of number_of_bins is not an integer, fallback to default behavior (no bins).
    '''
    
    if method == 'tcmars':
        if number_of_bins is None:
            # list comprehension may be inefficient
            # np.unique sorts by default
            return [np.unique(np.append(0, X_design[:, i]))[:-1] for i in range(X_design.shape[1])]
        else:
            # for loop may be inefficient
            column_unique = []
            for i in range(X_design.shape[1]):
                N = number_of_bins[i]
                if N is None:
                    column_unique.append(np.unique(np.append(0, X_design[:, i]))[:-1])
                else:
                    column_unique.append(np.array(np.arange(N))/N)
            return column_unique
    else:
        raise ValueError("method must be tcmars")