#- Modules
import numpy as np

#- Main
def get_duplicated_idx(df):
    """
    Description: 
        Checks for duplicate indices in dataframe

    Parameters:
        df(pd.Dataframe)

    Output: 
        Index position of duplicates
    """

    if np.where(df.index.duplicated() == True)[0].size == 0:
        return None
    else:
        return np.where(df.index.duplicated() == True)[0]

def assert_timeintervals(t1, t2):
    """
    Description: 
        Checks if time intervals are the same. Throws assertion error if not.
    Parameters:
        t1, t2 (np.array): Arrays of time points, shape:(timepoints,)
    Note: 
        Can be used in general for two np.arrays
    """

    if len(np.where(t1 != t2)[0]) == 0:
        is_equal = True
    else:
        is_equal = False

    assert is_equal, "Timeinterval is not equal"