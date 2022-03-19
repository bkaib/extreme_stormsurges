#- Modules
import numpy as np

#- Main
def get_duplicated_idx(df):
    """
    Description: Checks for duplicate indices in dataframe

    Output: Index position of duplicates
    """

    if np.where(df.index.duplicated() == True)[0].size == 0:
        return None
    else:
        return np.where(df.index.duplicated() == True)[0]