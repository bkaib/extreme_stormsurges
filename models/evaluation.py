#---
# Modules
#---
from matplotlib.pyplot import tight_layout
import numpy as np

#---
# Graphical evaluation
#---
def plot_cf(cf_matrix):
    """
    Description:
        Plots a 2x2 Confusion Matrix labeled with percentages and absolute values

    Parameters:
        cf_matrix (np.array): confusion matrix generated with sklearn.metrics.confusion_matrix. Shape(2,2)

    Returns:
        fig (matplotlib.figure.Figure): Figure of labeled confusion matrix
    
    Source (adjusted):
        "https://www.stackvidhya.com/plot-confusion-matrix-in-python-and-why/"
    """
    
    # Plot Confusion Matrix
    #---
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Labels for quadrants in matrix
    group_names = ['True Neg','False Pos','False Neg','True Pos']

    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                        cf_matrix.flatten()/np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
            zip(group_names, group_counts, group_percentages)]

    labels = np.asarray(labels).reshape(2,2)

    # Main Figure
    fig, ax = plt.subplots(tight_layout=True)
    
    ax = sns.heatmap(cf_matrix,
    annot=labels, 
    fmt='', 
    cmap='Blues',
    )

    # Axis labels
    ax.set_title('Confusion Matrix \n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    # Axis ticks
    ax.xaxis.set_ticklabels(['0','1'])
    ax.yaxis.set_ticklabels(['0','1'])

    return fig