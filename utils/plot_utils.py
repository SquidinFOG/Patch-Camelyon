import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report

def plot_training(results, save_path=None, name=None):
    # Plot losses
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(results['train_losses'], label='Training Losses')
    plt.plot(results['val_losses'], label='Validation Losses')
    plt.title('Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy losses
    plt.subplot(1, 2, 2)
    plt.plot(results['train_accuracies'], label='Training Accuracy')
    plt.plot(results['val_accuracies'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path + name, dpi=300, bbox_inches='tight')
    else:
        plt.show()    

def cf_matrix(y_true, y_pred, normalize='true', class_names=None, title=None,
              xlabel='Predicted label', ylabel='True label', color_map='Blues', 
              fmt='.2f', fig_width=5.5, fig_height=5.5, save_path=None, name=None):
    """""
    Returns a confusion matrix (built with scikit-learn) generated on a given set of true and predicted labels.

    Parameters:
        y_true : array
            True labels. Array shape = (n_spectra,) for integer labels
            and (n_spectra, n_classes) for binary labels.

        y_pred : array
            Predicted labels. Array shape = (n_spectra,) for integer labels
            and (n_spectra, n_classes) for binary labels.

        normalize : {'true', 'pred', None}, default=None
            - 'true': Normalizes confusion matrix by true labels (row). Gives the recall scores
            - 'predicted': Normalizes confusion matrix by predicted labels (col). Gives the precision scores
            - None: Confusion matrix is not normalized.

        class_names : list or tupple of string, default=None
            Names or labels associated to the class. If None, class names are not displayed.

        title : string, default = None
            Confusion matrix title. If None, there is no title displayed.

        xlabel : string, default='Predicted label'
            X-axis title. If None, there is no title displayed.

        ylabel : string, default='True label'
            Y-axis title. If None, there is no title displayed.

        color_map : string, default = 'Blues'
            Color map used for the confusion matrix heatmap.

        fmt: String, default = '.2f'
            String formatting code for confusion matrix values. Examples:
                - '.0f' = integer
                - '.2f' = decimal with two floating values
                - '.3%' = percentage with three floating values

        fig_width : positive float or int, default=5.5
            Figure width in inches.

        fig_height : positive float or int, default=5.5
            Figure height in inches.

        save_path : string, default=None
            Path where the figure is saved. If None, saving does not occur.

    Return:
        Scikit Learn confusion matrix
    """
    # Converts binary labels to integer labels. Does nothing if they are already integer labels.
    if y_true.ndim == 2 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim == 2 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)

    # scikit learn confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, normalize=normalize)  # return a sklearn conf. matrix
    
    # creates a figure object
    fig = plt.figure(figsize=(fig_width, fig_height))
    # add an axes object
    ax = fig.add_subplot(1, 1, 1)  # nrows, ncols, index
    # plot a Seaborn heatmap with the confusion matrix
    sns.heatmap(conf_matrix, annot=True, cmap=color_map, fmt=fmt, cbar=False, annot_kws={"fontsize": 10},
                square=True, )

    # titles settings
    ax.set_title(title, fontsize=11.2, color='k')  # sets the plot title, 1.2 points larger font size
    ax.set_xlabel(xlabel, fontsize=10, color='k')  # sets the x-axis title
    ax.set_ylabel(ylabel, fontsize=10, color='k')  # sets the y-axis title

    # tick settings
    ax.tick_params(axis='both', which='major',
                   labelsize=8,  
                   color='k')
    ax.tick_params(axis='x', colors='k')  # setting up X-axis values color
    ax.tick_params(axis='y', colors='k')  # setting up Y-axis values color

    # spine settings
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('k')

    # class names settings
    if class_names is not None:
        ax.set_xticklabels(class_names, rotation=45, ha="right", rotation_mode="anchor",
                           fontsize=8.5, color='k')
        ax.set_yticklabels(class_names, rotation=0, fontsize=8, color='k')

    # set figure and axes facecolor
    fig.set_facecolor('w')
    ax.set_facecolor('w')
    fig.tight_layout()
    
    # save figure
    if save_path is not None:
        plt.savefig(save_path + name, dpi=300, bbox_inches='tight')
    else:
        plt.show()  # display the confusion matrix image
    return conf_matrix

def clf_report(y_true, y_pred, digits=4, print_report=True, class_names=None, save_path=None, name=None):
    """ Returns a classification report generated from a given set of spectra

    Parameters:
        y_true : array
            True labels. Array shape = (n_spectra,) for integer labels
            and (n_spectra, n_classes) for one-hot encoded labels.

        y_pred : array
            Predicted labels. Array shape = (n_spectra,) for integer labels
            and (n_spectra, n_classes) for one-hot encoded labels.

        digits : non-zero positive integer values, default=4
            Number of digits (ie. precision) to display in the classification report.

        print_report : boolean, default=True
            If True, print the classification report

        class_names : list or tupple of string, default=None
            Names or labels associated to the class. Class names are not displayed if None.

        save_path: string, default=None
            Path where the report is saved. If None, saving does not occur.

    Returns:
        Scikit Learn classification report
    """
    # # do not converts labels to integer labels if multilabel == True
    # if not multilabel:
    # Converts one-hot encoded labels to integer labels . Does nothing if they are already integer labels.
    if y_true.ndim == 2 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim == 2 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)

    # generates the classification report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=digits)

    if print_report:
        print(report)

    if save_path is not None:
        text_file = open(save_path+name, "w")
        text_file.write(report)
        text_file.close()
    return report



