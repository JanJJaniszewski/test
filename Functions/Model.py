# Model Functions
import doctest
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import pydotplus
from IPython.display import Image
from matplotlib.backends.backend_pdf import PdfPages
from pandas import DataFrame
from sklearn.base import TransformerMixin
from sklearn.externals.six import StringIO
from sklearn.learning_curve import validation_curve
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.tree import export_graphviz
import numpy as np

class ModelTransformer(TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        return DataFrame(self.model.predict(X))


def my_visual_parameter_tuning(x_train, y_train, pipe, param_name, n_fold, param_range):
    """Shows a plot to see which value for a parameter is the best.

    Args:
        x_train (DataSeries): X variable
        y_train (DataSeries): y variable
        pipe (Pipeline): Pipeline with the input set
        n_fold (int): How often the cross validation is done
        param_name (String): The name of the parameter that should be tested
        param_range (List): List of a range of parameter values
    Returns:
        bool: True if the function worked
    """
    # Visualization Check
    train_scores, test_scores = validation_curve(estimator=pipe, X=x_train, y=y_train, param_name=param_name,
                                                 param_range=param_range, cv=n_fold, n_jobs=1, verbose=1)
    # Get mean and SD
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    # Plot Training and test group accuracy
    plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5,
             label='validation accuracy')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('Parameter')
    plt.ylabel('Score')
    plt.show()
    return True


def my_roc_curve(true, pred):
    """Compute a ROC-curve based on predicted and true values

    Args:
        true (DataSeries): true values
        pred (DataSeries): predicted values
    """
    fpr, tpr, _ = roc_curve(true, pred)
    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr, tpr, label='ROC')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.show()
    return True


def my_learning_curve(x_train, y_train, pipe, n_fold, scoring):
    """Show a plot with the learning curve

    Args:
        x_train (DataSeries): X variable
        y_train (DataSeries): y variable
        pipe (Pipeline): Pipeline with the input set
        n_fold (int): How often cross validation is done
        scoring (String): Scoring method
    """
    # Check for over- or underfitting
    train_sizes, train_scores, test_scores = learning_curve(scoring=scoring, estimator=pipe, X=x_train, y=y_train,
                                                            train_sizes=np.linspace(0.1, 1.0, 10), cv=n_fold, n_jobs=1,
                                                            verbose=1)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5,
             label='validation accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.show()
    return True





def model_summary(x, y, clf, scorers, pdf_path, test_size=0.3, timecolumn=None):
    with PdfPages(pdf_path) as pp:
        for score in model_error(x, y, clf, scorers):
            print(score)
        pp.savefig(plot_real_pred(x, y, clf, test_size=test_size))
        pp.savefig(plot_residuals(x, y, clf))
        if timecolumn:
            pp.savefig(time_residuals(timecolumn, x, y, clf, test_size=test_size))
        pp.savefig(plot_corr(x))


def model_error(x, y, clf, *scorers):
    """Provide model error for scorer

    :param x:
    :param y:
    :param clf:
    :param scorers:
    :return:
    """
    for s in scorers:
        cv_score = cross_val_score(clf, x, y, scoring=s, cv=5)
        print('{} test results: {}'.format(s, cv_score))
        print('{} test mean: {}'.format(s, round(numpy.mean(cv_score), 2)))
        yield '{} test mean: {}'.format(s, round(numpy.mean(cv_score), 2))


def train_predict(x, y, clf, test_size=0.3):
    """Split data, train model on test data, predict test data, and then make a cv of the data and
    return test data y and predicted test data y

    :param x:
    :param y:
    :param clf:
    :param scorers:
    :param test_size:
    :return:
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return y_test, y_pred


def plot_real_pred(x, y, clf, test_size=0.3):
    """Plot real vs. predicted values of the model

    :param x:
    :param y:
    :param clf:
    :param test_size:
    :return:
    """
    y_test, y_pred = train_predict(x, y, clf, test_size=test_size)

    # Figure
    fig, ax = plt.subplots()
    ax.set_facecolor('white')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.set_xlim([y_test.min(), y_test.max()])
    ax.set_ylim([y_test.min(), y_test.max()])
    ax.scatter(y_test, y_pred, color='black')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=1, color='black',
            label='Perfect Predictions')

    # Best fit line
    clf2 = Ridge()
    clf2.fit(y_test.reshape(-1, 1), y_pred)
    y_test = numpy.sort(y_test)
    line = clf2.predict(y_test.reshape(-1, 1))
    ax.plot(y_test, line, lw=1, color='green', label='Model Predictions')
    ax.legend()
    ax.set_title('Model Quality: Predictions vs. Real Values')
    ax.set_ylabel("PREDICTED Target Function")
    ax.set_xlabel("REAL Target Function")
    ax.set_xlim(ax.get_ylim())

    plt.show()
    plt.close()
    return fig


def plot_residuals(x, y, clf):
    y_test, y_pred = train_predict(x, y, clf)

    fig, ax = plt.subplots()
    ax.scatter(y_pred, y_test - y_pred)
    ax.plot([min(y_pred), max(y_pred)], [0, 0], 'k--', lw=1)
    ax.set_title('Model Quality: Residuals Plot')
    ax.set_ylabel("Residuals")
    ax.set_xlabel("PREDICTED")

    plt.show()
    return fig


def time_residuals(t, x, y, clf, test_size=0.3):
    """Plots the prediction error of the model over time

    :param t:
    :param x:
    :param y:
    :param clf:
    :param test_size:
    :return:
    """
    x_train, x_test, y_train, y_test, t_train, t_test = train_test_split(x, y, t, test_size=test_size)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    fig, ax = plt.subplots()
    ax.scatter(t_test, y_test - y_pred)
    ax.plot([min(t_test), max(t_test)], [0, 0], 'k--', lw=1)
    ax.set_title('Model Quality: Time bound residuals Plot')
    ax.set_ylabel("Residuals")
    ax.set_xlabel("Batch Code")
    plt.show()
    return fig


def plot_corr(df, size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''
    corr = abs(df.corr())
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr, vmin=0, vmax=1)
    fig.colorbar(cax)
    ax.set_title('Correlation Matrix (0: no corr.; 1: strong  corr.')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()
    return fig


def plot_tree(x, y, output_path, typ='classification', max_depth=3):
    assert typ in ['classification', 'regression'], 'Model type can only be classification or regression'
    if typ=='classification':
        dtree = DecisionTreeClassifier(max_depth=max_depth)
        scorer = 'accuracy'
    elif typ=='regression':
        dtree = DecisionTreeRegressor(max_depth=max_depth)
        scorer = 'r2'
    else:
        Exception('Type of tree can only be classification or regression')
    dtree.fit(x, y)
    dot_data = StringIO()
    export_graphviz(dtree, out_file=dot_data, feature_names=x.columns,
                    filled=True, rounded=True,
                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    print(model_error(x, y, dtree, scorer))
    graph.write_png(output_path)

    return Image(graph.create_png())


def feature_importance(x, feature_importances):
    return pd.DataFrame({'Feature': x.columns, 'Feature Importance': feature_importances}). \
        sort_values('Feature Importance', ascending=False)


if __name__ == '__main__':
    doctest.testmod()
