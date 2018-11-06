# Plotting Functions
import doctest
from sklearn import linear_model
import pandas as pd
import matplotlib as plt
from sklearn.metrics import r2_score
from Functions.General import breaks
from Functions.Model import feature_importance


def glm_scatter(x, y, pdf_path=None, title=None, x_label='x', y_label='y', formula='y ~ x', show_sum=False):
    """Creates a scatter plot with a GLM model running through the data

    :param x:
    :param y:
    :param pdf_name:
    :param title:
    :param x_label:
    :param y_label:
    :param formula:
    :param show_sum:
    :return:
    """
    # Make the plotting grid
    a = plt.axes(title=title)
    a.set_xlabel(x_label)
    a.set_ylabel(y_label)

    # Prepare data and model
    scatter_df = pd.DataFrame({'x': x, 'y': y}).sort_values('x')
    mod = smf.ols(formula=formula, data=scatter_df).fit()
    scatter_df['pred'] = mod.predict(scatter_df['x'])
    scatter_df.plot('x', 'y', kind='scatter', ax=a)
    scatter_df.plot('x', 'pred', ax=a, color='r')

    # Show plot
    plt.show()

    if pdf_path:
        pdf_path = add_filetype(pdf_path, '.pdf')
        with PdfPages(pdf_path) as pdf:
            plt.savefig(pdf, format='pdf')

    plt.close('all')
    if show_sum:
        print(mod.summary())
    breaks()


def feature_summary(x_col, y_col, show_r2=False):
    """Gives a summary of a feature

    :return:
    """
    # Preparation
    x_name = x_col.name
    y_name = y_col.name
    df = pd.concat([x_col, y_col], axis=1).sort_index()
    plt.rcParams["figure.figsize"] = (10, 7)
    breaks(1)
    print("%s" % x_name)
    print('Quantile:\n', x_col.quantile([0.0, 0.1, 0.25, 0.5, 0.75, 1.0]))

    # Histogram
    plt.subplot(221)
    try:
        plt.hist(x_col, bins=30)
        plt.xlabel(x_name)
        plt.title('Histogram (CF GHP): %s' % x_name)
    except ValueError:
        print("No histogram for %s available" % x_name)

    # Correlation
    if y_name != x_name:
        df = df.sort_values(x_name)
        # df[x_name + "_2"] = df[x_name] * df[x_name]
        # df[x_name + "_3"] = df[x_name] * df[x_name] * df[x_name]
        x = df.drop(y_name, 1)
        reg = linear_model.LinearRegression(normalize=True)
        reg.fit(x, df[y_name])
        # Plot
        plt.subplot(222)
        plt.scatter(df[x_name], df[y_name])
        plt.plot(df[x_name], reg.predict(x), color='g')
        plt.xlabel(x_name)
        plt.xlim([df[x_name].min(), df[x_name].max()])
        plt.title('x:%s / y:%s ' % (x_name, y_name))
        plt.ylabel("Target function: %s" % y_name)
        if show_r2:
            print("R²:", r2_score(df[y_name], reg.predict(x)))
            print(feature_importance(x, reg.coef_))

    # Show plots
    plt.show()

    # Timeline
    x_col.rolling(window=10, center=False).mean().plot(title='%s: Timeline' % x_name, figsize=(10, 2),
                                                       xlim=(170000, 175000))
    plt.show()

    plt.close('all')
    return " "

if __name__ == '__main__':
    doctest.testmod()