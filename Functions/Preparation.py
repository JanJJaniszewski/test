# Transformation Functions
import doctest
import matplotlib.pyplot as plt
from Functions.General import breaks
import numpy as np


def convert_columns_with_function(df, newtype, *columns_to_convert):
    for col in columns_to_convert:
        df[col] = df[col].astype(newtype)
    return df

def describe_features(df, timeline=False):
    """Filters variables based on filter introduced


    :param df: Dataframe without filters applied
    :param filters: Filters you want to apply
    :return: Filtered dataframe
    """
    print(df.columns)
    df = df.sort_index()
    print("\nLength of dataframe:", len(df))
    plt.rcParams["figure.figsize"] = (10, 7)

    for col in df.columns:
        if not np.issubdtype(df[col].dtype, np.number):
            continue
        print("Describing: %s\n" % (col))
        if timeline:
            df[col].rolling(window=10, center=False).mean().plot(title='%s: Timeline of Variable' % col, figsize=(10, 2),
                                                             xlim=(170000, 175000))
            plt.show()

        # Show distribution before filtering
        plt.hist(df[col].dropna(), bins=30)
        plt.title('%s: Distribution' % col)
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()
        plt.close()
        print("Quantiles:\n", df[col].quantile([0, 0.25, 0.5, 0.75, 1]))
        breaks(1)
    print(df.columns)

def filter_variables(df, **filter_tuples):
    """Filters variables based on filter introduced


    :param df: Dataframe without filters applied
    :param filters: Filters you want to apply
    :return: Filtered dataframe
    """
    print(df.columns)
    df = df.sort_index()
    print("\nFiltering dataframe. Length of dataframe before filtering: %i" % len(df))
    plt.rcParams["figure.figsize"] = (10, 7)

    for col, rule in filter_tuples.items():
        print("Filtering: %s %s\n" % (col, rule))

        # Show distribution before filtering
        plt.subplot(221)
        plt.hist(df[col].dropna(), bins=30)
        plt.title('%s: Distribution BEFORE filtering' % col)
        plt.xlabel(col)
        plt.ylabel('Frequency')
        print("Quantiles BEFORE filtering:\n", df[col].quantile([0.0, 0.25, 0.5, 0.75, 1]))

        # Filtering
        f = eval("df[col] %s" % (rule))
        df = df[f]

        # Distribution after filtering
        plt.subplot(222)
        plt.hist(df[col].dropna(), bins=30)
        plt.title('%s: Distribution AFTER filtering' % col)
        plt.xlabel(col)
        plt.ylabel('Frequency')

        plt.show()

        print("Quantiles AFTER filtering:\n", df[col].quantile([0.0, 0.25, 0.5, 0.75, 1]))
        print("Length of dataframe after filtering: %i" % len(df))
        breaks(1)
    print(df.columns)

    return df


# Time Functions
def up_to_minval(t, minval=1):
    """Round up to the minval value (can be 1 minute, 5 minutes, 10 minutes, etc.)

    :param t: Time value
    :examples:
    >>> t1 = datetime(2016, 3, 25, 4, 36, 0, 24000)
    >>> up_to_minval(t1, 5)
    datetime.datetime(2016, 3, 25, 4, 40)
    >>> up_to_minval(t1, 1)
    datetime.datetime(2016, 3, 25, 4, 37)
    """
    delta = timedelta(minutes=t.minute % minval, seconds=t.second, microseconds=t.microsecond)
    t -= delta
    if delta > timedelta(0):
        t += timedelta(minutes=minval)
    return t


def perdelta(start, end, delta):
    """Creates an index with timestamp values

    :param start: Starting date
    :param end: Ending Date
    :param delta: Increase of timestamp values
    :return: List with timestamp values

    :examples:
    >>> for result in perdelta(datetime(2017, 1, 1, 1, 1), datetime(2017, 1, 1, 1, 3), timedelta(minutes=1)):
    ...     print(result)
    2017-01-01 01:01:00
    2017-01-01 01:02:00
    """
    curr = start
    times = []
    while curr < end:
        times.append(curr)
        curr += delta
    return times


def join_minutes(minute_path, *joiner_names, start, end, deltatime=1):
    """Joins minute data with a basis timeline created by start-end index

    :param minute_path: Path to the minute datasets
    :param joiner_names: Names of the files to join
    :return: DataFrame that including all necessary values
    """
    time_index = perdelta(start, end, timedelta(minutes=deltatime))
    basis = pd.DataFrame({'time': time_index})

    for joiner_name in joiner_names:
        # Structure dataframe
        joiner = pd.read_csv(paths(minute_path, joiner_name))
        var_name = joiner.iloc[0, 0]
        joiner.columns = ['DEL', 'time', var_name]
        # Set time as datetime object
        joiner['time'] = joiner['time'].apply(lambda t: datetime.strptime(t, "%d-%b-%y %H:%M:%S.%f"))
        joiner = joiner[['time', var_name]].drop_duplicates().dropna(subset=["time"]).sort_values("time")

        # Join both data sets
        basis = pd.merge_asof(basis, joiner, direction='nearest', on='time')

    basis = basis.groupby('time').last().reset_index()

    return basis

if __name__ == '__main__':
    doctest.testmod()