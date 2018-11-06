import doctest
import logging
import os
from distutils.dir_util import copy_tree
from os.path import join as paths
from shutil import move
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

def add_filetype(*filenames, ending='.csv'):
    """Adds an ending to filename(s)

    :param filenames: filename(s)
    :param ending: File ending
    :return: Filename (list) with ending added

    :example:
    >>> add_filetype('foo', 'bar.csv')
    ['foo.csv', 'bar.csv']
    >>> add_filetype('foo')
    'foo.csv'

    """
    if len(filenames) == 1:
        return filenames[0] + ending
    else:
        return [f + ending if ending not in f else f for f in filenames]


def breaks(n=1):
    """Prints n line breaks

    :example:
    >>> breaks(1)
    ____________________________________________________________________________________________________
    <BLANKLINE>
    """
    print(n * ('%s\n' % ('_' * 100)))


def create_logger(logname, logpath):
    """

    :return:
    """
    # Logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(logname)
    logger.propagate = False

    handler = logging.FileHandler(logpath)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add Handler to logger
    logger.addHandler(handler)

    return logger


def save_plot_pdf(path, *plots):
    """Saves plots in a pdf

    :param path: Path odf the pdf
    :param plots: plots
    :return: None
    """
    with PdfPages(path) as pp:
        for p in plots:
            pp.savefig(p)


def new_model(typ: str, model_path: str, message: str):
    logger = create_logger('NEW MODEL', paths(model_path, 'Logs', 'Model.log'))

    subfolders = os.listdir(model_path)
    subfolders.sort()
    currentVersion = subfolders[-1].split('V')[1]

    if typ == 's':
        nextVersion = str(int(currentVersion) + 1)
    elif typ == 'm':
        nextVersion = str(int(currentVersion) + 100)
    elif typ == 'l':
        nextVersion = str(int(currentVersion) + 10000)

    nextVersion = nextVersion.zfill(6)
    logger.info('NEW MODEL VERSION %s: %s' % (nextVersion, message))

    copy_tree(paths(model_path, 'V' + currentVersion), paths(model_path, 'V' + nextVersion))
    move(paths(model_path, 'V' + currentVersion), paths(model_path, 'OLD'))


if __name__ == '__main__':
    doctest.testmod()
