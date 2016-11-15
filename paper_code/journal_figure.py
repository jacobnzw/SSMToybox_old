import numpy as np
import matplotlib as mpl
mpl.use('pgf')


def figsize(scale=1.0):
    """
    Calculates figure width and height given the scale.

    Parameters
    ----------
    scale: float
        Figure scale.

    Returns
    -------

    """

    FIG_WIDTH_PT = 347.12354  # Get this from LaTeX using \the\columnwidth
    INCH_PER_PT = 1.0 / 72.27  # Convert pt to inch
    PHI = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)

    fig_width = FIG_WIDTH_PT * INCH_PER_PT * scale    # width in inches
    fig_height = fig_width * PHI * 0.85         # height in inches
    return [fig_width, fig_height]

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "font.size": 10,
    "axes.labelsize": 10,               # LaTeX default is 10pt font.
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    # "axes.prop_cycle": ['#5DA5DA', '#FAA43A', '#60BD68',
    #                     '#F17CB0', '#B2912F', '#B276B2',
    #                     '#DECF3F', '#F15854', '#4D4D4D'],
    "figure.figsize": figsize(1.0),      # default fig size of 0.9 textwidth
    "pgf.preamble": [                    # plots will be generated using this preamble
        r"\usepackage[utf8]{inputenc}",  # use utf8 fonts
        r"\usepackage[T1]{fontenc}",
        ]
    }
mpl.rcParams.update(pgf_with_latex)
import matplotlib.pyplot as plt


def newfig(width):
    # for making new figure
    plt.clf()
    fig = plt.figure(figsize=figsize(width))
    ax = fig.add_subplot(111)
    return fig, ax


def savefig(filename):
    """
    Save figure to PGF. PDF copy created for viewing convenience.

    Parameters
    ----------
    filename

    Returns
    -------

    """
    plt.savefig('{}.pgf'.format(filename))
    plt.savefig('{}.pdf'.format(filename))
