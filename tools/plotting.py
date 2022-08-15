import numpy as np
import scipy
import pandas as pd
from skimage import filters
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import seaborn as sns
import proplot as pplt

from .analysis import rms_ellipse_dims
from . import utils


DEFAULT_COLORCYCLE = pplt.Cycle('colorblind').by_key()['color']

        
def remove_annotations(axes):
    """Delete all text (as well as arrows) from the figure."""
    if type(axes) is not np.ndarray:
        axes = np.array([axes])
    for ax in axes.flat:
        for annotation in ax.texts:
            annotation.set_visible(False)
        
        
def process_limits(mins, maxs, pad=0., zero_center=False):
    # Same limits for x/y and x'/y'
    widths = np.abs(mins - maxs)
    for (i, j) in [[0, 2], [1, 3]]:
        delta = 0.5 * (widths[i] - widths[j])
        if delta < 0.:
            mins[i] -= abs(delta)
            maxs[i] += abs(delta)
        elif delta > 0.:
            mins[j] -= abs(delta)
            maxs[j] += abs(delta)
    # Pad the limits by fractional amount `pad`.
    deltas = 0.5 * np.abs(maxs - mins)
    padding = deltas * pad
    mins -= padding
    maxs += padding
    if zero_center:
        maxs = np.max([np.abs(mins), np.abs(maxs)], axis=0)
        mins = -maxs
    return mins, maxs
    

def auto_limits(X, pad=0.0, zero_center=False, sigma=None):
    """Determine axis limits from coordinate array."""
    if sigma is None:
        mins = np.min(X, axis=0)
        maxs = np.max(X, axis=0)
    else:
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        widths = 2.0 * sigma * stds
        mins = means - 0.5 * widths
        maxs = means + 0.5 * widths
    mins, maxs = process_limits(mins, maxs, pad, zero_center)
    return [(lo, hi) for lo, hi in zip(mins, maxs)]
    

def corner(
    X, kind='hist', figsize=None, limits=None, hist_height_frac=0.6,
    samples=None, smooth_hist=False, thresh=None, blur=None, log=False,
    env_params=None, env_kws=None, rms_ellipse=False, rms_ellipse_kws=None,
    autolim_kws=None, grid_kws=None, diag_kws=None, **plot_kws
):
    """Plot the pairwise relationships between the coordinates.

    This is similar to routines in other packages like `scatter_matrix` in 
    Pandas or `pairplot` in Seaborn.
    
    Parameters
    ----------
    X : ndarray, shape (n, d)
        Array of d-dimensional coordinates.
    kind : {'hist', 'scatter'}:
        The type of bivariate plot.
    figsize : tuple or int
        Size of the figure (x_size, y_size). 
    limits : list
        List of (min, max) for each dimension.
    hist_height_frac : float
        Fractional reduction of 1D histogram heights.
    samples : int or float
        If an int, the number of points to use in the scatter plots. If a
        float, the fraction of total points to use.
    smooth_hist : bool
        If True, connect 1D histogram heights with lines. Otherwise use a
        bar plot.
    thresh : float
        In the 2D histograms, with count < thresh will not be plotted.
    blur : float
        Apply a Gaussian blur to the 2D histograms with sigma=blur.
    log : bool
        Whether to use log scale in 2D histogram.
    env_params : list or ndarray, shape (8,)
        Danilov envelope parameters passed to `corner_env`.
    env_kws : dict:
        Key word arguments passed to `ax.plot` for the Danilov envelope.
    rms_ellipse : bool
        Whether to plot the rms ellipses.
    rms_ellipse_kws : dict
        Key word arguments passed to `ax.plot` for the rms ellipses. Pass
        {'2rms': False} to plot the true rms ellipse instead of the 2-rms
        ellipse.
    autolim_kws : dict
        Key word arguments for `auto_limits` method.
    grid_kws : dict
        Key word arguments for `pair_grid` method.
    diag_kws : dict
        Key word arguments for the univariate plots.
    plot_kws : dict
        Key word arguments for the bivariate plots. They will go to either
        `ax.plot` or `ax.pcolormesh`.
        
    Returns
    -------
    axes : ndarray, shape (d, d)
        Array of subplots.
    
    To do
    -----
    * Option to plot d-dimensional histogram instead of a coordinate array.
    """
    # Default key word arguments.
    if kind =='scatter' or kind == 'scatter_density':
        plot_kws.setdefault('ms', 3)
        plot_kws.setdefault('color', 'black')
        plot_kws.setdefault('marker', '.')
        plot_kws.setdefault('markeredgecolor', 'none')
        plot_kws.setdefault('zorder', 5)
        plot_kws.setdefault('lw', 0)
        plot_kws['lw'] = 0
        if 'c' in plot_kws:
            plot_kws['color'] = plot_kws.pop('c')
        if 's' in plot_kws:
            plot_kws['ms'] = plot_kws.pop('s')
    elif kind == 'hist':
        plot_kws.setdefault('shading', 'auto')
        plot_kws.setdefault('bins', 'auto')
    if diag_kws is None:
        diag_kws = dict()
    diag_kws.setdefault('color', 'black')
    diag_kws.setdefault('histtype', 'step')
    diag_kws.setdefault('bins', 'auto')
    if autolim_kws is None:
        autolim_kws = dict()
    if rms_ellipse_kws is None:
        rms_ellipse_kws = dict()
        rms_ellipse_kws.setdefault('zorder', int(1e9))
    if rms_ellipse:
        Sigma = np.cov(X.T)
        rms_ellipse_kws.setdefault('2rms', True)
        if rms_ellipse_kws.pop('2rms'):
            Sigma *= 4.0
    if env_kws is None:
        env_kws = dict()
        env_kws.setdefault('zorder', int(1 + 1e9))

    # Create figure.
    n_points, n_dims = X.shape
    if figsize is None:
        f = n_dims * 7.5 / 6.0
        figsize = (1.025 * f, f)
    if limits is None:
        limits = auto_limits(X, **autolim_kws)
    if grid_kws is None:
        grid_kws = dict()
    grid_kws.setdefault('labels', ["x [mm]", "x' [mrad]",
                                   "y [mm]", "y' [mrad]",
                                   "z [m]", "dE [MeV]"])
    grid_kws.setdefault('limits', limits)
    grid_kws.setdefault('figsize', figsize)
    fig, axes = pair_grid(n_dims, **grid_kws)
        
    # Univariate plots.
    if smooth_hist:
        diag_kws.pop('histtype')
    bins = diag_kws.pop('bins')
    n_bins = []
    for i, ax in enumerate(axes.diagonal()):
        heights, edges = np.histogram(X[:, i], bins, limits[i])
        centers = utils.get_bin_centers(edges)
        n_bins.append(len(edges) - 1)
        if smooth_hist:
            ax.plot(centers, heights, **diag_kws)
        else:
            ax.hist(centers, len(centers), weights=heights, **diag_kws)
        
    # Take random sample.
    idx = np.arange(n_points)
    if samples is not None and samples < n_points:
        if type(samples) is float:
            n = int(samples * n_points)
        else:
            n = samples
        idx = utils.rand_rows(idx, n)
    
    # Bivariate plots.
    if kind == 'hist':
        bins = plot_kws.pop('bins')
    for i in range(1, len(axes)):
        for j in range(i):
            ax = axes[i, j]
            if kind == 'scatter':
                x, y = X[idx, j], X[idx, i]
                ax.plot(x, y, **plot_kws)
            elif kind == 'hist':
                x, y = X[:, j], X[:, i]
                if bins == 'auto':
                    Z, xedges, yedges = np.histogram2d(
                        x, y, (n_bins[j], n_bins[i]), (limits[j], limits[i]))
                else:
                    Z, xedges, yedges = np.histogram2d(
                        x, y, bins, (limits[j], limits[i]))
                if log:
                    Z = np.log10(Z + 1.0)
                if blur:
                    Z = filters.gaussian(Z, sigma=blur)
                if thresh is not None:
                    Z = np.ma.masked_less_equal(Z, thresh)
                xcenters = utils.get_bin_centers(xedges)
                ycenters = utils.get_bin_centers(yedges)
                ax.pcolormesh(xcenters, ycenters, Z.T, **plot_kws)
                
    # Ellipses
    scatter_axes = axes[1:, :-1]        
    if rms_ellipse:
        rms_ellipses(Sigma, axes=scatter_axes, **rms_ellipse_kws)
    if env_params is not None:
        corner_env(env_params, dims='all', axes=scatter_axes, **env_kws)
    
    # Reduce height of 1D histograms. 
    max_hist_height = 0.
    for ax in axes.diagonal():
        max_hist_height = max(max_hist_height, ax.get_ylim()[1])
    max_hist_height /= hist_height_frac
    for ax in axes.diagonal():
        ax.set_ylim(0, max_hist_height)
        
    return axes


def corner(
    data, 
    dtype='points',
    coords=None,
    labels=None, 
    kind='hist',
    diag_kind='line',
    frac_thresh=None,
    fig_kws=None, 
    diag_kws=None, 
    prof=False,
    prof_kws=None,
    cbar=False,
    return_fig=False,
    **plot_kws
):
    """Plot all 1D/2D projections in a matrix of subplots."""
    if data.ndim > 2:
        dtype = 'image'
    
    n = data.ndim
    if labels is None:
        labels = n * ['']
    if fig_kws is None:
        fig_kws = dict()
    fig_kws.setdefault(
        'figwidth', 
        1.5 * (n - 1 if diag_kind in ['None', 'none', None] else n),
    )
    fig_kws.setdefault('aligny', True)
    
    # Set default key word arguments.
    if diag_kws is None:
        diag_kws = dict()                
    if dtype == 'image':
        plot_kws.setdefault('ec', 'None')
        diag_kws.setdefault('color', 'black')
        if diag_kind == 'step':
            diag_kws.setdefault('drawstyle', 'steps-mid')
    elif dtype == 'points':
        if kind =='scatter' or kind == 'scatter_density':
            plot_kws.setdefault('s', 5)
            plot_kws.setdefault('color', 'black')
            plot_kws.setdefault('marker', '.')
            plot_kws.setdefault('ec', 'none')
        elif kind == 'hist':
            plot_kws.setdefault('bins', 'auto')
        diag_kws.setdefault('color', 'black')
        diag_kws.setdefault('histtype', 'step')
        diag_kws.setdefault('bins', 'auto')
    
    if coords is None:
        coords = [np.arange(s) for s in image.shape]
    
    if diag_kind is None or diag_kind.lower() == 'none':
        axes = _corner_nodiag(
            image, 
            coords=coords,
            labels=labels, 
            frac_thresh=frac_thresh,
            fig_kws=fig_kws, 
            prof=prof,
            prof_kws=prof_kws,
            return_fig=return_fig,
            cbar=cbar,
            **plot_kws
        )
        return axes
    
    fig, axes = pplt.subplots(
        nrows=n, ncols=n, sharex=1, sharey=1,
        spanx=False, spany=False, **fig_kws
    )
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.axis('off')
            elif i == j:
                h = utils.project(image, j)
                h = h / np.max(h)
                if diag_kind == 'line':
                    ax.plot(coords[j], h, **diag_kws)
                elif diag_kind == 'bar':
                    ax.bar(coords[j], **diag_kws)
                elif diag_kind == 'step':
                    ax.plot(coords[j], h, **diag_kws)
            else:
                if prof == 'edges':
                    profx = i == n - 1
                    profy = j == 0
                else:
                    profx = profy = prof
                H = utils.project(image, (j, i))
                plot_image(
                    H, ax=ax, x=coords[j], y=coords[i],
                    profx=profx, profy=profy, prof_kws=prof_kws, 
                    frac_thresh=frac_thresh, 
                    **plot_kws
                )
                
    for ax, label in zip(axes[-1, :], labels):
        ax.format(xlabel=label)
    for ax, label in zip(axes[1:, 0], labels[1:]):
        ax.format(ylabel=label)
    for i in range(n):
        axes[:-1, i].format(xticklabels=[])
        if i > 0:
            axes[i, 1:].format(yticklabels=[])
    xlims = [ax.get_xlim() for ax in axes[-1, :]]
    xlims[-1] = axes[-1, 0].get_ylim()
    for i, xlim in enumerate(xlims):
        axes[i, i].format(xlim=xlim)
    if return_fig:
        return fig, axes
    return axes


def _corner_nodiag(
    image, 
    coords=None,
    labels=None, 
    frac_thresh=None,
    fig_kws=None, 
    prof=False,
    prof_kws=None,
    return_fig=False,
    cbar=False,
    **plot_kws
):
    n = image.ndim
    if labels is None:
        labels = n * ['']
    if fig_kws is None:
        fig_kws = dict()
    fig_kws.setdefault('figwidth', 1.5 * (n - 1))
    fig_kws.setdefault('aligny', True)
    plot_kws.setdefault('ec', 'None')
    
    fig, axes = pplt.subplots(
        nrows=n-1, ncols=n-1, 
        spanx=False, spany=False, **fig_kws
    )
    for i in range(n - 1):
        for j in range(n - 1):
            ax = axes[i, j]
            if j > i:
                ax.axis('off')
                continue
            if prof == 'edges':
                profy = j == 0
                profx = i == n - 2
            else:
                profx = profy = prof
            H = utils.project(image, (j, i + 1))
            H = H / np.max(H)
            x = coords[j]
            y = coords[i + 1]
            if x.ndim > 1:
                axis = [k for k in range(x.ndim) if k not in (j, i + 1)]
                ind = len(axis) * [0]
                idx = utils.make_slice(x.ndim, axis, ind)
                x = x[idx]
                y = y[idx]
            #########
            if cbar:
                if i == 0 and j == 0:
                    plot_kws['colorbar'] = 't'
                    plot_kws['colorbar_kw'] = dict(width=0.065)
                else:
                    plot_kws['colorbar'] = False
            ########
            plot_image(H, ax=ax, x=x, y=y,
                       profx=profx, profy=profy, prof_kws=prof_kws, 
                       frac_thresh=frac_thresh, **plot_kws)
    for ax, label in zip(axes[-1, :], labels):
        ax.format(xlabel=label)
    for ax, label in zip(axes[:, 0], labels[1:]):
        ax.format(ylabel=label)
    if return_fig:
        return fig, axes
    return axes




def _corner_no_diag(axes, X, kind, thresh, blur, idx, n_bins, limits,
                    env_params, env_kws, Sigma, rms_ellipse_kws, **plot_kws):
    """Helper function for `corner`."""
    if kind == 'hist':
        bins = plot_kws.pop('bins')
    for i in range(1, len(axes) + 1):
        for j in range(i + 1):
            ax = axes[i - 1, j]
            if kind == 'scatter':
                x, y = X[idx, j], X[idx, i]
                ax.plot(x, y, **plot_kws)
            elif kind == 'hist':
                x, y = X[:, j], X[:, i]
                if bins == 'auto':
                    Z, xedges, yedges = np.histogram2d(
                        x, y, (n_bins[j], n_bins[i]), (limits[j], limits[i]))
                else:
                    Z, xedges, yedges = np.histogram2d(
                        x, y, bins, (limits[j], limits[i]))
                if blur:
                    Z = filters.gaussian(Z, sigma=blur)
                if thresh:
                    Z = np.ma.masked_less_equal(Z, thresh)
                xcenters = utils.get_bin_centers(xedges)
                ycenters = utils.get_bin_centers(yedges)
                ax.pcolormesh(xcenters, ycenters, Z.T, **plot_kws)
    if Sigma is not None:
        rms_ellipses(Sigma, axes=axes, **rms_ellipse_kws)
    if env_params is not None:
        corner_env(env_params, dims='all', axes=axes, **env_kws)
    return axes
    
    
def ellipse(ax, c1, c2, angle=0.0, center=(0, 0), **plt_kws):
    """Plot ellipse with semi-axes `c1` and `c2`. Angle is given in radians
    and is measured below the x axis."""
    plt_kws.setdefault('fill', False)
    return ax.add_patch(Ellipse(center, 2*c1, 2*c2, -np.degrees(angle), **plt_kws))


def rms_ellipses(
    Sigmas, axes=None, figsize=(5, 5), pad=0.5, constrained_layout=True,
    cmap=None, cmap_range=(0, 1), centers=None, return_artists=False, 
    return_fig=False, 
    **plt_kws
):
    """Plot rms ellipse parameters directly from covariance matrix."""
    Sigmas = np.array(Sigmas)
    if Sigmas.ndim == 2:
        Sigmas = Sigmas[np.newaxis, :, :]
    if axes is None:
        x2_max, y2_max = np.max(Sigmas[:, 0, 0]), np.max(Sigmas[:, 2, 2])
        xp2_max, yp2_max = np.max(Sigmas[:, 1, 1]), np.max(Sigmas[:, 3, 3])
        umax = (1 + pad) * np.sqrt(max(x2_max, y2_max))
        upmax = (1 + pad) * np.sqrt(max(xp2_max, yp2_max))
        limits = 2 * [(-umax, umax), (-upmax, upmax)]
        fig, axes = pair_grid_nodiag(4, figsize, limits, constrained_layout=constrained_layout)

    colors = None
    if len(Sigmas) > 1 and cmap is not None:
        start, end = cmap_range
        colors = [cmap(i) for i in np.linspace(start, end, len(Sigmas))]
        
    if centers is None:
        centers = 4 * [0.0]
    
    dims = {0:'x', 1:'xp', 2:'y', 3:'yp'}
    artists = []
    for l, Sigma in enumerate(Sigmas):
        for i in range(3):
            for j in range(i + 1):
                angle, c1, c2 = rms_ellipse_dims(Sigma, dims[j], dims[i + 1])
                if colors is not None:
                    plt_kws['color'] = colors[l]
                xcenter = centers[j]
                ycenter = centers[i + 1]
                artist = ellipse(axes[i, j], c1, c2, angle,
                             center=(xcenter, ycenter), **plt_kws)
                artists.append(artist)
    items = []
    if return_fig:
        items.append(fig)
    items.append(axes)
    if return_artists:
        return items.append(artists)
    if len(items) == 1:
        items = items[0]
    return items
