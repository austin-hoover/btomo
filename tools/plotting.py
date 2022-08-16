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
from ipywidgets import widgets
from ipywidgets import interactive

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
    X, kind='hist', figsize=None, limits=None, diag_height_frac=0.6,
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
    diag_height_frac : float
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
    max_height = 0.
    for ax in axes.diagonal():
        max_height = max(max_height, ax.get_ylim()[1])
    max_height /= diag_height_frac
    for ax in axes.diagonal():
        ax.set_ylim(0, max_height)
        
    return axes
    


# Images
# ------------------------------------------------------------------------------
def prep_image_for_log(image, handle_log='floor'):
    image_max = np.max(image)
    if np.any(image <= 0):
        if handle_log == 'floor':
            floor = 1e-12
            if image_max > 0:
                floor = np.min(image[image > 0])
            image = image + floor
        elif handle_log == 'mask':
            image = np.ma.masked_less_equal(image, 0)
    return image


def plot1d(x, y, ax=None, flipxy=False, kind='step', **kws):
    funcs = {
        'line': ax.plot,
        'bar': ax.bar,
        'step': ax.plot,
    }
    if kind == 'step':
        kws.setdefault('drawstyle', 'steps-mid')
    if flipxy:
        x, y = y, x
        funcs['bar'] = ax.barh
    return funcs[kind](x, y, **kws)
    

def plot_profile(image, xcoords=None, ycoords=None, ax=None, profx=True, profy=True, 
                 kind='step', scale=0.12, **plot_kws):
    """Plot 1D projection of image along axis."""
    if xcoords is None:
        xcoords = np.arange(image.shape[1])
    if ycoords is None:
        ycoords = np.arange(image.shape[0])
    plot_kws.setdefault('lw', 0.75)
    plot_kws.setdefault('color', 'white')
    
    def _normalize(profile):
        pmax = np.max(profile)
        if pmax > 0:
            profile = profile / pmax
        return profile
    
    px, py = [_normalize(np.sum(image, axis=i)) for i in (1, 0)]
    yy = ycoords[0] + scale * np.abs(ycoords[-1] - ycoords[0]) * px
    xx = xcoords[0] + scale * np.abs(xcoords[-1] - xcoords[0]) * py
    for i, (x, y) in enumerate(zip([xcoords, ycoords], [yy, xx])):
        if i == 0 and not profx:
            continue
        if i == 1 and not profy:
            continue
        plot1d(x, y, ax=ax, flipxy=i, kind=kind, **plot_kws)
    return ax


def plot_image(
    image, 
    x=None, 
    y=None, 
    ax=None, 
    profx=False, 
    profy=False, 
    prof_kws=None, 
    frac_thresh=None, 
    contour=False, 
    contour_kws=None,
    return_mesh=False, 
    handle_log='mask', 
    **plot_kws
):
    """Plot 2D image."""
    log = 'norm' in plot_kws and plot_kws['norm'] == 'log'
    if log:
        if 'colorbar' in plot_kws and plot_kws['colorbar']:
            if 'colorbar_kw' not in plot_kws:
                plot_kws['colorbar_kw'] = dict()
            plot_kws['colorbar_kw']['formatter'] = 'log'
        image = prep_image_for_log(image, handle_log)
    if contour and contour_kws is None:
        contour_kws = dict()
        contour_kws.setdefault('color', 'white')
        contour_kws.setdefault('lw', 1.0)
        contour_kws.setdefault('alpha', 0.5)
    if prof_kws is None:
        prof_kws = dict()
    if x is None:
        x = np.arange(image.shape[0])
    if y is None:
        y = np.arange(image.shape[1])
    if x.ndim == 2:
        x = x.T
    if y.ndim == 2:
        y = y.T
    mesh = ax.pcolormesh(x, y, image.T, **plot_kws)
    if contour:
        ax.contour(x, y, image.T, **contour_kws)
    plot_profile(image, xcoords=x, ycoords=y, ax=ax, profx=profx, profy=profy, **prof_kws)
    if return_mesh:
        return ax, mesh
    else:
        return ax

    
def _setup_corner(n, diag, labels, limits=None, **fig_kws):
    if labels is None:
        labels = n * ['']
    nrows = ncols = n if diag else n - 1 
    fig_kws.setdefault('figwidth', 1.5 * nrows)
    fig_kws.setdefault('aligny', True)
    fig, axes = pplt.subplots(nrows=nrows, ncols=ncols, sharex=1, sharey=1, 
                              spanx=False, spany=False, **fig_kws)
    for i in range(nrows):
        for j in range(ncols):
            if j > i:
                axes[i, j].axis('off')
    for ax, label in zip(axes[-1, :], labels):
        ax.format(xlabel=label)
    start = 1 if diag else 0
    for ax, label in zip(axes[start:, 0], labels[start:]):
        ax.format(ylabel=label)
    for i in range(nrows):
        axes[:-1, i].format(xticklabels=[])
        axes[i, 1:].format(yticklabels=[])
    for ax in axes:
        ax.format(xspineloc='bottom', yspineloc='left')
    if diag:
        for i in range(n):
            axes[i, i].format(yspineloc='neither')
    if limits is not None:
        for i in range(n):
            axes[:, i].format(xlim=limits[i])
            if i >= start:
                axes[i, :].format(ylim=limits[i])
    return fig, axes


def corner(
    data, 
    kind='hist',
    diag_kind='step',
    coords=None,
    limits=None,
    labels=None, 
    samples=None,
    autolim_kws=None, 
    diag_kws=None, 
    fig_kws=None, 
    prof=False,
    prof_kws=None,
    return_fig=False,
    diag_height_frac=0.8,
    **plot_kws
):
    # Figure out if data is discrete points or N-D image
    n = data.ndim
    pts = False
    if n == 2:
        n = data.shape[1]
        pts = True
    diag = diag_kind in ['line', 'bar', 'step']
    start = 1 if diag else 0
    if diag_kws is None:
        diag_kws = dict()
    diag_kws.setdefault('kind', 'step')
    diag_kws.setdefault('color', 'black')
    diag_kws.setdefault('lw', 1.0)
    if pts and kind =='scatter':
        plot_kws.setdefault('s', 6)
        plot_kws.setdefault('c', 'black')
        plot_kws.setdefault('marker', '.')
        plot_kws.setdefault('ec', 'none')
        if 'color' in plot_kws:
            plot_kws['c'] = plot_kws.pop('color')
        if 'ms' in plot_kws:
            plot_kws['s'] = plot_kws.pop('ms')
    elif (pts and kind == 'hist') or not pts:
        plot_kws.setdefault('ec', 'None')
        
    # Create the figure.
    if autolim_kws is None:
        autolim_kws = dict()  
    if fig_kws is None:
        fig_kws = dict()
    if limits is None and pts:
        limits = auto_limits(data, **autolim_kws)
    fig, axes = _setup_corner(n, diag, labels, limits, **fig_kws)
    
    # Discrete points
    if pts:    
        # Univariate plots
        bins = 'auto'
        if 'bins' in plot_kws:
            bins = plot_kws.pop('bins')
        edges, centers = [], []
        for i in range(n):
            heights, _edges = np.histogram(data[:, i], bins, limits[i])
            _centers = utils.get_bin_centers(_edges)
            edges.append(_edges)
            centers.append(_centers)
            if diag:
                plot1d(_centers, heights, ax=axes[i, i], **diag_kws)

        # Take random sample.
        idx = np.arange(data.shape[0])
        if samples is not None and samples < data.shape[0]:
            if type(samples) is float:
                N = int(samples * data.shape[0])
            else:
                N = samples
            idx = utils.rand_rows(idx, N)

        # Bivariate plots
        for ii, i in enumerate(range(start, axes.shape[0])):
            for j in range(ii + 1):
                ax = axes[i, j]
                if kind == 'scatter':
                    ax.scatter(data[idx, j], data[idx, ii + 1], **plot_kws)
                elif kind == 'hist':
                    _im, _, _ = np.histogram2d(data[:, j], data[:, ii + 1], (edges[j], edges[ii + 1]))
                    if prof == 'edges':
                        profy = j == 0
                        profx = i == axes.shape[0] - 1
                    else:
                        profx = profy = prof
                    plot_image(
                        _im, 
                        x=centers[j],
                        y=centers[ii + 1],
                        ax=ax,
                        profx=profx, profy=profy, prof_kws=prof_kws, 
                        **plot_kws
                    )       
    # N-D image
    else:
        if coords is None:
            coords = [np.arange(s) for s in data.shape]
            
        # Plot off-diagonal.
        for ii, i in enumerate(range(start, axes.shape[0])):
            for j in range(ii + 1):
                ax = axes[i, j]
                if prof == 'edges':
                    profy = j == 0
                    profx = i == axes.shape[0] - 1
                else:
                    profx = profy = prof
                plot_image(
                    utils.project(data, (j, ii + 1)), 
                    x=coords[j], y=coords[ii + 1], ax=ax, 
                    profx=profx, profy=profy, prof_kws=prof_kws, 
                    **plot_kws
                )             
        # Plot diagonal.
        if diag:
            for i in range(n):
                h = utils.project(data, j)
                plot1d(coords[i], h / np.max(h), ax=axes[i, i], **diag_kws)

    # Modify diagonal y axis limits.
    if diag:
        max_height = 0.
        for i in range(n):
            max_height = max(max_height, axes[i, i].get_ylim()[1])
        max_height /= diag_height_frac
        for i in range(n):
            axes[i, i].set_ylim(0, max_height)

    if return_fig:
        return fig, axes
    return axes




# Interactive
# ------------------------------------------------------------------------------
def interactive_proj2d(
    f, 
    coords=None,
    default_ind=(0, 1),
    slider_type='int',  # {'int', 'range'}
    dims=None,
    units=None,
    prof_kws=None,
    cmaps=None,
    **plot_kws,
):
    """Interactive plot of 2D projection of distribution `f`.
    
    The distribution is projected onto the specified axes. Sliders provide the
    option to slice the distribution before projecting.
    
    Parameters
    ----------
    f : ndarray
        An n-dimensional array.
    coords : list[ndarray]
        Coordinate arrays along each dimension. A square grid is assumed.
    default_ind : (i, j)
        Default x and y index to plot.
    slider_type : {'int', 'range'}
        Whether to slice one index along the axis or a range of indices.
    dims : list[str], shape (n,)
        Dimension names.
    units : list[str], shape (n,)
        Dimension units.
    prof_kws : dict
        Key word arguments for 1D profile plots.
    cmaps : dict
    
    Returns
    -------
    gui : ipywidgets.widgets.interaction.interactive
        This widget can be displayed by calling `IPython.display.display(gui)`. 
    """
    n = f.ndim
    if coords is None:
        coords = [np.arange(f.shape[k]) for k in range(n)]
    
    if dims is None:
        dims = [f'x{i + 1}' for i in range(n)]
    if units is None:
        units = n * ['']
    dims_units = []
    for dim, unit in zip(dims, units):
        dims_units.append(f'{dim}' + f' [{unit}]' if unit != '' else dim)
    dim_to_int = {dim: i for i, dim in enumerate(dims)}
    if prof_kws is None:
        prof_kws = dict()
    prof_kws.setdefault('lw', 1.0)
    prof_kws.setdefault('alpha', 0.5)
    prof_kws.setdefault('color', 'white')
    prof_kws.setdefault('scale', 0.14)
    if cmaps is None:
        cmaps = ['viridis', 'dusk_r', 'mono_r', 'plasma']
    plot_kws.setdefault('colorbar', True)
    plot_kws['prof_kws'] = prof_kws
    
    # Widgets
    cmap = widgets.Dropdown(options=cmaps, description='cmap')
    thresh = widgets.FloatSlider(value=-5.0, min=-8.0, max=0.0, step=0.1, 
                                 description='thresh', continuous_update=True)
    discrete = widgets.Checkbox(value=False, description='discrete')
    log = widgets.Checkbox(value=False, description='log')
    contour = widgets.Checkbox(value=False, description='contour')
    profiles = widgets.Checkbox(value=True, description='profiles')
    scale = widgets.FloatSlider(value=0.15, min=0.0, max=1.0, step=0.01, description='scale',
                                continuous_update=True)
    dim1 = widgets.Dropdown(options=dims, index=default_ind[0], description='dim 1')
    dim2 = widgets.Dropdown(options=dims, index=default_ind[1], description='dim 2')
    vmax = widgets.FloatSlider(value=1.0, min=0.0, max=1.0, step=0.01, description='vmax',
                               continuous_update=True)
    fix_vmax = widgets.Checkbox(value=False, description='fix vmax')
    
    # Sliders
    sliders, checks = [], []
    for k in range(n):
        if slider_type == 'int':
            slider = widgets.IntSlider(
                min=0, max=f.shape[k], value=f.shape[k]//2,
                description=dims[k], 
                continuous_update=True,
            )
        elif slider_type == 'range':
            slider = widgets.IntRangeSlider(
                value=(0, f.shape[k]), min=0, max=f.shape[k],
                description=dims[k], 
                continuous_update=True,
            )
        else:
            raise ValueError('Invalid `slider_type`.')
        slider.layout.display = 'none'
        sliders.append(slider)
        checks.append(widgets.Checkbox(description=f'slice {dims[k]}'))
        
    # Hide/show sliders.
    def hide(button):
        for k in range(n):
            # Hide elements for dimensions being plotted.
            valid = dims[k] not in (dim1.value, dim2.value)
            disp = None if valid else 'none'
            for element in [sliders[k], checks[k]]:
                element.layout.display = disp
            # Uncheck boxes for dimensions being plotted. 
            if not valid and checks[k].value:
                checks[k].value = False
            # Make sliders respond to check boxes.
            if not checks[k].value:
                sliders[k].layout.display = 'none'
        # Hide vmax slider if fix_vmax checkbox is not checked.
        vmax.layout.display = None if fix_vmax.value else 'none' 
                    
    for element in (dim1, dim2, *checks, fix_vmax):
        element.observe(hide, names='value')
    # Initial hide
    for k in range(n):
        if k in default_ind:
            checks[k].layout.display = 'none'
            sliders[k].layout.display = 'none'
    vmax.layout.display = 'none'
                
    # I don't know how else to do this.
    def _update3(
        cmap, log, profiles, fix_vmax, vmax,
        dim1, dim2, 
        check1, check2, check3,
        slider1, slider2, slider3,
        thresh, 
    ):
        checks = [check1, check2, check3]
        sliders = [slider1, slider2, slider3]
        for dim, check in zip(dims, checks):
            if check and dim in (dim1, dim2):
                return
        return _plot_figure(dim1, dim2, checks, sliders, log, profiles, thresh, cmap, fix_vmax, vmax)

    def _update4(
        cmap, log, profiles, fix_vmax, vmax,
        dim1, dim2, 
        check1, check2, check3, check4, 
        slider1, slider2, slider3, slider4,
        thresh,
    ):
        checks = [check1, check2, check3, check4]
        sliders = [slider1, slider2, slider3, slider4]
        for dim, check in zip(dims, checks):
            if check and dim in (dim1, dim2):
                return
        return _plot_figure(dim1, dim2, checks, sliders, log, profiles, thresh, cmap, fix_vmax, vmax)

    def _update5(
        cmap, log, profiles, fix_vmax, vmax,
        dim1, dim2, 
        check1, check2, check3, check4, check5,
        slider1, slider2, slider3, slider4, slider5,
        thresh,
    ):
        checks = [check1, check2, check3, check4, check5]
        sliders = [slider1, slider2, slider3, slider4, slider5]
        for dim, check in zip(dims, checks):
            if check and dim in (dim1, dim2):
                return
        return _plot_figure(dim1, dim2, checks, sliders, log, profiles, thresh, cmap, fix_vmax, vmax)

    def _update6(
        cmap, log, profiles, fix_vmax, vmax,
        dim1, dim2, 
        check1, check2, check3, check4, check5, check6,
        slider1, slider2, slider3, slider4, slider5, slider6,
        thresh,
    ):
        checks = [check1, check2, check3, check4, check5, check6]
        sliders = [slider1, slider2, slider3, slider4, slider5, slider6]
        for dim, check in zip(dims, checks):
            if check and dim in (dim1, dim2):
                return
        return _plot_figure(dim1, dim2, checks, sliders, log, profiles, thresh, cmap, fix_vmax, vmax)

    update = {
        3: _update3,
        4: _update4,
        5: _update5,
        6: _update6,
    }[n]
    
    def _plot_figure(dim1, dim2, checks, sliders, log, profiles, thresh, cmap, fix_vmax, vmax):
        if (dim1 == dim2):
            return
        axis_view = [dim_to_int[dim] for dim in (dim1, dim2)]
        axis_slice = [dim_to_int[dim] for dim, check in zip(dims, checks) if check]
        ind = sliders
        for k in range(n):
            if type(ind[k]) is int:
                ind[k] = (ind[k], ind[k] + 1)
        ind = [ind[k] for k in axis_slice]
        H = f[utils.make_slice(f.ndim, axis_slice, ind)]
        H = utils.project(H, axis_view)
        plot_kws.update({
            'profx': profiles,
            'profy': profiles,
            'cmap': cmap,
            'frac_thresh': 10.0**thresh,
            'norm': 'log' if log else None,
            'vmax': vmax if fix_vmax else None,
        })
        fig, ax = pplt.subplots()
        plot_image(H, x=coords[axis_view[0]], y=coords[axis_view[1]], ax=ax, **plot_kws)
        ax.format(xlabel=dims_units[axis_view[0]], ylabel=dims_units[axis_view[1]])
        plt.show()
        
    kws = dict()
    kws['dim1'] = dim1
    kws['dim2'] = dim2
    for i, check in enumerate(checks, start=1):
        kws[f'check{i}'] = check
    for i, slider in enumerate(sliders, start=1):
        kws[f'slider{i}'] = slider
    kws['log'] = log
    kws['profiles'] = profiles
    kws['thresh'] = thresh
    kws['fix_vmax'] = fix_vmax
    kws['vmax'] = vmax
    kws['cmap'] = cmap
    gui = interactive(update, **kws)
    return gui