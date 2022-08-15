import numpy as np
from scipy import optimize as opt
from matplotlib import pyplot as plt
import proplot as pplt
from plotly import graph_objects as go
from ipywidgets import interactive
from ipywidgets import widgets

from . import utils


def linear_fit(x, y):
    def fit(x, slope, intercept):
        return slope * x + intercept
    
    popt, pcov = opt.curve_fit(fit, x, y)
    slope, intercept = popt
    yfit = fit(x, *popt)
    return yfit, slope, intercept


def plot_profile(image, xcoords=None, ycoords=None, ax=None, 
                 profx=True, profy=True, kind='line', scale=0.12, **plot_kws): 
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
    x1 = xcoords
    y1 = ycoords[0] + scale * np.abs(ycoords[-1] - ycoords[0]) * px
    x2 = xcoords[0] + scale * np.abs(xcoords[-1] - xcoords[0]) * py
    y2 = ycoords
    for i, (x, y) in enumerate(zip([x1, x2], [y1, y2])):
        if i == 0 and not profx:
            continue
        if i == 1 and not profy:
            continue
        if kind == 'line':
            ax.plot(x, y, **plot_kws)
        elif kind == 'bar':
            if i == 0:
                ax.bar(x, y, **plot_kws)
            else:
                ax.barh(y, x, **plot_kws)
        elif kind == 'step':
            ax.plot(x, y, drawstyle='steps-mid', **plot_kws)
    return ax


def prepare_for_log_norm(image, method='floor'):
    if np.all(image > 0):
        return image
    if method == 'floor':
        floor = 1e-12
        if np.max(image) > 0:
            floor = np.min(image[image > 0])
        return image + floor
    elif method == 'mask':
        return np.ma.masked_less_equal(image, 0)


def plot_image(
    image, 
    x=None, 
    y=None, 
    ax=None, 
    profx=False, 
    profy=False, 
    prof_kws=None, 
    fill_value=None,
    frac_thresh=None, 
    contour=False, 
    contour_kws=None,
    return_mesh=False, 
    handle_log='mask',  # {'floor', 'mask'}
    **plot_kws
):
    plot_kws.setdefault('ec', 'None')
    log = 'norm' in plot_kws and plot_kws['norm'] == 'log'
    if log:
        if 'colorbar' in plot_kws and plot_kws['colorbar']:
            if 'colorbar_kw' not in plot_kws:
                plot_kws['colorbar_kw'] = dict()
            plot_kws['colorbar_kw']['formatter'] = 'log'
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
    if fill_value is not None:
        image = np.ma.filled(image, fill_value=fill_value)
    image_max = np.max(image)
    if frac_thresh is not None:
        floor = max(1e-12, frac_thresh * image_max)
        image[image < floor] = 0
    if log:
        image = prepare_for_log_norm(image, method=handle_log)
    mesh = ax.pcolormesh(x, y, image.T, **plot_kws)
    if contour:
        ax.contour(x, y, image.T, **contour_kws)
    plot_profile(image, xcoords=x, ycoords=y, ax=ax, 
                 profx=profx, profy=profy, **prof_kws)
    if return_mesh:
        return ax, mesh
    else:
        return ax


def corner(
    image, 
    coords=None,
    labels=None, 
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
    """Plot all 1D/2D projections in a matrix of subplots.
    
    To do: 
    
    Clean this up and merge with `scdist.tools.plotting.corner`, 
    which performs binning first. I believe in scdist I also found
    a nicer way to handle diag on/off. (One function that plots
    the off-diagonals, recieving axes as an argument.
    """
    n = image.ndim
    if labels is None:
        labels = n * ['']
    if fig_kws is None:
        fig_kws = dict()
    fig_kws.setdefault('figwidth', 1.5 * (n - 1 if diag_kind in ['None', 'none', None] else n))
    fig_kws.setdefault('aligny', True)
    if diag_kws is None:
        diag_kws = dict()
    diag_kws.setdefault('color', 'black')
    if diag_kind == 'step':
        diag_kws.setdefault('drawstyle', 'steps-mid')
    plot_kws.setdefault('ec', 'None')
    
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


def interactive_proj2d(
    f, 
    coords=None,
    default_ind=(0, 1),
    slice_type='int',  # {'int', 'range'}
    dims=None,
    units=None,
    prof_kws=None,
    cmaps=None,
    handle_log='floor',
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
    slice_type : {'int', 'range'}
        Whether to slice one index along the axis or a range of indices.
    dims : list[str], shape (n,)
        Dimension names.
    units : list[str], shape (n,)
        Dimension units.
    prof_kws : dict
        Key word arguments for 1D profile plots.
    cmaps : list
        Color map options.
    handle_log : {'floor', 'mask'}
        See `plot_image`.
    
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
    plot_kws['handle_log'] = handle_log
    
    # Widgets
    handle_log = widgets.Dropdown(options=['floor', 'mask'], description='handle_log')
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
        if slice_type == 'int':
            slider = widgets.IntSlider(
                min=0, max=f.shape[k], value=f.shape[k]//2,
                description=dims[k], 
                continuous_update=True,
            )
        elif slice_type == 'range':
            slider = widgets.IntRangeSlider(
                value=(0, f.shape[k]), min=0, max=f.shape[k],
                description=dims[k], 
                continuous_update=True,
            )
        else:
            raise ValueError('Invalid `slice_type`.')
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
        handle_log, cmap, log, profiles, fix_vmax, vmax,
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
        return _plot_figure(dim1, dim2, checks, sliders, log, profiles, 
                            thresh, cmap, handle_log, fix_vmax, vmax)

    def _update4(
        handle_log, cmap, log, profiles, fix_vmax, vmax,
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
        return _plot_figure(dim1, dim2, checks, sliders, log, profiles, 
                            thresh, cmap, handle_log, fix_vmax, vmax)

    def _update5(
        handle_log, cmap, log, profiles, fix_vmax, vmax,
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
        return _plot_figure(dim1, dim2, checks, sliders, log, profiles, 
                            thresh, cmap, handle_log, fix_vmax, vmax)

    def _update6(
        handle_log, cmap, log, profiles, fix_vmax, vmax,
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
        return _plot_figure(dim1, dim2, checks, sliders, log, profiles,
                            thresh, cmap, handle_log, fix_vmax, vmax)

    update = {
        3: _update3,
        4: _update4,
        5: _update5,
        6: _update6,
    }[n]
    
    def _plot_figure(dim1, dim2, checks, sliders, log, profiles, thresh, cmap, handle_log, fix_vmax, vmax):
        if (dim1 == dim2):
            return
        axis_view = [dims.index(dim) for dim in (dim1, dim2)]
        axis_slice = [dims.index(dim) for dim, check in zip(dims, checks) if check]
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
            'handle_log': handle_log,
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
    kws['handle_log'] = handle_log
    gui = interactive(update, **kws)
    return gui


def interactive_proj1d(
    f, 
    coords=None,
    default_ind=0,
    slice_type='int',  # {'int', 'range'}
    dims=None,
    units=None,
    kind='bar',
    **plot_kws,
):
    """Interactive plot of 1D projection of distribution `f`.
    
    The distribution is projected onto the specified axis. Sliders provide the
    option to slice the distribution before projecting.
    
    Parameters
    ----------
    f : ndarray
        An n-dimensional array.
    coords : list[ndarray]
        Grid coordinates for each dimension.
    default_ind : int
        Default index to plot.
    slice_type : {'int', 'range'}
        Whether to slice one index along the axis or a range of indices.
    dims : list[str], shape (n,)
        Dimension names.
    units : list[str], shape (n,)
        Dimension units.
    kind : {'bar', 'line'}
        The kind of plot to draw.
    **plot_kws
        Key word arguments passed to 1D plotting function.
        
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
    plot_kws.setdefault('color', 'black')
    
    # Widgets
    dim1 = widgets.Dropdown(options=dims, index=default_ind, description='dim')
    
    # Sliders
    sliders, checks = [], []
    for k in range(n):
        if slice_type == 'int':
            slider = widgets.IntSlider(
                min=0, max=f.shape[k], value=f.shape[k]//2,
                description=dims[k], 
                continuous_update=True,
            )
        elif slice_type == 'range':
            slider = widgets.IntRangeSlider(
                value=(0, f.shape[k]), min=0, max=f.shape[k],
                description=dims[k], 
                continuous_update=True,
            )
        else:
            raise ValueError('Invalid `slice_type`.')
        slider.layout.display = 'none'
        sliders.append(slider)
        checks.append(widgets.Checkbox(description=f'slice {dims[k]}'))
        
    # Hide/show sliders.
    def hide(button):
        for k in range(n):
            # Hide elements for dimensions being plotted.
            valid = dims[k] != dim1.value
            disp = None if valid else 'none'
            for element in [sliders[k], checks[k]]:
                element.layout.display = disp
            # Uncheck boxes for dimensions being plotted. 
            if not valid and checks[k].value:
                checks[k].value = False
            # Make sliders respond to check boxes.
            if not checks[k].value:
                sliders[k].layout.display = 'none'
                    
    for element in (dim1, *checks):
        element.observe(hide, names='value')
    # Initial hide
    for k in range(n):
        if k == default_ind:
            checks[k].layout.display = 'none'
            sliders[k].layout.display = 'none'
                
    # I don't know how else to do this.
    def _update2(
        dim1, 
        check1, check2,
        slider1, slider2,
    ):
        checks = [check1, check2]
        sliders = [slider1, slider2]
        for dim, check in zip(dims, checks):
            if check and dim == dim1:
                return
        return _plot_figure(dim1, checks, sliders)

    def _update3(
        dim1, 
        check1, check2, check3,
        slider1, slider2, slider3,
    ):
        checks = [check1, check2, check3]
        sliders = [slider1, slider2, slider3]
        for dim, check in zip(dims, checks):
            if check and dim == dim1:
                return
        return _plot_figure(dim1, checks, sliders)
    
    def _update4(
        dim1, 
        check1, check2, check3, check4,
        slider1, slider2, slider3, slider4,
    ):
        checks = [check1, check2, check3, check4]
        sliders = [slider1, slider2, slider3, slider4]
        for dim, check in zip(dims, checks):
            if check and dim == dim1:
                return
        return _plot_figure(dim1, checks, sliders)

    def _update5(
        dim1, 
        check1, check2, check3, check4, check5,
        slider1, slider2, slider3, slider4, slider5,
    ):
        checks = [check1, check2, check3, check4, check5]
        sliders = [slider1, slider2, slider3, slider4, slider5]
        for dim, check in zip(dims, checks):
            if check and dim == dim1:
                return
        return _plot_figure(dim1, checks, sliders)
    
    def _update6(
        dim1, 
        check1, check2, check3, check4, check5, check6,
        slider1, slider2, slider3, slider4, slider5, slider6,
    ):
        checks = [check1, check2, check3, check4, check5, check6]
        sliders = [slider1, slider2, slider3, slider4, slider5, slider6]
        for dim, check in zip(dims, checks):
            if check and dim == dim1:
                return
        return _plot_figure(dim1, checks, sliders)

    update_dict = {
        2: _update2,
        3: _update3,
        4: _update4,
        5: _update5,
        6: _update6,
    }
    update = update_dict[n]
    
    def _plot_figure(dim1, checks, sliders):
        axis_view = dims.index(dim1)
        axis_slice = [dims.index(dim) for dim, check in zip(dims, checks) if check]
        ind = sliders
        for k in range(n):
            if type(ind[k]) is int:
                ind[k] = (ind[k], ind[k] + 1)
        ind = [ind[k] for k in axis_slice]
        _f = f[utils.make_slice(f.ndim, axis_slice, ind)]
        p = utils.project(_f, axis_view)

        fig, ax = pplt.subplots(figsize=(4.5, 1.5))
        ax.format(xlabel=dims_units[axis_view])
        x = coords[axis_view]
        y = p / np.sum(p)
        if kind == 'bar':
            ax.bar(x, y, **plot_kws)
        elif kind == 'line':
            ax.plot(x, y, **plot_kws)
        elif kind == 'step':
            ax.plot(x, y, drawstyle='steps-mid', **plot_kws)
        plt.show()
        
    kws = dict()
    kws['dim1'] = dim1
    for i, check in enumerate(checks, start=1):
        kws[f'check{i}'] = check
    for i, slider in enumerate(sliders, start=1):
        kws[f'slider{i}'] = slider
    gui = interactive(update, **kws)
    return gui



# Plotly
# --------------------------------------------------------------------
def plotly_wire(data=None, x=None, y=None, layout=None, uaxis=None):
    Z = data
    if x is None:
        x = np.arange(Z.shape[0])
    if y is None:
        y = np.arange(Z.shape[1])
    X, Y = np.meshgrid(x, y, indexing='ij')    
    lines = []
    line_marker = dict(color='black', width=3)
    for x, y, z in zip(X, Y, Z):
        lines.append(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=line_marker))

    if uaxis is None:
        uaxis= dict(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)',
        )
    if layout is None:
        layout = go.Layout(
            width=500,
            height=500,
            showlegend=False,
        )
    fig = go.Figure(data=lines, layout=layout)
    fig.update_layout(
        scene=dict(
            xaxis=uaxis, 
            yaxis=uaxis,
            zaxis=uaxis,
        ),
    )
    return fig