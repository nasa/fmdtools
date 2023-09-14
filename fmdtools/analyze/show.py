# -*- coding: utf-8 -*-
"""
Module for showing the physical 2d/3d properties of the model (mainly focused on
grids, environments, trajectories, etc).

Uses the following primary methods:

- :func: `grid`: Plots a Grid collections/properties on an x-y grid.
- :func: `grid3d`: Plots a Grid collection.property on an x-y grid.
- :func: `trajectories`: Plots trajectories from a history.

And secondary methods:

- :func:`grid_property`: Plots a given Grid property on an x-y grid.
- :func:`grid_property3d`: Plots a given Grid property in 3d space.
- :func: `grid_collection`: Plots a given Grid collections on an x-y grid.

And helper functions:

- :func:`consolidate_legend`: For plotting legends of grids.
"""
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle
from matplotlib import colormaps, cm
from mpl_toolkits.mplot3d import art3d
from matplotlib.colors import to_rgba
import numpy as np
from .plot import prep_hists


def grid_property(grd, prop, xlab="x", ylab="y", proplab="prop", **kwargs):
    """
    Plots a given property 'prop' as a colormesh on an x-y grid.
    See matplotlib.pyplot.pcolormesh.

    Parameters
    ----------
    grd: Grid
        define.environment.grid to plot
    prop : str
        Name of the property to plot.
    xlab : str, optional
        Label for x-axis. The default is "x".
    ylab : str, optional
        Label for y-axis. The default is "y".
    proplab : str, optional
        Label for the property. The default is "prop", which uses the name of the
        property provided.
    **kwargs : kwargs
        Keyword arguments to matplotlib.pyplot.pcolormesh (e.g., cmap, edgecolors)

    Returns
    -------
    fig : mpl.figure
        Plotted figure object
    ax : mpl.axis
        Ploted axis object.
    """
    default_kwargs = dict(edgecolors='black', cmap="Greens")
    kwargs = {**kwargs, **default_kwargs}

    fig, ax = plt.subplots(1)

    p = getattr(grd, prop)
    # im = ax.matshow(p, **kwargs)
    offset = grd.p.blocksize/2
    x = np.linspace(0., grd.p.blocksize*(grd.p.x_size-1), grd.p.x_size)
    y = np.linspace(0., grd.p.blocksize*(grd.p.y_size-1), grd.p.y_size)
    X, Y = np.meshgrid(x, y)

    im = ax.pcolormesh(X, Y, p.swapaxes(0, 1), **kwargs)

    plt.xlabel(xlab)
    plt.ylabel(ylab)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    if proplab == "prop":
        proplab = prop
    cbar.set_label(proplab, rotation=270)
    return fig, ax


def grid_property3d(grd, prop, z="prop", z_res=10, collections = {},
                    xlab="x", ylab="y", zlab="prop",
                    proplab="prop", cmap="Greens", **kwargs):
    """
    Plots a given properties 'prop' and 'z' as a voxels on an x-y-z grid.
    See mpl_toolkits.mplot3d.axes3d.Axes3D.voxels.

    Parameters
    ----------
    grd: Grid
        define.environment.grid to plot
    prop : str
        Name of the property to represent a color.
    z : str, optional
        Name of the property to plot as z. The default is "prop", which uses the same
        property as prop.
    z_res : int, optional
        Resolution to plot z at. The default is 10.
    xlab : str, optional
        Label for x-axis. The default is "x".
    ylab : str, optional
        Label for y-axis. The default is "y".
    zlab : str, optional
        Label for the z-axis. The default is "prop", which uses the name of the
        property.
    proplab : str, optional
        Label for the property. The default is "prop", which uses the name of the
        property provided.
    cmap : str, optional
        Name of the matplotlib colormap to use for colors. The default is "Greens".
    **kwargs : kwargs
        Kwargs to pass to Axes3D.voxels

    Returns
    -------
    fig : mpl.figure
        Plotted figure object
    ax : mpl.axis
        Ploted axis object.
    """
    default_kwargs = dict(edgecolor='k')
    kwargs = {**kwargs, **default_kwargs}

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    c_array = getattr(grd, prop)
    if z == "prop":
        z_array = c_array
    else:
        z_array = getattr(grd, z)

    dims = z_array.shape
    X, Y, Z = np.indices((dims[0]+1, dims[1]+1, z_res+1))
    z_shape = Z[:-1, :-1, :-1].swapaxes(0, 2).swapaxes(1, 2)

    max_z = z_array.max()
    min_z = z_array.min()
    norm_z_array = z_res * (z_array - min_z)/(max_z - min_z + 0.00000001)
    round_z_array = np.digitize(norm_z_array, [i for i in range(z_res)])
    shape = z_shape < round_z_array
    shape = shape.swapaxes(0, 1).swapaxes(1, 2)
    X_scale = X * grd.p.blocksize - grd.p.blocksize/2
    Y_scale = Y * grd.p.blocksize - grd.p.blocksize/2
    Z_scale = Z * (max_z - min_z) / z_res + min_z

    color_shape = np.array([c_array for i in range(z_res)])
    norm = plt.Normalize(color_shape.min(), color_shape.max())
    cmap = colormaps[cmap]
    colors = cmap(norm(color_shape)).swapaxes(0, 1).swapaxes(1, 2)

    for i, (prop, coll_kwargs) in enumerate(collections.items()):
        coll_colors = cm.rainbow(np.linspace(0, 1, len(collections)))
        coll = grd.get_collection(prop)
        if 'color' not in coll_kwargs:
            coll_color = colormaps['rainbow'](coll_colors[i])
        else:
            coll_color = to_rgba(coll_kwargs['color'])

        if "text_z_offset" not in coll_kwargs:
            coll_kwargs['text_z_offset'] = (max_z - min_z) / z_res
        for pt in coll:
            index = grd.to_index(*pt)
            inds = np.where(shape[index])
            if any(inds[0]):
                z_index = inds[0][-1]
            else:
                z_index = 0
            colors[index[0], index[1], z_index] = coll_color

    ax.voxels(X_scale, Y_scale, Z_scale, shape, facecolors=colors, **kwargs)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_zlabel(zlab)
    return fig, ax


def grid_collection(grd, prop, fig=None, ax=None, label=True, z="",
                    legend_args=False, text_z_offset=0.0, **kwargs):
    """
    Shows a collection on the grid as square patches.

    Parameters
    ----------
    grd: Grid
        define.environment.grid to plot
    prop : str
        Name of the collection
    fig : matplotlib.figure, optional
        Existing Figure. The default is None.
    ax : matplotlib.axis, optional
        Existing axis. The default is None.
    label : str/bool, optional
        Label for the collection. The default is True, which shows the collection
        name. If False, no label is provided. If a string, the string is used as
        the label.
    z: str
        Argument to plot as third dimension on 3d plot. Default is "", which
        returns a 2d plot.
    legend_args : dict/False
        Specifies arguments to legend. Default is False, which shows no legend.
    **kwargs : kwargs
        Kwargs to matplotlib.patches.Rectangle

    Returns
    -------
    fig : mpl.figure
        Plotted figure object
    ax : mpl.axis
        Ploted axis object.

    """
    offset = grd.p.blocksize/2
    if not ax:
        fig, ax = plt.subplots(1)
        if not z:
            fig, ax = plt.subplots(1)
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_zlim(getattr(grd, z).min(), getattr(grd, z).max())
        ax.set_xlim(-offset, grd.p.x_size*grd.p.blocksize+offset)
        ax.set_ylim(-offset, grd.p.y_size*grd.p.blocksize+offset)

    coll = grd.get_collection(prop)
    for i, pt in enumerate(coll):
        corner = pt - np.array([offset, offset])
        rect = Rectangle(corner, grd.p.blocksize, grd.p.blocksize,
                         label=prop, **kwargs)
        ax.add_patch(rect)
        if z:
            art3d.patch_2d_to_3d(rect, z=grd.get(pt[0], pt[1], z))
        if label:
            if type(label) != str:
                lab = rect.get_label()
            else:
                lab = label
            if z:
                ax.text(pt[0], pt[1], grd.get(pt[0], pt[1], z)+text_z_offset,
                        lab,
                        horizontalalignment="center", verticalalignment="center")
            else:
                ax.text(pt[0], pt[1], lab,
                        horizontalalignment="center", verticalalignment="center")
    if not legend_args == False:
        if legend_args == True:
            legend_args = {}
        consolidate_legend(ax, **legend_args)
    return fig, ax


def grid(grd, prop, collections={}, legend_args=False, **kwargs):
    """
    Plots a property and set of collections on the grid.

    Parameters
    ----------
    grd: Grid
        define.environment.grid to plot
    prop : str
        Property to plot.
    collections : dict, optional
        Collections to plot and their respective kwargs for show_collection.
        The default is {}.
    **kwargs : kwargs
        kwargs to show_property.

    Returns
    -------
    fig : mpl.figure
        Plotted figure object
    ax : mpl.axis
        Ploted axis object.
    """
    fig, ax = grid_property(grd, prop, **kwargs)
    for coll in collections:
        grid_collection(grd, coll, fig=fig, ax=ax, **collections[coll])
    return fig, ax


def grid3d(grd, prop, z="prop", collections={}, legend_args=False, **kwargs):
    """
    Plots a property and set of collections in a discretized (voxelized) version
    of the grid.

    Parameters
    ----------
    grd: Grid
        define.environment.grid to plot
    prop : str
        Property to plot.
    z : str, optional
        Property to use as the height. The default is "prop".
    collections : dict, optional
        Collections to plot and their respective kwargs for show_collection.
        The default is {}.
    legend_args : dict/False
        Specifies arguments to legend. Default is False, which shows no legend.
    **kwargs : kwargs
        kwargs to show_property3d.

    Returns
    -------
    fig : mpl.figure
        Plotted figure object
    ax : mpl.axis
        Ploted axis object.
    """
    if z == "prop":
        z = prop
    fig, ax = grid_property3d(grd, prop, z=z, collections=collections, **kwargs)
    for coll in collections:
        grid_collection(grd, coll, fig=fig, ax=ax, legend_args=legend_args,
                        **collections[coll], z=z)
    return fig, ax


def trajectories(simhists, *plot_values,
                 comp_groups={}, indiv_kwargs={}, figsize=(4, 4),
                 time_groups=[], time_ticks=5.0, time_fontsize=8,
                 xlim=(), ylim=(), zlim=(), legend=True, title='', **kwargs):
    """
    Shows trajectories from the environment in 2d or 3d space.

    Parameters
    ----------
    simhists : History
        History to get trajectories from.
    *plot_values : str
        Plot values corresponding to the x/y/z values (e.g, 'position.s.x')
    comp_groups : dict, optional
        Dictionary for comparison groups (if more than one) with structure given by:
        ::
            {'group1': ('scen1', 'scen2'),
             'group2':('scen3', 'scen4')}.

        Default is {}, which compares nominal and faulty.
        If {'default': 'default'} is passed, all scenarios will be put in one group.
        If a legend is shown, group names are used as labels.
    indiv_kwargs : dict, optional
        Dict of kwargs to use to differentiate each comparison group.
        Has structure::
            {comp1: kwargs1, comp2: kwargs2}

        where kwargs is an individual dict of plt.plot arguments for the
        comparison group comp (or scenario, if not aggregated) which overrides
        the global kwargs (or default behavior). If no comparison groups are given,
        use 'default' for a single history or 'nominal'/'faulty' for a fault history
        e.g.::
            kwargs = {'nominal': {color: 'green'}} 

        would make the nominal color green. Default is {}.
    figsize : tuple (float,float)
        x-y size for the figure. The default is 'default', which dymanically gives 3 for
        each column and 2 for each row.
    time_groups : list, optional
        List of strings corresponding to groups (e.g., 'nominal') to label the time
        at each point in the trajectory. The default is [].
    time_ticks : float, optional
        Ticks for times (if used). The default is 5.0.
    time_fontsize : int, optional
        Fontsize for time-ticks. The default is 8.
    xlim : tuple, optional
        Limits on the x-axis. The default is ().
    ylim : tuple, optional
        Limits on the y-axis. The default is ().
    zlim : tuple, optional
        Limits on the z-axis. The default is ().
    legend : bool, optional
        Whether to show a legend. The default is True.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    fig : figure
        Matplotlib figure object
    ax : axis
        Corresponding matplotlib axis
    """
    simhists, plot_values, grouphists, indiv_kwargs = prep_hists(simhists,
                                                                 plot_values,
                                                                 comp_groups,
                                                                 indiv_kwargs)
    if len(plot_values) == 2:
        fig, ax = plt.subplots(figsize=figsize)
        plot_meth = traj
    elif len(plot_values) == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plot_meth = traj3
    else:
        raise Exception("Number of plot values must be 2 or 3, not "+len(plot_values))

    for group, hists in grouphists.items():
        local_kwargs = {**kwargs, **indiv_kwargs.get(group, {})}
        mark_time = group in time_groups
        plot_meth(ax, hists, *plot_values, label=group,
                  mark_time=mark_time, time_ticks=time_ticks,
                  time_fontsize=time_fontsize, **local_kwargs)

    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    if zlim:
        ax.set_zlim(*zlim)
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
    if title:
        ax.title(title)
    return fig, ax


def traj(ax, hists, xlab, ylab,
         mark_time=False, time_ticks=1.0, time_fontsize=8, **kwargs):
    """
    Plots a single set of trajectories on an existing matplotlib axis.

    Parameters
    ----------
    ax : matplotlib axis
        Axis object to mark on
    hists : History
        History to get the values from
    xlab : str
        Name to use for the x-values.
    ylab : str
        Name to use for the y-values.
    mark_time : bool, optional
        Whether to mark the time of the trajectory at given ticks. The default is False.
    time_ticks : float, optional
        Time tick frequency. The default is 1.0.
    time_fontsize : int, optional
        Size of font for time ticks. The default is 8.
    **kwargs : kwargs
        kwargs to ax.plot
    """
    xs = [*hists.get_values(xlab).values()]
    ys = [*hists.get_values(ylab).values()]
    times = [*hists.get_values("time").values()]
    for i, x in enumerate(xs):
        ax.plot(x, ys[i], **kwargs)
        if mark_time:
            mark_times(ax, time_ticks, times[i], x, ys[i], fontsize=time_fontsize)


def traj3(ax, hists, xlab, ylab, zlab,
          mark_time=False, time_ticks=1.0, time_fontsize=8, **kwargs):
    """
    Plots a single set of trajectories on an existing matplotlib axis. See show.traj.
    """
    xs = [*hists.get_values(xlab).values()]
    ys = [*hists.get_values(ylab).values()]
    zs = [*hists.get_values(zlab).values()]
    times = [*hists.get_values("time").values()]
    for i, x in enumerate(xs):
        ax.plot(x, ys[i], zs[i], **kwargs)
        if mark_time:
            mark_times(ax, time_ticks, times[i], x, ys[i], zs[i],
                       fontsize=time_fontsize)


def mark_times(ax, tick, time, *plot_values, fontsize=8):
    """
    Marks times on an axis at a particular tick interval.

    Parameters
    ----------
    ax : matplotlib axis
        Axis object to mark on
    tick : float
        Tick frequency.
    time : np.array
        Time vector.
    *plot_values : np.array
        x,y,z vectors
    fontsize : int, optional
        Size of the font. The default is 8.
    """
    for st in zip(*plot_values, time):
        tt = st[-1]
        xyz = st[:-1]
        if tt % tick == 0:
            ax.text(*xyz, 't='+str(tt), fontsize=fontsize)


def consolidate_legend(ax, **kwargs):
    """
    Creates a consolidated legend with all grid properties

    Parameters
    ----------
    ax : axis
        Matplotlib axis.
    **kwargs : kwargs
        kwargs to ax.legend
    """
    kwargs = {**dict(bbox_to_anchor=(1.05, 1), loc='upper left'), **kwargs}
    ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.get_legend().remove()
    ax.legend(by_label.values(), by_label.keys(), **kwargs)


if __name__ == "__main__":
    from fmdtools.define.environment import ExampleGrid

    ex = ExampleGrid()
    grid_property(ex, "v", cmap="Greys")
    grid_collection(ex, "high_v")
    grid(ex, "h", collections={"high_v": {"alpha": 0.5, "color": "red"}})
    grid_property3d(ex, "h", z="v",
                    collections={"high_v": {"alpha": 0.5, "color": "red"}})

    grid_property(ex, "v", cmap="Greys")
    grid_property3d(ex, "v")
    grid_property3d(ex, "h", z="v")
    grid_collection(ex, "high_v")
    grid_collection(ex, "high_v", z="v")
    grid3d(ex, "h", z="v",
           collections={"pts": {"color": "blue"},
                        "high_v": {"alpha": 0.5, "color": "red"}},
           legend_args=True)
    


