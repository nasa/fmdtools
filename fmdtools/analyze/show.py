# -*- coding: utf-8 -*-
"""
Module for showing the physical 2d/3d properties of the model (mainly focused on
grids, environments, trajectories, etc).

Uses the following primary methods:

- :func: `coord`: Plots a Coords collections/properties on an x-y grid.
- :func: `coord3d`: Plots a Coords collection.property on an x-y grid.
- :func: `trajectories`: Plots trajectories from a history.

And secondary methods:

- :func:`coord_property`: Plots a given Coords property on an x-y grid.
- :func:`coord_property3d`: Plots a given Coords property in 3d space.
- :func: `coord_collection`: Plots a given Coords collections on an x-y grid.

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
from fmdtools.analyze.plot import prep_hists
from shapely import LineString, Point, Polygon


def coord_property(crd, prop, xlab="x", ylab="y", proplab="prop", **kwargs):
    """
    Plot a given property 'prop' as a colormesh on an x-y grid.

    See matplotlib.pyplot.pcolormesh.

    Parameters
    ----------
    crd: Coords
        coord to plot
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

    p = getattr(crd, prop)
    # im = ax.matshow(p, **kwargs)
    offset = crd.p.blocksize/2
    x = np.linspace(0., crd.p.blocksize*(crd.p.x_size-1), crd.p.x_size)
    y = np.linspace(0., crd.p.blocksize*(crd.p.y_size-1), crd.p.y_size)
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


def coord_property3d(crd, prop, z="prop", z_res=10, collections={},
                    xlab="x", ylab="y", zlab="prop",
                    proplab="prop", cmap="Greens",
                    fig=None, ax=None, figsize=(4, 5), **kwargs):
    """
    Plot a given properties 'prop' and 'z' as a voxels on an x-y-z grid.

    See mpl_toolkits.mplot3d.axes3d.Axes3D.voxels.

    Parameters
    ----------
    crd: Coord
        coord to plot
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
    fig : matplotlib.figure, optional
        Existing Figure. The default is None.
    ax : matplotlib.axis, optional
        Existing axis. The default is None.
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

    fig, ax = init_figure(fig=fig, ax=ax, z=True, figsize=figsize)

    c_array = getattr(crd, prop)
    if z == "prop":
        z_array = c_array
    elif not z:
        z_array = c_array * 0
    else:
        z_array = getattr(crd, z)

    dims = z_array.shape
    X, Y, Z = np.indices((dims[0]+1, dims[1]+1, z_res+1))
    z_shape = Z[:-1, :-1, :-1].swapaxes(0, 2).swapaxes(1, 2)

    max_z = 1 * z_array.max()
    min_z = 1 * z_array.min()
    norm_z_array = z_res * (1*z_array - min_z)/(max_z - min_z + 0.00000001)
    round_z_array = np.digitize(norm_z_array, [i for i in range(z_res)])
    shape = z_shape < round_z_array
    shape = shape.swapaxes(0, 1).swapaxes(1, 2)
    X_scale = X * crd.p.blocksize - crd.p.blocksize/2
    Y_scale = Y * crd.p.blocksize - crd.p.blocksize/2
    Z_scale = Z * (max_z - min_z) / z_res + min_z

    color_shape = np.array([c_array for i in range(z_res)])
    norm = plt.Normalize(color_shape.min(), color_shape.max())
    cmap = colormaps[cmap]
    colors = cmap(norm(color_shape)).swapaxes(0, 1).swapaxes(1, 2)

    for i, (prop, coll_kwargs) in enumerate(collections.items()):
        coll_colors = cm.rainbow(np.linspace(0, 1, len(collections)))
        coll = crd.get_collection(prop)
        if 'color' not in coll_kwargs:
            coll_color = colormaps['rainbow'](coll_colors[i])
        else:
            coll_color = to_rgba(coll_kwargs['color'])

        if "text_z_offset" not in coll_kwargs:
            coll_kwargs['text_z_offset'] = (max_z - min_z) / z_res
        for pt in coll:
            index = crd.to_index(*pt)
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


def init_figure(fig=None, ax=None, z=False, figsize=()):
    """
    Initialize a 2d or 3d figure at a given size.

    If there is a pre-existing figure or axis, uses that instead.
    """
    if not fig:
        if z or (type(z) in (int, float)):
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots(1, figsize=figsize)
    return fig, ax


def coord_collection(crd, prop, fig=None, ax=None, label=True, z="",
                     legend_args=False, text_z_offset=0.0, figsize=(4, 4), **kwargs):
    """
    Show a collection on the grid as square patches.

    Parameters
    ----------
    crd: Coords
        coord to plot
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
        Argument to plot as third dimension on 3d plot. Default is '', which
        returns a 2d plot. If a number is provided, the plot will be 3d with
        the height at that constant z-value.
    legend_args : dict/False
        Specifies arguments to legend. Default is False, which shows no legend.
    text_z_offset : float
        Offset for text. Default is 0.0
    figsize : tuple
        Size for the figure. Default is (4,4)
    **kwargs : kwargs
        Kwargs to matplotlib.patches.Rectangle

    Returns
    -------
    fig : mpl.figure
        Plotted figure object
    ax : mpl.axis
        Ploted axis object.

    """
    offset = crd.p.blocksize/2
    if not ax:
        fig, ax = init_figure(z=z, figsize=figsize)
        if type(z) == str and z:
            ax.set_zlim(getattr(crd, z).min(), getattr(crd, z).max())
        ax.set_xlim(-offset, crd.p.x_size*crd.p.blocksize+offset)
        ax.set_ylim(-offset, crd.p.y_size*crd.p.blocksize+offset)
    else:
        fig, ax = init_figure(fig=fig, ax=ax, z=z, figsize=figsize)

    coll = crd.get_collection(prop)
    for i, pt in enumerate(coll):
        corner = pt - np.array([offset, offset])
        rect = Rectangle(corner, crd.p.blocksize, crd.p.blocksize,
                         label=prop, **kwargs)
        ax.add_patch(rect)
        if type(z) == str and z:
            z_h = crd.get(pt[0], pt[1], z)
            art3d.patch_2d_to_3d(rect, z=z_h)
        elif type(z) in [float, int]:
            z_h = z
            art3d.patch_2d_to_3d(rect, z=z_h)
        else:
            z_h = None
        if label:
            if type(label) != str:
                lab = rect.get_label()
            else:
                lab = label
            if not z_h == None:
                ax.text(pt[0], pt[1], z_h+text_z_offset, lab,
                        horizontalalignment="center", verticalalignment="center")
            else:
                ax.text(pt[0], pt[1], lab,
                        horizontalalignment="center", verticalalignment="center")
    if not legend_args == False:
        if legend_args == True:
            legend_args = {}
        consolidate_legend(ax, **legend_args)
    return fig, ax


def coord(crd, prop, collections={}, legend_args=False, **kwargs):
    """
    Plot a property and set of collections on the grid.

    Parameters
    ----------
    crd: Coords
        coord to plot
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
    fig, ax = coord_property(crd, prop, **kwargs)
    for coll in collections:
        coord_collection(crd, coll, fig=fig, ax=ax, **collections[coll])
    return fig, ax


def coord3d(crd, prop, z="prop", collections={}, legend_args=False, voxels=True,
           **kwargs):
    """
    Plot a property and set of collections in a discretized version of the grid.

    Parameters
    ----------
    crd: Coords
        coord to plot
    prop : str
        Property to plot.
    z : str, optional
        Property to use as the height. The default is "prop".
    collections : dict, optional
        Collections to plot and their respective kwargs for show_collection.
        The default is {}.
    legend_args : dict/False
        Specifies arguments to legend. Default is False, which shows no legend.
    voxels : bool
        Whether or not to plot the grid as voxels. Default is True.
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
    elif z == '':
        z = 0.0
    if voxels:
        fig, ax = coord_property3d(crd, prop, z=z, collections=collections, **kwargs)
    else:
        fig, ax = coord_collection(crd, "pts", z=z,
                                  legend_args=legend_args, label=False, **kwargs)
    for coll in collections:
        coord_collection(crd, coll, fig=fig, ax=ax, legend_args=legend_args,
                        **collections[coll], z=z)
    return fig, ax


def trajectories(simhists, *plot_values,
                 comp_groups={}, indiv_kwargs={}, figsize=(4, 4),
                 time_groups=[], time_ticks=5.0, time_fontsize=8,
                 xlim=(), ylim=(), zlim=(), legend=True, title='',
                 fig=None, ax=None, **kwargs):
    """
    Show trajectories from the environment in 2d or 3d space.

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
    title : str, optional
        Title to add. Default is '' (no title).
    fig : matplotlib.figure, optional
        Existing Figure. The default is None.
    ax : matplotlib.axis, optional
        Existing axis. The default is None.
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
        fig, ax = init_figure(fig=fig, ax=ax, z=False, figsize=figsize)
        plot_meth = traj
    elif len(plot_values) == 3:
        fig, ax = init_figure(fig=fig, ax=ax, z=True, figsize=figsize)
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
        ax.set_title(title)
    return fig, ax


def traj(ax, hists, xlab, ylab,
         mark_time=False, time_ticks=1.0, time_fontsize=8, **kwargs):
    """
    Plot a single set of trajectories on an existing matplotlib axis.

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
    Plot a single set of trajectories on an existing matplotlib axis.

    See show.traj.
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


def geomarch(geomarch, geoms = {'all': {}}, fig=None, ax=None, figsize = (4, 4),
             z=False, **kwargs):
    """
    Show the shapes of a GeomArch all on one plot.

    Parameters
    ----------
    geomarch : GeomArch
        Geometric architecture to plot.
    geoms : dict, optional
        Individual shapes to plot and their corresponding kwargs.
        The default is {'all': {}}.
    fig : matplotlib.figure, optional
        Existing Figure. The default is None.
    ax : matplotlib.axis, optional
        Existing axis. The default is None.
    figsize : tuple, optional
        Size for figure (if instantiating). The default is (4, 4).
    z : bool/number, optional
        If plotting on a 3d axis, set z to a number which will be the z-level.
        The default is False.
    **kwargs : kwargs
        Overall kwargs to show.geom for all geoms.

    Returns
    -------
    fig : figure
        Matplotlib figure object
    ax : axis
        Corresponding matplotlib axis
    """
    if not ax:
        fig, ax = init_figure(z=z, figsize=figsize)
    if 'all' in geoms:
        geoms = {g: {'shapes': 'all'} for g in geomarch.geoms}

    for geomname, geom_kwargs in geoms.items():
        local_kwargs = {**kwargs, 'geomlabel': geomname, **geom_kwargs}
        fig, ax = geom(geomarch.geoms[geomname], ax=ax, fig=fig, z=z, **local_kwargs)
    return fig, ax


def geom(geom, shapes={'all': {}}, fig=None, ax=None, figsize=(4, 4), z=False,
         geomlabel='', **kwargs):
    """
    Show a Geom (shape and buffers) as lines on a plot.

    Parameters
    ----------
    geom : Geom
        Geom object.
    shapes : dict, optional
        Aspects of the Geom to plot and their corresponding plot kwargs.
        The default is {'all': {}}.
    fig : matplotlib.figure, optional
        Existing Figure. The default is None.
    ax : matplotlib.axis, optional
        Existing axis. The default is None.
    figsize : tuple, optional
        Size for figure (if instantiating). The default is (4, 4).
    z : bool/number, optional
        If plotting on a 3d axis, set z to a number which will be the z-level.
        The default is False.
    geomlabel : str, optional
        Overall label for the geom (if desired). The default is ''.
    **kwargs : kwargs
        overall kwargs for plt.plot for all shapes.

    Returns
    -------
    fig : figure
        Matplotlib figure object
    ax : axis
        Corresponding matplotlib axis
    """
    if not ax:
        fig, ax = init_figure(z=z, figsize=figsize)
    if 'all' in shapes:
        shapes = {'shape': {}, **{v: {} for v in geom.buffers}}
    if type(z) in (int, float):
        plot_kwargs = {'zs': z, 'zdir': 'z', **kwargs}
    else:
        plot_kwargs = kwargs
    for shape, shape_kwargs in shapes.items():
        if geomlabel:
            shape_label = geomlabel + "." + shape
        else:
            shape_label = shape
        local_kwargs = {**plot_kwargs, 'label': shape_label, **shape_kwargs}
        shap = getattr(geom, shape)
        if isinstance(shap, Point):
            ax.scatter(shap.x, shap.y, **local_kwargs)
        elif isinstance(shap, LineString):
            linecoords = np.array([*shap.coords])
            ax.plot(linecoords[:, 0], linecoords[:, 1], **local_kwargs)
        elif isinstance(shap, Polygon):
            ax.plot(*shap.exterior.xy, **local_kwargs)
    ax.axis('equal')
    consolidate_legend(ax, **kwargs)
    return fig, ax


def mark_times(ax, tick, time, *plot_values, fontsize=8):
    """
    Mark times on an axis at a particular tick interval.

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
    Create a consolidated legend with all grid properties.

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
    from fmdtools.define.coords import ExampleCoords

    ex = ExampleCoords()
    coord_property(ex, "v", cmap="Greys")
    coord_collection(ex, "high_v")
    coord(ex, "h", collections={"high_v": {"alpha": 0.5, "color": "red"}})
    coord_property3d(ex, "h", z="v",
                     collections={"high_v": {"alpha": 0.5, "color": "red"}})

    coord_property(ex, "v", cmap="Greys")
    coord_property3d(ex, "v")
    coord_property3d(ex, "h", z="v")
    coord_collection(ex, "high_v")
    coord_collection(ex, "high_v", z="v")
    coord3d(ex, "h", z="v",
            collections={"pts": {"color": "blue"},
                         "high_v": {"alpha": 0.5, "color": "red"}},
            legend_args=True)
