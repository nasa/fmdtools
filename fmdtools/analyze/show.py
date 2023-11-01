# -*- coding: utf-8 -*-
"""
Module for showing the properties of the model (mainly focused on
grids, environments, trajectories, etc).

Uses the following primary methods:

- :func: `coord`: Plots a Coords collections/properties on an x-y grid.
- :func: `coord3d`: Plots a Coords collection.property on an x-y grid.
- :func:`sim_order`: Plots the run order for the model during the dynamic propagation
  step used by dynamic_behavior() methods.

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
from matplotlib.collections import PolyCollection
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from fmdtools.analyze.plot import consolidate_legend, setup_plot
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

    fig, ax = setup_plot(fig=fig, ax=ax, z=True, figsize=figsize)

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
        fig, ax = setup_plot(z=z, figsize=figsize)
        if type(z) == str and z:
            ax.set_zlim(getattr(crd, z).min(), getattr(crd, z).max())
        ax.set_xlim(-offset, crd.p.x_size*crd.p.blocksize+offset)
        ax.set_ylim(-offset, crd.p.y_size*crd.p.blocksize+offset)
    else:
        fig, ax = setup_plot(fig=fig, ax=ax, z=z, figsize=figsize)

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


def sim_order(mdl, rotateticks=False, title="Dynamic Run Order"):
    """
    Plot the run order for the model during the dynamic propagation step used
    by dynamic_behavior() methods, where the x-direction is the order of each
    function executed and the y are the corresponding flows acted on by the
    given methods.

    Parameters
    ----------
    mdl : Model
        fmdtools model
    rotateticks : Bool, optional
        Whether to rotate the x-ticks (for bigger plots). The default is False.
    title : str, optional
        String to use for the title (if any). The default is "Dynamic Run Order".

    Returns
    -------
    fig : figure
        Matplotlib figure object
    ax : axis
        Corresponding matplotlib axis

    """
    fxnorder = list(mdl.dynamicfxns)
    times = [i+0.5 for i in range(len(fxnorder))]
    fxntimes = {f: i for i, f in enumerate(fxnorder)}

    flowtimes = {f: [fxntimes[n] for n in mdl.graph.neighbors(
        f) if n in mdl.dynamicfxns] for f in mdl.flows}

    lengthorder = {k: v for k, v in
                   sorted(flowtimes.items(), key=lambda x: len(x[1]), reverse=True)
                   if len(v) > 0}
    starttimeorder = {k: v for k, v in
                      sorted(lengthorder.items(), key=lambda x: x[1][0], reverse=True)}
    endtimeorder = [k for k, v in
                    sorted(starttimeorder.items(), key=lambda x: x[1][-1], reverse=True)]
    flowtimedict = {flow: i for i, flow in enumerate(endtimeorder)}

    fig, ax = plt.subplots()

    for flow in flowtimes:
        phaseboxes = [((t, flowtimedict[flow]-0.5),
                       (t, flowtimedict[flow]+0.5),
                       (t+1.0, flowtimedict[flow]+0.5),
                       (t+1.0, flowtimedict[flow]-0.5))
                      for t in flowtimes[flow]]
        bars = PolyCollection(phaseboxes)
        ax.add_collection(bars)

    flowtimes = [i+0.5 for i in range(len(mdl.flows))]
    ax.set_yticks(list(flowtimedict.values()))
    ax.set_yticklabels(list(flowtimedict.keys()))
    ax.set_ylim(-0.5, len(flowtimes)-0.5)
    ax.set_xticks(times)
    ax.set_xticklabels(fxnorder, rotation=90*rotateticks)
    ax.set_xlim(0, len(times))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(which='minor',  linewidth=2)
    ax.tick_params(axis='x', bottom=False, top=False, labelbottom=False, labeltop=True)
    if title:
        if rotateticks:
            fig.suptitle(title, fontweight='bold', y=1.15)
        else:
            fig.suptitle(title, fontweight='bold')
    return fig, ax


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
        fig, ax = setup_plot(z=z, figsize=figsize)
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
        fig, ax = setup_plot(z=z, figsize=figsize)
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
