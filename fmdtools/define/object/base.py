#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines :class:`BaseObject` class used to define objects.

Classes in this module:

- :class:`BaseObject`: Base object class used throughout.
- :class:`ObjectGraph`: Generic ModelGraph representation for objects.

Functions contained in this module:

- :func:`check_pickleability`:Checks to see which attributes of an object will pickle
  (and thus parallelize)"

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The “"Fault Model Design tools - fmdtools version 2"” software is licensed
under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0. 

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

from fmdtools.define.base import get_var, get_methods, get_obj_name, get_memory
from fmdtools.analyze.common import get_sub_include
from fmdtools.analyze.history import History
from fmdtools.analyze.graph.model import add_node, add_edge, remove_base, ModelGraph
from fmdtools.analyze.graph.model import create_inheritance_subgraph

import dill
import pickle
import time
import sys
import numpy as np
from inspect import signature, isclass, ismethod


class ObjectGraph(ModelGraph):
    """Objectgraph represents the definition of an Object."""

    def __init__(self, mdl, with_methods=True, with_inheritance=True,
                 with_subgraph_edges=False, **kwargs):
        ModelGraph.__init__(self, mdl, with_methods=with_methods,
                            with_inheritance=with_inheritance,
                            with_subgraph_edges=with_subgraph_edges, **kwargs)

    def set_edge_labels(self, title='edgetype', title2='', subtext='role',
                        **edge_label_styles):
        super().set_edge_labels(title=title, title2=title2, subtext=subtext,
                                **edge_label_styles)

    def set_node_labels(self, title='shortname', title2='classname', **node_label_styles):
        super().set_node_labels(title=title, title2=title2, **node_label_styles)


example_object_code = """
from fmdtools.define.container.state import ExampleState
class ExampleObject(BaseObject):
    container_s = ExampleState
    def indicate_high_x(self):
        return self.s.x > 1.0
    def indicate_y_over_t(self, t):
        return self.s.y > t
"""


class BaseObject(object):
    """
    Base object for Blocks, Flows, and Architectures.

    Enables the instantiation of roles via class variables and object parameters, as
    well as representation of indictators and tracking.

    Examples
    --------
    The roletypes class variable lets one add specific types of roles to the class.
    By default, 'container' is included in roletypes, enabling:

    >>> from fmdtools.define.container.state import ExampleState
    >>> class ExampleObject(BaseObject):
    ...    container_s = ExampleState
    ...    def indicate_high_x(self):
    ...        return self.s.x > 1.0
    ...    def indicate_y_over_t(self, t):
    ...        return self.s.y > t


    >>> ex = ExampleObject()
    >>> ex.roletypes
    ['container']
    >>> ex.containers
    ('s',)
    >>> ex.s
    ExampleState(x=1.0, y=1.0)

    If an already-instanced role is passed, the BaseObject will take this
    copy instead of instancing its own:

    >>> ex2 = ExampleObject(s=ExampleState(2.0, 4.0))
    >>> ex2.s
    ExampleState(x=2.0, y=4.0)

    The method `indicate_high_x` is called an indicator. Indicators show up in the
    indicators property:

    >>> ex.indicators
    ('high_x', 'y_over_t')

    And are used to evaluate conditions, e.g.:

    >>> ex.indicate_high_x()
    False
    >>> ex2.indicate_high_x()
    True

    Time may be used as an optional argument to indicators:

    >>> ex.indicate_y_over_t(0.0)
    True
    >>> ex2.return_true_indicators(0.0)
    ['high_x', 'y_over_t']

    A history may be created using create_hist:

    >>> ex.create_hist([0.0, 1.0])
    i.high_x:                       array(2)
    i.y_over_t:                     array(2)

    Note that adding roles to the class often means modifying default_track.
    Initializing all possible using the 'all' option:

    >>> ex = ExampleObject(track='all')
    >>> ex.create_hist([0.0, 1.0])
    i.high_x:                       array(2)
    i.y_over_t:                     array(2)
    s.x:                            array(2)
    s.y:                            array(2)
    """

    __slots__ = ('name', 'containers', 'indicators', 'track', 'root')
    roletypes = ['container']
    roledicts = []
    rolevars = []
    immutable_roles = ['p']
    flexible_roles = []
    default_track = ['i']
    check_dict_creation = False

    def __init__(self, name='', roletypes=[], track='default', root='', **kwargs):
        """
        Initialize the baseobject.

        Parameters
        ----------
        name : str, optional
            Name to give the object. The default is '', which defaults to the class name
        roletypes : list, optional
            Role types to instance in this method using init_roletypes.
            The default is [], which initializes all of them.
        track : str/dict
            Which model states to track over time (overwrites mdl.default_track).
            Default is 'default'
            Options:

            - 'default'
            - 'all'
            - 'none'
            - or a dict of form ::

                {'fxns':{'fxn1':'att1'}, 'flows':{'flow1':'att1'}}
        root : str
            Name of object containing the object. Default is 'self'.
        **kwargs : dict, object
            Keywork arguments for the roles.
            May be a dict of non-default arguments (e.g. s={'x': 1.0}) or
            a fully instantiated object (e.g., s=ExampleState()),
        """
        if not name:
            self.name = self.__class__.__name__.lower()
        else:
            self.name = name
        self.root = root
        self.init_indicators()
        self.init_roletypes(*roletypes, **kwargs)
        self.init_track(track)
        self.check_slots()

    def base_type(self):
        """Return fmdtools type of the model class."""
        return BaseObject

    def get_typename(self):
        """Return name of the fmdtools type of the model class."""
        return self.base_type().__name__

    def check_slots(self):
        """Check if __slots__ were defined for the class to preemt __dict__ creation."""
        if self.check_dict_creation and hasattr(self, '__dict__'):
            raise Exception("__slots__ not defined for class: "
                            + self.__class__.__name__)

    def init_track(self, track):
        """Add .track attribute based on default and input."""
        if track == 'default':
            track = self.default_track
        if track == 'all':
            track = self.get_all_possible_track()
        elif track == 'none':
            track = []
        elif isinstance(track, str):
            track = (track,)

        if not track:
            track = []

        if isinstance(track, dict):
            self.track = track
        else:
            # make sure role dict attributes are also tracked.
            self.track = []
            for t in track:
                if t in self.roledicts:
                    self.track.extend(getattr(self, t))
                else:
                    self.track.append(t)

    def init_roletypes(self, *roletypes, **kwargs):
        """
        Initialize roletypes with given kwargs.

        Parameters
        ----------
        *roletypes : str
            Names of roles (e.g., container, flow, etc)
        **kwargs : dict
            Dictionary arguments (or already instantiated objects) to use for the
            attributes.
        """
        if not roletypes:
            roletypes = self.roletypes
        for roletype in roletypes:
            if roletype not in self.roletypes:
                raise Exception("Roletype: " + roletype + " not in class variable" +
                                " self.roletypes: " + str(self.roletypes))
            self.init_roles(roletype, **kwargs)

    def find_roletype_initiators(self, roletype):
        return tuple([at[len(roletype)+1:]
                     for at in dir(self) if at.startswith(roletype+'_')])

    def get_full_name(self, with_root=True):
        """Get the full name of the object (root + name)."""
        if self.root and with_root:
            return self.root + "." + self.name
        else:
            return self.name

    def init_roles(self, roletype, **kwargs):
        """
        Initialize the role 'roletype' for a given object.

        Roles defined using roletype_x in its class variables for the attribute x.

        Object is instantiated with the attribute x, y, z corresponding to output of the
        class variable roletype_x, roletype_y, roletype_z, etc.

        Parameters
        ----------
        roletype : str
            Role to initialize (e.g., 'container'). If none provided, initializes all.
        **kwargs : dict
            Dictionary arguments (or already instantiated objects) to use for the
            attributes.
        """
        # creates tuple of roles at .roletypes
        container_collection = roletype + 's'
        roles = self.find_roletype_initiators(roletype)
        setattr(self, container_collection, roles)

        # initialize roles and add as attributes to the object
        for rolename in roles:
            container_initializer = getattr(self, roletype+'_'+rolename)
            container_args = kwargs.get(rolename, dict())
            if isinstance(container_args, container_initializer):
                container = container_args
            elif isinstance(container_args, dict):
                if issubclass(container_initializer, BaseObject):
                    container_args['root'] = self.get_full_name()
                try:
                    container = container_initializer(**container_args)
                except AttributeError as ae:
                    raise Exception("Problem initializing " + roletype + "_" + rolename
                                    + ": " + str(container_initializer)) from ae
            elif isinstance(container_args, BaseObject):
                raise Exception(str(container_args.__class__) + " not a recognized" +
                                " instance of " + str(container_initializer) +
                                " (did you use relative instead of absolute imports?)")
            else:
                raise Exception(str(container_args) + "not a dict or not a recognized" +
                                "instance of " + str(container_initializer))
            container.check_role(roletype, rolename)
            setattr(self, rolename, container)

    def assign_roles(self, roletype, other_obj, **kwargs):
        """
        Assign copies of the roles of another BaseObject to the current object.

        Used in object copying to ensure copies have the same attributes as the current.

        Parameters
        ----------
        roletype : str
            Roletype to assign.
        other_obj : BaseObject
            Object to assign from
        **kwargs : kwargs
            Any keyword arguments to the role copy method.

        Examples
        --------
        >>> exec(example_object_code)
        >>> ex = ExampleObject(s={'x': 1.0, 'y': 2.0})
        >>> ex2 = ExampleObject(s={'x': 3.0, 'y': 4.0})
        >>> ex.assign_roles('container', ex2)
        >>> ex.s
        ExampleState(x=3.0, y=4.0)

        Note that these these roles should be independent after assignment:

        >>> ex.s.x = 4.0
        >>> ex.s
        ExampleState(x=4.0, y=4.0)
        >>> ex2.s
        ExampleState(x=3.0, y=4.0)
        """
        roles = getattr(self, roletype+'s')
        for role in roles:
            other_role = getattr(other_obj, role)
            if bool(signature(other_role.copy).parameters):
                other_role_copy = other_role.copy(**kwargs)
            else:
                other_role_copy = other_role.copy()
            setattr(self, role, other_role_copy)

    def reset(self):
        for role in self.get_all_roles():
            getattr(self, role).reset()

    def init_role_dict(self, spec, name_end="s", set_attr=False):
        """
        Create a collection dict for the attribute 'spec'.

        Works by finding all attributes from the obj's parameter with the name 'spec' in
        them and adding them to the dict. Adds the dict to the object.

        Used in more flexible classes like Coords and Geom to enable properties to be
        set via parameters.

        Parameters
        ----------
        spec : str
            Name of the attributes to initialize
        name_end: str
            Last letter or set of lettersof the attribute used to give the role name to where 
            the dictionary will be placed. It may be "ions" with collections. Default is 's'.
        set_attr : bool
            Whether to also add the individual attributes attr to the obj
        """
        spec_len = len(spec) + 1
        specs = {p[spec_len:]: self.p[p] for p in self.p.__fields__ if spec in p}
        specname = spec + name_end
        setattr(self, specname, specs)
        if set_attr:
            for s_name in specs:
                setattr(self, s_name, specs[s_name])

    def init_indicators(self):
        """Find all indicator methods and initialize in .indicator tuple."""
        self.indicators = tuple([at[9:] for at in dir(self)
                                 if at.startswith('indicate_')])

    def get_indicators(self):
        """
        Get the names of the indicators.

        Returns
        -------
        indicators : dict
            dict of indicator names and their associated method handles.
        """
        return {i: getattr(self, 'indicate_'+i) for i in self.indicators}

    def return_true_indicators(self, time=0.0):
        """
        Get list of indicators.

        Parameters
        ----------
        time : float
            Time to execute the indicator method at.

        Returns
        -------
        list
            List of inticators that return true at time
        """
        return [f for f, ind in self.get_indicators().items()
                if (bool(signature(ind).parameters) and ind(time))
                or (not bool(signature(ind).parameters) and ind())]

    def init_indicator_hist(self, h, timerange, track):
        """
        Create a history for an object with indicator methods (e.g., obj.indicate_XX).

        Parameters
        ----------
        h : History
            History of Function/Flow/Model object with indicators appended in h['i']
        timerange : iterable, optional
            Time-range to initialize the history over. The default is None.
        track : list/str/dict, optional
            argument specifying attributes for :func:`get_sub_include'.
            The default is None.

        Returns
        -------
        h : History
            History of states with structure {'XX':log} for indicator `obj.indicate_XX`
        """
        sub_track = get_sub_include('i', track)
        if sub_track:
            indicators = self.get_indicators()
            if indicators:
                h['i'] = History()
                for i, val in indicators.items():
                    h['i'].init_att(i, val, timerange, sub_track, dtype=bool)

    def get_all_possible_track(self):
        """Get all possible tracking options."""
        rs = self.get_all_roles(with_immutable=False)
        return rs + ['i'] + self.rolevars

    def get_default_roletypes(self, *roletypes, no_flows=False):
        if not roletypes or roletypes[0] == 'all':
            roletypes = [*self.roletypes]
        elif roletypes[0] == 'none':
            roletypes = []
        if no_flows and 'flows' in roletypes:
            roletypes.remove('flows')
        return roletypes

    def get_roles(self, *roletypes, with_immutable=True, no_flows=False, **kwargs):
        """Get all roles."""
        roletypes = self.get_default_roletypes(*roletypes, no_flows=no_flows)
        return [role for roletype in roletypes
                for role in getattr(self, roletype+'s', [])
                if with_immutable or role not in self.immutable_roles]

    def get_flex_role_objs(self, *flexible_roles, flex_prefixes=False):
        """Get the objects in flexible roles (e.g., functions, flows, components)."""
        if not flexible_roles:
            flexible_roles = self.flexible_roles
        role_objs = {}
        for role in flexible_roles:
            roledict = getattr(self, role)
            if isinstance(roledict, list) or isinstance(roledict, tuple):
                roledict = {k: getattr(self, k) for k in roledict}
            if not flex_prefixes:
                role_objs.update(roledict)
            else:
                role_objs.update({role+'.'+k: v for k, v in roledict.items()})
        return role_objs

    def copy_mut_containers(self):
        """Return copies of the mutable containers."""
        return {k: v.copy()
                for k, v in self.get_roles_as_dict('container',
                                                   with_immutable=False).items()}

    def get_roles_as_dict(self, *roletypes, with_immutable=True, with_prefix=False,
                          flex_prefixes=False, with_flex=True, no_flows=False,
                          **kwargs):
        """Return all roles and their objects as a dict."""
        roletypes = self.get_default_roletypes(*roletypes)
        flex_roles = [r+'s' for r in roletypes if r+'s' in self.flexible_roles]
        if with_flex:
            flex_roles = self.get_flex_role_objs(*flex_roles,
                                                 flex_prefixes=flex_prefixes)
        else:
            flex_roles = {}
        non_flex_roletypes = [r for r in roletypes if r+'s' not in self.flexible_roles]
        if not non_flex_roletypes:
            non_flex_roletypes = 'none'

        roles = self.get_roles(*non_flex_roletypes,
                               with_immutable=with_immutable,
                               no_flows=no_flows)
        non_flex_roles = {role: getattr(self, role) for role in roles}
        all_roles = {**flex_roles, **non_flex_roles}
        if with_prefix:
            all_roles = {self.name+"."+k: v for k, v in all_roles.items()}
        return all_roles

    def get_roledicts(self, *roledicts, with_immutable=True):
        """Get all roles in roledicts."""
        if not roledicts:
            roledicts = self.roledicts
        return [role for roledict in roledicts for role in getattr(self, roledict)
                if with_immutable or roledict not in self.immutable_roles]

    def get_all_roles(self, with_immutable=True):
        """Get all roles in the object."""
        roles = self.get_roles(with_immutable=with_immutable)
        roledict_roles = self.get_roledicts(with_immutable=with_immutable)
        rolevars = [role for role in self.rolevars
                    if with_immutable or role not in self.immutable_roles]
        return roles + roledict_roles + rolevars

    def get_vars(self, *variables, trunc_tuple=True):
        """
        Get variable values in the object.

        Parameters
        ----------
        *variables : list/string
            Variables to get from the model. Can be specified as a list
            ['fxnname2', 'comp1', 'att2'], or a str 'fxnname.comp1.att2'

        Returns
        -------
        variable_values: tuple
            Values of variables. Passes (non-tuple) single value if only one variable.
        """
        if isinstance(variables, str):
            variables = [variables]
        variable_values = [None]*len(variables)
        for i, var in enumerate(variables):
            if isinstance(var, str):
                var = var.split(".")
            if var[0] in self.roletypes + [rt+"s" for rt in self.roletypes]:
                f = self.get_roles_as_dict()[var[1]]
                var = var[2:]
            elif var[0] in self.get_roles():
                f = self.get_roles_as_dict()[var[0]]
                var = var[1:]
            else:
                f = self
            if var:
                variable_values[i] = get_var(f, var)
            else:
                variable_values[i] = f
        if len(variable_values) == 1 and trunc_tuple:
            return variable_values[0]
        else:
            return tuple(variable_values)

    def get_memory(self):
        """
        Get the memory taken up by the object and its containing roles.

        Returns
        -------
        mem : float
            Approximate memory taken by the object.
        """
        mem = sys.getsizeof(self)
        mem_profile = {'base': mem}
        roles_to_check = self.get_roles_as_dict(no_flows=True)
        for rolename, roleobj in roles_to_check.items():
            mem_profile[rolename] = get_memory(roleobj)
            mem += mem_profile[rolename]
        return mem, mem_profile

    def create_hist(self, timerange):
        """
        Create state history of the object over the timerange.

        Parameters
        ----------
        timerange : array
            Numpy array of times to initialize in the dictionary.

        Returns
        -------
        hist : History
            A history of each recorded block property over the given timerange.
        """
        if hasattr(self, 'h') and self.h and len([*self.h.values()][0]) == len(timerange):
            return self.h
        else:
            track = self.track
            hist = History()
            if track:
                self.init_indicator_hist(hist, timerange, track)
                other_tracks = [t for t in track if t not in ('i', 'flows')]
                for at in other_tracks:
                    at_track = get_sub_include(at, track)
                    attr = getattr(self, at, False)
                    if hasattr(self, at):
                        if hasattr(attr, 'create_hist'):
                            if 'track' in signature(attr.create_hist).parameters:
                                at_h = attr.create_hist(timerange, at_track)
                            else:
                                at_h = attr.create_hist(timerange)
                            if at_h:
                                hist[at] = at_h
                        elif isinstance(attr, np.ndarray):
                            hist.init_att(at, attr, timerange, at_track,
                                          dtype=np.ndarray)
                        else:
                            hist.init_att(at, attr, timerange, at_track)
            return hist.flatten()

    def find_mutables(self):
        """Return list of mutable roles."""
        return [v for v in self.get_roles_as_dict(with_immutable=False).values()
                if not ismethod(v)]

    def return_mutables(self):
        """
        Return all mutable values in the block.

        Used in static propagation steps to check if the block has changed.

        Returns
        -------
        mutables : tuple
            tuple of all mutable roles for the object.
        """
        return tuple([mut.return_mutables() if hasattr(mut, 'return_mutables')
                      else mut for mut in self.find_mutables()])

    def get_node_attrs(self, roles=['container'], with_immutable=False,
                       time=0.0, indicators=True, obj=False):
        """
        Get attributes from the node to attach to a graph node.

        Parameters
        ----------
        g : nx.Graph
            Graph where the object is a node.
        roles : list, optional
            Roles to set as node attributes. The default is ['container'].
        with_immutable : bool, optional
            Whether to include immutable roles. The default is False.
        time : float, optional
            Time to evaluate indicators at. The default is None.
        indicators : bool, optional
            Whether to evaluate indicators. The default is True.

        Examples
        --------
        >>> exec(example_object_code)
        >>> ExampleObject().get_node_attrs()
        {'s': ExampleState(x=1.0, y=1.0), 'indicators': ['y_over_t']}
        >>> ExampleObject().get_node_attrs(roles=["none"])
        {'indicators': ['y_over_t']}
        """
        attdict = self.get_roles_as_dict(*roles, with_immutable=with_immutable)

        if indicators:
            attdict['indicators'] = self.return_true_indicators(time)
        if obj:
            attdict['obj'] = self
        return attdict

    def set_node_attrs(self, g, with_root=True, **kwargs):
        """
        Set attributes of the object to a graph.

        Parameters
        ----------
        g : nx.Graph
            Graph where the object is a node.
        kwargs : kwargs
            Arguments to get_node_attrs.
        """
        attdict = self.get_node_attrs(**kwargs)
        g.nodes[self.get_full_name(with_root=with_root)].update(attdict)

    def get_att_roletype(self, attname, raise_if_none=False):
        """
        Get the roletype for a given attribute.

        Parameters
        ----------
        attname : str
            Name of a variable in the object.

        Returns
        -------
        att_roletype : str
            Type of role that initiated the variable.

        Examples
        --------
        >>> exec(example_object_code)
        >>> ExampleObject().get_att_roletype("s")
        'container'
        """
        att_roletype = ''
        for roletype in self.roletypes:
            if attname in self.get_roles(roletype):
                att_roletype = roletype
        if not att_roletype and raise_if_none:
            raise Exception("Role not found for attribute: "+attname)
        elif not att_roletype:
            att_roletype = 'variable'
        return att_roletype

    def get_role_edgetype(self, attname, raise_if_none=False):
        """
        Get the edgetype for a given variable in the object.

        Parameters
        ----------
        attname : TYPE
            DESCRIPTION.
        raise_if_none : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        edgetype : str
            Edgetype to give the contained object.
        """
        roletype = self.get_att_roletype(attname, raise_if_none=raise_if_none)
        roleobj = self.get_vars(attname)
        if roletype == 'container':
            return "containment"
        elif hasattr(roleobj, 'get_full_name'):
            if self.get_full_name() in roleobj.root:
                return "containment"
            else:
                return "aggregation"
        elif roletype == 'flow':
            return "flow"
        elif roletype == 'variable':
            return "containment"
        else:
            raise Exception("Unknown edge type for role: " + roletype)

    def _prep_graph(self, g=None, name='', **kwargs):
        g = add_node(self, g=g, **kwargs)
        if not name:
            name = self.get_full_name()
        return g, name

    def create_graph(self, g=None, name='', with_methods=True, with_root=True,
                     with_inheritance=False, end_at_fmdtools=True, **kwargs):
        """
        Create a networkx graph view of the Block.

        Parameters
        ----------
        g : nx.Graph
            Existing networkx graph (if any). Default is None.
        name : str
            Name of the node. Default is '', which uses the get_full_name().
        with_methods : bool
            Whether to include methods. Default is True.
        end_at_fmdtools : bool
            Whether to end inheritance graph at first fmdtools class. Default is True.
        **kwargs : kwargs
            Keyword arguments to create_role_subgraph.

        Returns
        -------
        g : nx.Graph
            Networkx graph.
        """
        g, name = self._prep_graph(g=g, name=name, **kwargs)
        self.create_role_subgraph(g=g, name=name, with_inheritance=with_inheritance,
                                  with_methods=with_methods,
                                  end_at_fmdtools=end_at_fmdtools, **kwargs)
        if with_methods:
            self.create_method_subgraph(g=g, name=name, **kwargs)
        if not with_root:
            remove_base(g, name)
        elif with_inheritance:
            g = create_inheritance_subgraph(self, g, end_at_fmdtools=end_at_fmdtools)
        return g

    def create_method_subgraph(self, g=None, name='', **kwargs):
        """Create networkx graph of the Block and its methods."""
        g, name = self._prep_graph(g=g, name=name, **kwargs)
        for methodname, methodobj in get_methods(self).items():
            mname = get_obj_name(methodobj)
            add_node(methodobj, g=g, name=mname,
                     classname="method", nodetype="method", **kwargs)
            add_edge(g, name, mname, methodname, "containment")
        return g

    def add_subgraph_edges(self, g, roles_to_connect=[], **kwargs):
        """Add non-role edges to the graph for the roles."""
        if roles_to_connect:
            self.create_role_con_edges(g, roles_to_connect=roles_to_connect, **kwargs)

    def create_role_con_edges(self, g, roles_to_connect=[], role="connection",
                              edgetype="connection"):
        """Connect roles at the same level of hierarchy."""
        basename = self.get_full_name()
        roledict = self.get_roles_as_dict(*roles_to_connect)
        for rolename, roleobj in roledict.items():
            name = get_obj_name(roleobj, rolename, basename)
            for rolename2, roleobj2 in roledict.items():
                name2 = get_obj_name(roleobj2, rolename2, basename)
                if not ((name2, name) in [*g.edges]) and name2 != name:
                    add_edge(g, name, name2, role, edgetype)

    def create_role_subgraph(self, g=None, name='', role_nodes=["all"], recursive=False,
                             with_containment=True, with_inheritance=False,
                             with_methods=True, with_subgraph_edges=True,
                             end_at_fmdtools=True, **kwargs):
        """
        Create a networkx graph view of the Block and its roles.

        Parameters
        ----------
        g : nx.Graph
            Existing networkx graph (if any). Default is None.
        name : str
            Name of the node. Default is '', which uses the get_full_name().
        role_nodes : list, optional
            Roletypes to include in the subgraph. The default is ["all"].
        recursive : bool, optional
            Whether to add nodes to the subgraph recursively from contained objects.
        with_containment : bool
            Whether to include containment edges. Default is True.
        with_inheritance : bool
            Whether to include class inheritance subgraphs. Default is False.
            The default is False.
        with_methods : bool
            Whether to include methods as nodes. Default is False.
        with_subgraph_edges : bool
            Whether to include subgraph edges, e.g. function/flow containment in an
            architecture graph.
        end_at_fmdtools : bool
            Whether to end inheritance graph at first fmdtools class. Default is True.
        **kwargs : kwargs
            kwargs to add_node

        Returns
        -------
        g : nx.Graph
            Networkx graph.
        """
        g, name = self._prep_graph(g=g, name=name, **kwargs)
        roledict = self.get_roles_as_dict(*role_nodes, flex_prefixes=True)
        for rolename, roleobj in roledict.items():
            subname = get_obj_name(roleobj, role=rolename, basename=name)
            add_node(roleobj, g, name=subname, **kwargs)
            if with_containment:
                edgetype = self.get_role_edgetype(rolename)
                add_edge(g, name, subname, rolename, edgetype)
            if with_inheritance:
                g = create_inheritance_subgraph(roleobj, g=g, name=subname,
                                                end_at_fmdtools=end_at_fmdtools)
            if recursive and hasattr(roleobj, 'create_graph'):
                roleobj.create_graph(g=g, role_nodes=role_nodes, recursive=recursive,
                                     name=subname, with_containment=with_containment,
                                     with_inheritance=with_inheritance,
                                     with_methods=with_methods,
                                     end_at_fmdtools=end_at_fmdtools, **kwargs)
        if with_containment and with_subgraph_edges:
            self.add_subgraph_edges(g, **kwargs)
        return g

    def as_modelgraph(self, gtype=ObjectGraph, **kwargs):
        """Create and return the corresponding ModelGraph for the Object."""
        return gtype(self, **kwargs)


def check_pickleability(obj, verbose=True, try_pick=False, pause=0.2):
    """Check to see which attributes of an object will pickle (and parallelize)."""
    from fmdtools.define.container.base import check_container_pick, BaseContainer
    unpickleable = []
    try:
        itera = vars(obj)
    except TypeError:
        itera = {a: getattr(obj, a) for a in obj.__slots__}
    for name, attribute in itera.items():
        print(name)
        time.sleep(pause)
        try:
            if (isinstance(attribute, BaseContainer)
                    or (isclass(attribute)
                        and issubclass(attribute, BaseContainer))):
                if not check_container_pick(attribute):
                    unpickleable.append(name)
            elif isinstance(attribute, BaseObject):
                if any(check_pickleability(attribute, verbose=False)):
                    unpickleable.append(name)
            elif not dill.pickles(attribute):
                unpickleable = unpickleable + [name]
        except ValueError as e:
            raise ValueError("Problem in " + name +
                             " with attribute " + str(attribute)) from e
        if try_pick:
            try_pickle_obj(attribute)
    if try_pick:
        try_pickle_obj(obj)
    if verbose:
        if unpickleable:
            print("The following attributes will not pickle: " + str(unpickleable))
        else:
            print("The object is pickleable")
    return unpickleable


def try_pickle_obj(obj):
    """Try to pickle an object. Raise exception if it doesn't work."""
    from pickle import PicklingError
    try:
        a = pickle.dumps(obj)
        b = pickle.loads(a)
    except PicklingError:
        raise Exception(obj.name + " will not pickle")


def init_obj(name, objclass=BaseObject, track='default', as_copy=False, **kwargs):
    """
    Initialize an object.

    Enables one to instantiate different types of objects with given
    states/parameters or pass an already-constructured object.

    Parameters
    ----------
    name : str
        Name to give the flow object
    objclass: class or object
        Class inheriting from BaseObject, or already instantiated object.
        Default is BaseObject.
    track: str/dict
        Which model states to track over time (overwrites mdl.default_track).
        Default is 'default'
        Options:

        - 'default'
        - 'all'
        - 'none'
        - or a dict of form ::

            {'functions':{'fxn1':'att1'}, 'flows':{'flow1':'att1'}}
    as_copy: bool
        If an object is provided for objclass, whether to copy that object (or just pass it).
        Default is False.
    **kwargs :dict
        Other specialized roles to overrride
    """
    if not isclass(objclass):
        if not as_copy:
            fl = objclass
            fl.init_track(track)
        else:
            fl = objclass.copy(name=name, track=track, **kwargs)
    else:
        try:
            fl = objclass(name=name, track=track, **kwargs)
        except TypeError as e:
            raise TypeError("Poorly specified class "+str(objclass) +
                            " (or poor arguments) "+str(kwargs)) from e
    return fl


if __name__ == "__main__":
    exec(example_object_code)
    ExampleObject().get_roles_as_dict("none")
    import doctest
    doctest.testmod(verbose=True)

