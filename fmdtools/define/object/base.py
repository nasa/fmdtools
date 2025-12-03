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

from fmdtools.define.base import get_var, set_var, get_methods, get_obj_name, get_memory
from fmdtools.define.base import get_repr, get_dict_repr
from fmdtools.analyze.common import get_sub_include
from fmdtools.analyze.history import History
from fmdtools.analyze.graph.model import add_node, add_edge, remove_base, ModelGraph
from fmdtools.analyze.graph.model import create_inheritance_subgraph

import dill
import pickle
import time
import sys
import numpy as np
from inspect import signature, isclass, ismethod, isfunction


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


def find_roletype_initiators(attrlist, roletype):
    """Find initiators from list that start with 'roletype_'."""
    return tuple([at[len(roletype)+1:]
                  for at in attrlist if at.startswith(roletype+'_')])


class BaseType(type):
    """
    Base type for BaseObject-based classes.

    BaseType enables the use of __slots__ without defining explicit slots by relying
    on the defined class roletypes (e.g., containers, flows, etc) This enables lower
    memory use and faster access without requiring the user to define __slots__ for
    their classes.
    """

    def __new__(cls, name, bases, dct):
        """Construct the type from its class variables."""
        slots = list(dct.get('__slots__', []))  # get defined __slots__ if available
        base_slots = []
        roletypes = []
        for base in bases:
            if hasattr(base, '__slots__'):
                base_slots += base.__slots__
            if hasattr(base, 'roletypes'):
                roletypes = base.roletypes

        if 'roletypes' in dct:
            roletypes = dct['roletypes']

        role_slots = []  # get slots to be inferred by roletypes (e.g., containers)
        for roletype in roletypes:
            roles = list(find_roletype_initiators(dct, roletype))
            role_slots += roles
            # add to class roles tuple
            sub_roles = []
            for base in bases:
                for sub_role in getattr(base, roletype+"s", []):
                    if sub_role not in sub_roles:
                        sub_roles.append(sub_role)
            for r in roles:
                if r not in sub_roles:
                    sub_roles.append(r)
            dct[roletype+'s'] = tuple(sub_roles)

        all_slots = set(slots+base_slots+role_slots)  # combine slots
        inherit_dict = "__dict__" in base_slots

        if inherit_dict and "__dict__" in all_slots:
            all_slots.remove("__dict__")
        dct['__slots__'] = tuple(all_slots)

        try:
            return type.__new__(cls, name, bases, dct)
        except Exception as e:
            raise TypeError("Incorrect class specification for "+name) from e


example_object_code = """
from fmdtools.define.container.state import ExampleState
class ExampleObject(BaseObject):
    container_s = ExampleState
    def indicate_high_x(self):
        return bool(self.s.x > 1.0)
    def indicate_y_over_x(self):
        return bool(self.s.y > self.s.x)
"""


class BaseObject(metaclass=BaseType):
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
    ...        return bool(self.s.x > 1.0)
    ...    def indicate_y_over_x(self):
    ...        return bool(self.s.y > self.s.x)


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
    ('high_x', 'y_over_x')

    And are used to evaluate conditions, e.g.:

    >>> ex.indicate_high_x()
    False
    >>> ex2.indicate_high_x()
    True

    Time may be used as an optional argument to indicators:

    >>> ex.indicate_y_over_x()
    False
    >>> ex2.return_true_indicators()
    ['high_x', 'y_over_x']

    A history may be created using create_hist:

    >>> ex.create_hist([0.0, 1.0])
    i.high_x:                       array(2)
    i.y_over_x:                     array(2)

    Note that adding roles to the class often means modifying default_track.
    Initializing all possible using the 'all' option:

    >>> ex = ExampleObject(track='all')
    >>> ex.create_hist([0.0, 1.0])
    i.high_x:                       array(2)
    i.y_over_x:                     array(2)
    s.x:                            array(2)
    s.y:                            array(2)
    """

    __slots__ = ('name', 'indicators', 'track', 'root', 'mutables', '_mutobjs')
    roletypes = ['container']
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
        self.name = self.create_name(name)
        self.root = root
        self.init_indicators()
        self.init_track(track)
        self.init_roletypes(*roletypes, **kwargs)
        self.check_slots()
        self._mutobjs = []
        self.mutables = ()

    def create_repr(self, rolenames=[], with_classname=True, with_name=True,
                    one_line=False):
        """
        Provide a repl-friendly string showing the states of the Object.

        Returns
        -------
        repr: str
            console string
        """
        if hasattr(self, 'name'):
            namestrs = []
            if with_name:
                namestrs.append(getattr(self, 'name', ''))
            if with_classname:
                namestrs.append(self.__class__.__name__)
            objstr = " ".join(namestrs)
            if not rolenames:
                rolenames = self.get_roles(with_immutable=False, with_flex=False)
            roledict = {k: getattr(self, k) for k in rolenames if hasattr(self, k)}
            objstr += get_dict_repr(roledict, one_line=one_line)
            return objstr
        else:
            return 'New uninitialized '+self.__class__.__name__

    def __repr__(self):
        return self.create_repr(rolenames=[])

    @classmethod
    def create_name(cls, name=''):
        """Create a name for the object (default is class name)."""
        if name:
            return name
        else:
            return cls.__name__.lower()

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
                if t in self.roletypes:
                    self.track.extend(getattr(self, t+"s"))
                elif t in self.flexible_roles:
                    self.track.append(t+'s')
                else:
                    self.track.append(t)

    def init_roletypes(self, *roletypes, initializer=None, **kwargs):
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
            self.init_roles(roletype, initializer=initializer, **kwargs)

    def find_roletype_initiators(self, roletype):
        return find_roletype_initiators(dir(self), roletype)

    def get_full_name(self, with_root=True):
        """Get the full name of the object (root + name)."""
        if self.root and with_root:
            return self.root + "." + self.name
        else:
            return self.name

    def init_roles(self, roletype, initializer=None, **kwargs):
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
        # initialize roles and add as attributes to the object
        for rolename in getattr(self, roletype+'s'):
            if not initializer:
                obj_initializer = getattr(self, roletype+'_'+rolename)
                obj_args = kwargs.get(rolename, dict())
            else:
                obj_initializer = initializer
                obj_args = getattr(self, roletype+'_'+rolename)

            if ismethod(obj_initializer) or isfunction(obj_initializer):
                obj = obj_initializer(*obj_args)
            elif isinstance(obj_args, obj_initializer):
                obj = obj_args
            elif isinstance(obj_args, dict):
                default_args = getattr(self, 'default_'+rolename, dict())
                obj_args = {**default_args, **obj_args}
                if issubclass(obj_initializer, BaseObject):
                    obj_args['root'] = self.get_full_name()
                    obj_args['track'] = get_sub_include(rolename, self.track)
                try:
                    obj = obj_initializer(**obj_args)
                except AttributeError as ae:
                    raise Exception("Problem initializing " + roletype + "_" + rolename
                                    + ": " + str(initializer)) from ae
            elif isinstance(obj_args, BaseObject):
                raise Exception(str(obj_args.__class__) + " not a recognized" +
                                " instance of " + str(initializer) +
                                " (did you use relative instead of absolute imports?)")
            else:
                raise Exception(str(obj_args) + "not a dict or not a recognized "
                                + "instance of " + str(obj_initializer))
            if hasattr(obj, 'check_role'):
                obj.check_role(roletype, rolename)
            setattr(self, rolename, obj)

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

        >>> ex.s.x = np.float64(4.0)
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

    def return_true_indicators(self):
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
        return [f for f, ind in self.get_indicators().items() if ind()]

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

    def get_default_roletypes(self, *roletypes, exclude=[], with_flex=True):
        """
        Get the default role types for the object.

        Parameters
        ----------
        *roletypes : str
            Types of roles (e.g. 'containers' or 'flows'). If not provided, all role
            types are included.
        exclude : list
            Roletypes to exclude. Default is [].
        with_flex : bool, optional
            Whether to include flexible roletypes (e.g., functions).
            The default is True.

        Returns
        -------
        roletypes : list
            List of default role types to iterate over, e.g. ['containers', 'flows'].
        """
        if not roletypes or roletypes[0] == 'all':
            roletypes = [*self.roletypes]
            if with_flex:
                for flex_role in self.flexible_roles:
                    if flex_role not in roletypes:
                        roletypes.append(flex_role)
        elif roletypes[0] == 'none':
            roletypes = []
        else:
            roletypes = list(roletypes)
        for rt in exclude:
            if rt in roletypes:
                roletypes.remove(rt)
        if not with_flex:
            for flex_role in self.flexible_roles:
                if flex_role in roletypes:
                    roletypes.remove(flex_role)
        return roletypes

    def get_roles(self, *roletypes, with_immutable=True, with_flex=True,
                  with_prefix=False, exclude=[], **kwargs):
        """
        Get all roles from the object.

        Parameters
        ----------
        *roletypes : str
            Types of roles to get. If not provided, gets default roletypes.
        with_immutable : bool, optional
            Whether to include immutable roles (e.g., parameters at container_p). The
            default is True.
        with_flex : bool, optional
            Whether to include flexible roles. The default is True.
        with_prefix : bool, optional
            Wheter to provide a prefix for the role.
        **kwargs : kwargs
            (unused) keyword arguments

        Returns
        -------
        roles : list
            List of roles in the object, e.g. ['p', 's' 'r'] for an object with the
            parameter, state, and rand roles filled.
        """
        roletypes = self.get_default_roletypes(*roletypes,
                                               exclude=exclude, with_flex=with_flex)
        return [roletype+"s."+role if with_prefix else role for roletype in roletypes
                for role in getattr(self, roletype+'s', [])
                if (with_immutable or role not in self.immutable_roles) and
                role not in exclude]

    def get_roles_values(self, *roletypes, with_immutable=True, with_flex=True,
                         exclude=[], with_method=False, **kwargs):
        """Get the objects associated with each role."""
        roletypes = self.get_default_roletypes(*roletypes,
                                               exclude=exclude, with_flex=with_flex)
        roleiters = [getattr(self, roletype+"s") for roletype in roletypes]
        return [roleiter[role] if isinstance(roleiter, dict) else getattr(self, role)
                for roleiter in roleiters for role in roleiter
                if (with_immutable or role not in self.immutable_roles) and
                role not in exclude]

    def get_flex_role_objs(self, *flexible_roles, flex_prefixes=False):
        """
        Get the objects in flexible roles (e.g., functions, flows, components).

        Parameters
        ----------
        *flexible_roles : str
            Names of flexible roles (e.g., 'fxns', 'flows').
        flex_prefixes : bool, optional
            Whether to include the prefixes in the keys of the returned dictionary. The
            default is False.

        Returns
        -------
        flex_role_objs : dict
            Dict of flexible role objects with structure {'rolename': roleobject}
        """
        if not flexible_roles:
            flexible_roles = self.flexible_roles
        return self.get_roles_as_dict(*flexible_roles, with_flex=True,
                                      with_prefixes=flex_prefixes)

    def copy_mutes(self, *mute_types, **kwargs):
        """Return copies of the mutable containers."""
        return {k: v.copy()
                for k, v in self.get_roles_as_dict(*mute_types, with_immutable=False,
                                                   **kwargs).items()}

    def get_roles_as_dict(self, *roletypes, with_immutable=True, with_prefix=False,
                          with_flex=True, exclude=[], **kwargs):
        """
        Return all roles and their objects as a dict.

        Parameters
        ----------
        *roletypes : str
            Types of roles to get (e.g. "container"). If not provided, gets default
            roletypes.
        with_immutable : bool, optional
            Whether to include immutable roles (e.g., parameters at container_p). The
            default is True.
        with_prefix : bool, optional
            Whether to include the prefixes in the keys of the returned dictionary. The
            default is False.
        exclude : bool, optional
            Roletypes and roles to exclude. The default is [].
        with_flex : bool, optional
            Whether to include flexible roles. The default is True.
        **kwargs : kwargs
            (unused) keyword arguments

        Returns
        -------
        roles : list
            List of roles in the object, e.g. ['p', 's' 'r'] for an object with the
            parameter, state, and rand roles filled.
        """
        roletypes = self.get_default_roletypes(*roletypes, exclude=exclude,
                                               with_flex=with_flex)
        role_objs = {}
        for roletype in roletypes:
            roledict = getattr(self, roletype+"s")
            if isinstance(roledict, list) or isinstance(roledict, tuple):
                roledict = {k: getattr(self, k) for k in roledict}
            roledict = {k: v for k, v in roledict.items()
                        if (with_immutable or k not in self.immutable_roles)
                        and k not in exclude and getattr(v, 'name', '') not in exclude}
            if not with_prefix:
                role_objs.update(roledict)
            else:
                role_objs.update({roletype+'s.'+k: v for k, v in roledict.items()})
        return role_objs

    def get_all_roles(self, with_immutable=True):
        """Get all roles in the object."""
        rolenames = []
        for roletype in self.roletypes:
            rolenames.extend(self.find_roletype_initiators(roletype))
        if with_immutable:
            return rolenames
        else:
            return [r for r in rolenames if r not in self.immutable_roles]

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
        all_roles = self.get_roles_as_dict()
        for i, var in enumerate(variables):
            f, var = self._get_role_call(var, all_roles)

            if var:
                variable_values[i] = get_var(f, var)
            else:
                variable_values[i] = f
        if len(variable_values) == 1 and trunc_tuple:
            return variable_values[0]
        else:
            return tuple(variable_values)

    def set_vars(self, *args, **kwargs):
        """
        Set variables in the model to set values (useful for optimization, etc.).

        Parameters
        ----------
        varlist : list of lists/tuples
            List of variables to set, with possible structures:
                [['fxnname', 'att1'], ['fxnname2', 'comp1', 'att2'], ['flowname', 'att3']]
                ['fxnname.att1', 'fxnname.comp1.att2', 'flowname.att3']
        varvalues : list
            List of values corresponding to varlist
        kwargs : kwargs
            attribute-value pairs. If provided, must be passed using ** syntax:
            mdl.set_vars(**{'fxnname.varname':value})
        """
        if len(args) > 0:
            varlist = args[0]
            varvalues = args[1]
            if isinstance(varlist, str):
                varlist = [varlist]
            if type(varvalues) in [str, float, int]:
                varvalues = [varvalues]
            if len(varlist) != len(varvalues):
                raise Exception("length of varlist and varvalues do not correspond: "
                                + str(len(varlist)) + ", "+str(len(varvalues)))
        else:
            varlist = []
            varvalues = []
        if kwargs:
            varlist = varlist + [*kwargs.keys()]
            varvalues = varvalues + [*kwargs.values()]
        all_roles = self.get_roles_as_dict()
        for i, var in enumerate(varlist):
            if var == 'seed':
                self.update_seed(seed=varvalues[i])
            else:
                f, var = self._get_role_call(var, all_roles)
                set_var(f, var, varvalues[i])

    def _get_role_call(self, var, all_roles):
        """Get obj to use set_var and get_var on."""
        if isinstance(var, str):
            var = var.split(".")
        if var[0] == self.name:
            var = var[1:]
        if var[0] in self.roletypes + [rt+"s" for rt in self.roletypes]:
            f = all_roles[var[1]]
            var = var[2:]
        elif var[0] in all_roles:
            f = all_roles[var[0]]
            var = var[1:]
        else:
            f = self
        return f, var

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
        roles_to_check = self.get_roles_as_dict(exclude=['flow'])
        for rolename, roleobj in roles_to_check.items():
            mem_profile[rolename] = get_memory(roleobj)
            mem += mem_profile[rolename]
        return mem, mem_profile

    def create_hist(self, timerange, track=None):
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
        track : Track
            Argument to set track. Only used if history has not already been created.
            Default is None.
        """
        if hasattr(self, 'h') and self.h:
            len_timerange = len(timerange)
            if len_timerange != len([*self.h.values()][0]):
                self.h = self.h.cut(reshape_to=len_timerange)
            return self.h
        else:
            if track is not None:
                self.init_track(track)
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

    def find_mutables(self, exclude=[]):
        """Return list of mutable roles."""
        if not self._mutobjs:
            roleobjs = self.get_roles_values(with_immutable=False, exclude=exclude)
            roleobjs.reverse()
            self._mutobjs = [v for v in roleobjs
                             if not ismethod(v)
                             and not getattr(v, 'name', '') in exclude]
        return self._mutobjs

    def return_mutables(self, exclude=[]):
        """
        Return all mutable values in the block.

        Used in static propagation steps to check if the block has changed.

        Returns
        -------
        mutables : tuple
            tuple of all mutable roles for the object.
        """
        return tuple([mut.return_mutables() if hasattr(mut, 'return_mutables')
                      else mut for mut in self.find_mutables(exclude=exclude)])

    def set_mutables(self, exclude=[]):
        """Set the current mutables of the object (to check against later)."""
        self.mutables = self.return_mutables(exclude=exclude)

    def has_changed(self, update=False, exclude=[]):
        """Check if the mutables of the object have changed."""
        new_mutables = self.return_mutables(exclude=exclude)
        try:
            is_changed = new_mutables != self.mutables
        except ValueError as e:
            raise Exception("Invalid mutables in "+self.name) from e
        if update:
            self.mutables = new_mutables
        return is_changed

    def get_node_attrs(self, roles=['container'], with_immutable=False,
                       indicators=True, obj=False):
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
        indicators : bool, optional
            Whether to evaluate indicators. The default is True.

        Examples
        --------
        >>> exec(example_object_code)
        >>> exo = ExampleObject(s={'y': 2.0})
        >>> exo.get_node_attrs()
        {'s': ExampleState(x=1.0, y=2.0), 'indicators': ['y_over_x']}
        >>> exo.get_node_attrs(roles=["none"])
        {'indicators': ['y_over_x']}
        """
        attdict = self.get_roles_as_dict(*roles, with_immutable=with_immutable)

        if indicators:
            attdict['indicators'] = self.return_true_indicators()
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
        """Prepare graph by adding self as a node and returning name."""
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
        itera = {a: getattr(obj, a) for a in dir(obj)
                 if hasattr(obj, a) and (not ismethod(getattr(obj, a)) and not a.startswith("__"))}
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
            elif isinstance(attribute, History) or isinstance(attribute, np.ndarray):
                pass  # ignoring history to prevent numpy bug in dill
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


def init_obj(name='', objclass=BaseObject, track='default', as_copy=False, **kwargs):
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
            if hasattr(fl, "init_track"):
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

