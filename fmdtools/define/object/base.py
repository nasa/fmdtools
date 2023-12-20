# -*- coding: utf-8 -*-
"""
Description: A module defining BaseObjects.

Classes in this module:

- :class:`BaseObject`: Base object class used throughout.
-:class:`ExampleObject`: Example base object for testing.

Functions contained in this module:

- :func:`check_pickleability`:Checks to see which attributes of an object will pickle
  (and thus parallelize)"
"""
import dill
import pickle
import time
import sys
import inspect
import numpy as np
from fmdtools.analyze.common import get_sub_include
from fmdtools.analyze.history import History


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

    __slots__ = ('name', 'containers', 'indicators', 'track')
    roletypes = ['container']
    roledicts = []
    rolevars = []
    immutable_roles = ['p']
    default_track = ['i']

    def __init__(self, name='', roletypes=[], track='default', **kwargs):
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

                {'functions':{'fxn1':'att1'}, 'flows':{'flow1':'att1'}}
        **kwargs : dict, object
            Keywork arguments for the roles.
            May be a dict of non-default arguments (e.g. s={'x': 1.0}) or
            a fully instantiated object (e.g., s=ExampleState()),
        """
        if not name:
            self.name = self.__class__.__name__.lower()
        else:
            self.name = name
        self.init_track(track)
        self.init_indicators()
        self.init_roletypes(*roletypes, **kwargs)

    def init_track(self, track):
        """Add .track attribute."""
        if not track:
            self.track = []
        elif track == 'default':
            self.track = self.default_track
        else:
            self.track = track

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
            container_args = kwargs.get(rolename, {})
            if isinstance(container_args, container_initializer):
                container = container_args
            else:
                container = container_initializer(**container_args)
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
        >>> from fmdtools.define.container.state import ExampleState
        >>> class ExampleObject(BaseObject):
        ...    container_s = ExampleState
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
            if bool(inspect.signature(other_role.copy).parameters):
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
        obj : object
            Object with _spec_ attributes
        spec : str
            Name of the attributes to initialize
        set_attr : bool
            Whether to also add the individual attributes attr to the obj
        sub_obj : str
            Sub-object to form the object from (e.g., 'p' if defined in a parameter).
            Default is '', which gets from obj.
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
                if (bool(inspect.signature(ind).parameters) and ind(time))
                or (not bool(inspect.signature(ind).parameters) and ind())]

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

    def get_track(self, track, all_possible=()):
        """
        Get tracking params for a given object (block, model, etc).

        Parameters
        ----------
        track : track
            str/tuple. Attributes to track.
            'all' tracks all fields
            'default' tracks fields defined in default_track for the dataobject
            'none' tracks none of the fields

        Returns
        -------
        track : tuple
            fields to track
        """
        if track == 'default':
            track = self.default_track
        if track == 'all':
            if not all_possible:
                track = self.get_all_possible_track()
            else:
                track = all_possible
        elif track in ['none', False]:
            track = ()
        elif type(track) == str:
            track = (track,)
        return track

    def get_all_possible_track(self):
        """Get all possible tracking options."""
        rs = [role for role in self.get_all_roles()
              if role not in self.immutable_roles]
        return rs + ['i'] + self.rolevars

    def get_role_memory(self, rolename):
        """Get memory from a particular role."""
        role = getattr(self, rolename)
        if hasattr(role, 'get_memory'):
            mem, _ = role.get_memory()
        else:
            mem = sys.getsizeof(role)
        return mem

    def get_roles(self, *roletypes):
        """Get all roles."""
        if not roletypes:
            roletypes = self.roletypes
        return [role for roletype in roletypes for role in getattr(self, roletype+'s')]

    def get_roledicts(self, *roledicts):
        """Get all roles in roledicts."""
        if not roledicts:
            roledicts = self.roledicts
        return [role for roledict in roledicts for role in getattr(self, roledict)]

    def get_all_roles(self):
        """Get all roles in the object."""
        return self.get_roles() + self.get_roledicts() + self.rolevars

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
        roles_to_check = self.get_all_roles()
        for rolename in roles_to_check:
            mem_profile[rolename] = self.get_role_memory(rolename)
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
        if hasattr(self, 'h'):
            return self.h
        else:
            track = self.get_track(self.track)
            hist = History()
            if track:
                self.init_indicator_hist(hist, timerange, track)
                other_tracks = [t for t in track if t not in ('i', 'flows')]
                for at in other_tracks:
                    at_track = get_sub_include(at, track)
                    attr = getattr(self, at, False)
                    if hasattr(self, at):
                        if hasattr(attr, 'create_hist'):
                            if 'track' in inspect.signature(attr.create_hist).parameters:
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


def check_pickleability(obj, verbose=True, try_pick=False, pause=0.2):
    """Check to see which attributes of an object will pickle (and parallelize)."""
    from pickle import PicklingError
    unpickleable = []
    try:
        itera = vars(obj)
    except:
        itera = {a: getattr(obj, a) for a in obj.__slots__}
    for name, attribute in itera.items():
        print(name)
        time.sleep(pause)
        try:
            if not dill.pickles(attribute):
                unpickleable = unpickleable + [name]
        except ValueError as e:
            raise ValueError("Problem in " + name +
                             " with attribute " + str(attribute)) from e
        if try_pick:
            try:
                a = pickle.dumps(attribute)
                b = pickle.loads(a)
            except:
                raise Exception(obj.name + " will not pickle")
    if try_pick:
        try:
            a = pickle.dumps(obj)
            b = pickle.loads(a)
        except PicklingError as e:
            raise Exception(obj.name + " will not pickle") from e
    if verbose:
        if unpickleable:
            print("The following attributes will not pickle: " + str(unpickleable))
        else:
            print("The object is pickleable")
    return unpickleable


def init_obj(name, objclass=BaseObject, track='default', as_copy=False, **kwargs):
    """
    Initialize an object.

    Enables one to instantiate different types of objects with given
    states/parameters or pass an already-constructured object.

    Parameters
    ----------
    name : str
        Name to give the flow object
    fclass : Flow/MultiFlow/Comms/CustomFlow
        Flow class to instantiate OR already-instanced object to pass
    **kwargs :dict
        Other specialized roles to overrride
    """
    if not inspect.isclass(objclass):
        if not as_copy:
            fl = objclass
            fl.init_track(track)
        else:
            fl = objclass.copy(name=name, track=track, **kwargs)
    else:
        try:
            fl = objclass(name, track=track, **kwargs)
        except TypeError as e:
            raise TypeError("Poorly specified class "+str(objclass) +
                            " (or poor arguments) "+str(kwargs)) from e
    return fl


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
