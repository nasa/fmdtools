# -*- coding: utf-8 -*-
"""
Module providing the `BaseContainer` class which other containers inherit from.

A container is a dataobject (from the recordclass library) that fulfills a specific role
in a block.
"""
from recordclass import dataobject
import copy
from fmdtools.define.base import set_arg_as_type


class BaseContainer(dataobject, mapping=True, iterable=True, copy_default=True):
    """Base container class."""

    rolename = 'x'

    def check_role(self, rolename):
        """
        Check that the container will be given the correct name for its class.

        The correct container-names correspond to the role for the class embody, e.g.:
            State : s
            Rand : r
            Mode : m
            Parameter : p
            SimParam : sp
        """
        if rolename != self.rolename:
            raise Exception("Invalid rolename "+rolename+" for "
                            + self.__class__.__name__ + ": "
                            + "should be " + self.rolename + " instead.")

    def get_track(self, track):
        """
        Get tracking params for a given dataobject (State, Mode, Rand, etc).

        Parameters
        ----------
        obj : dataobject
            State/Mode/Rand. Requires .default_track class variable.
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
        if not track or track == 'default':
            track = self.default_track
        if track == 'all':
            track = self.__fields__
        elif track == 'none':
            track = ()
        elif type(track) == str:
            track = (track,)
        return track

    def get_true_fields(self, *args, force_kwargs=False, **kwargs):
        """
        Resolve the args to pass given certain defaults, *args and **kwargs.

        NOTE: must be used for pickling, since pickle passes arguments as *args and not
        **kwargs.
        """
        true_args = list([copy.copy(i) for i in self.__default_vals__])
        for i, n in enumerate(self.__fields__):
            if force_kwargs:
                true_args[i] = kwargs[n]
            if i < len(args):
                true_args[i] = args[i]
            elif n in kwargs:
                true_args[i] = kwargs[n]
        return true_args

    def get_true_field(self, fieldname, *args, **kwargs):
        """Get the value that will be set to fieldname given *args and **kwargs."""
        if fieldname in kwargs:
            return kwargs[fieldname]
        field_ind = dataobject.__fields__.index(fieldname)
        if args and len(args) > field_ind:
            return args[field_ind]
        else:
            return copy.copy(self.__defaults__[field_ind])

    def set_arg_type(self, *args, **kwargs):
        """
        Set Parameter field input to the predetermined field type.

        e.g., if the input to parameter is int for a float field, this converts it to a
        float in initialization.

        Parameters
        ----------
        obj : dataobject (or class)
            dataobject to get argument type for
        *args : *args
            args to dataobject
        **kwargs : **kwargs
            kwargs to dataobject

        Returns
        -------
        *new_args : tuple
            new args to dataobject (with proper type)
        **new_kwargs : dict
            new kwargs to dataobject (with proper type)
        """
        new_args = []
        new_kwargs = {}
        for i, typed_field in enumerate(self.__annotations__):
            true_type = self.__annotations__.get(typed_field, False)
            try:
                if i < len(args):
                    new_arg = set_arg_as_type(true_type, args[i])
                    new_args.append(new_arg)
                elif typed_field in kwargs:
                    new_arg = set_arg_as_type(true_type, kwargs[typed_field])
                    new_kwargs[typed_field] = new_arg

            except TypeError as e:
                try:
                    raise Exception("For field " + typed_field + " " + str(true_type) +
                                    ": unable to convert from " + str(new_arg) + " " +
                                    str(type(new_arg))) from e
                except UnboundLocalError as e1:
                    raise e
        return tuple(new_args), new_kwargs