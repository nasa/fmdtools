# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 14:12:11 2023

@author: dhulse
"""
from recordclass import dataobject, asdict, astuple


class BaseRole(dataobject, mapping=True, iterable=True, copy_default=True):
    """Base role class."""
    rolename = 'x'

    def check_role(self, rolename):
        """
        Check that the role will be given the correct name for its class.

        The correct role-names correspond to the roles that the classes embody, e.g.:
            State : s
            Rand : r
            Mode : m
            Parameter : p
            SimParam : sp
            ...
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