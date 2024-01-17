# -*- coding: utf-8 -*-
"""
Module for action architectures.

Classes
-------
:class:`Architecture`: Generic Architecture Class.
"""
from fmdtools.define.object.base import check_pickleability, BaseObject
from fmdtools.define.flow.base import Flow
from fmdtools.define.block.base import Simulable
from fmdtools.define.object.base import init_obj
from fmdtools.analyze.common import get_sub_include
from recordclass import asdict
import time


class Architecture(Simulable):
    """
    Superclass for architectures.

    Architectures are distinguished from Block classes in that they have flexible role
    dictionaries that are populated using add_xxx methods in an overall user-defined
    init_architecture method.

    This method is called for a copy using the as_copy option, which copies passed
    flexible roles.
    """

    __slots__ = ['flows', 'as_copy', 'h', '_init_flexroles']
    flexible_roles = ['flows']
    roletype = 'arch'

    def __init__(self, *args, as_copy=False, h={}, **kwargs):
        self.as_copy = as_copy
        Simulable.__init__(self, *args, h=h, **kwargs)
        self.init_hist(h=h)
        self._init_flexroles = []
        self.init_flexible_roles(**kwargs)
        self.init_architecture(**kwargs)
        self.build(**kwargs)

    def check_role(self, roletype, rolename):
        """Check that 'arch_xa' role is used for the arch."""
        if roletype != self.roletype:
            raise Exception("Invalid roletype for Architecture: " + roletype +
                            ", should be: " + self.roletype)
        if rolename != self.rolename:
            raise Exception("invalid roletype for " + str(self.__class__) +
                            ", should be: " + self.rolename)

    def init_flexible_roles(self, **kwargs):
        """
        Initialize flexible roles.

        If initializing as a copy, uses a passed copy instead.

        Parameters
        ----------
        **kwargs : kwargs
            Existing roles (if any).
        """
        for role in self.flexible_roles:
            if self.as_copy and role in kwargs:
                setattr(self, role, {**kwargs[role]})
            elif self.as_copy:
                raise Exception("No role argument "+role+" to copy.")
            elif role in kwargs:
                setattr(self, role, {**kwargs[role]})
            else:
                setattr(self, role, dict())

    def add_flex_role_obj(self, flex_role, name, objclass=BaseObject, use_copy=False,
                          **kwargs):
        """
        Add a flexible role object to the architecture.

        Used for add_fxn, add_flow, etc methods in subclasses.
        If called during copying (self.as_copy=True), the object is copied instead
        of instantiated.

        Parameters
        ----------
        flex_role : str
            Name of the role dictionary to initialize the object in.
        name : str
            Name of the object
        objclass : class, optional
            Class to instantiate in the dict. The default is BaseObject.
        as_copy : bool
            Whether to instantiate obj as a copy. The default is fault.
        **kwargs : kwargs
            Non-default kwargs to send to the object class.
        """
        roledict = getattr(self, flex_role)
        if name in roledict:
            objclass = roledict[name]

        if use_copy:
            as_copy = False
        else:
            as_copy = self.as_copy

        track = get_sub_include(name, get_sub_include(flex_role, self.track))
        obj = init_obj(name=name, objclass=objclass, track=track,
                       as_copy=as_copy, **kwargs)

        if hasattr(obj, 'h') and obj.h:
            hist = obj.h
        elif isinstance(obj, BaseObject):
            timerange = self.sp.get_histrange()
            hist = obj.create_hist(timerange)
        else:
            hist = False
        if hist:
            self.h[flex_role + '.' + name] = hist
        roledict[name] = obj
        self._init_flexroles.append(name)

    def add_flow(self, name, fclass=Flow, **kwargs):
        """
        Add a flow with given attributes to the model.

        Parameters
        ----------
        name : str
            Unique flow name to give the flow in the model
        fclass : Class, optional
            Class to instantiate (e.g. CommsFlow, MultiFlow). Default is Flow.
            Class must take flowname, p, s as input to __init__()
            May alternatively provide already-instanced object.
        kwargs: kwargs
            Dicts for non-default values to p, s, etc
        """
        if name in self.flows:
            use_copy = True
        else:
            use_copy = False
        self.add_flex_role_obj('flows', name, objclass=fclass, use_copy=use_copy,
                               **kwargs)

    def add_flows(self, *names, fclass=Flow, **kwargs):
        """
        Add a set of flows with the same type and initial parameters.

        Parameters
        ----------
        flownames : list
            Unique flow names to give the flows in the model
        fclass : Class, optional
            Class to instantiate (e.g. CommsFlow, MultiFlow). Default is Flow.
            Class must take flowname, p, s as input to __init__()
            May alternatively provide already-instanced object.
        kwargs: kwargs
            Dicts for non-default values to p, s, etc
        """
        for name in names:
            self.add_flow(name, fclass=fclass, **kwargs)

    def add_sim(self, flex_role, name, simclass, *flownames, **kwargs):
        """
        Add a Simulable to the given flex_role.

        Parameters
        ----------
        flex_role : str
            Name of the flexible role to add to.
        name : str
            Name to give the Simulable.
        simclass: Simulable
            Simulable to instantiate.
        flownames : list
            List of flows to associate with the function.
        **kwargs : kwargs
            Flows, dicts for non-default values to p, s, etc.
        """
        flows = self.get_flows(*flownames, all_if_empty=False)
        fkwargs = {**{'sp': asdict(self.sp)}, **kwargs}
        if hasattr(self, 'r'):
            fkwargs = {**{'r': {"seed": self.r.seed}}, **fkwargs}
        if not self.sp.use_local:
            fkwargs = {**{'t': {'dt': self.sp.dt}}, **fkwargs}

        self.add_flex_role_obj(flex_role, name, objclass=simclass, flows=flows, **fkwargs)

    def init_architecture(self, *args, **kwargs):
        """Use to initialize architecture."""
        return 0

    def build(self, update_seed=True, **kwargs):
        """
        Construct the overall model structure.

        Use in subclasses to build the model after init_architecture is called.

        Parameters
        ----------
        update_seed : bool
            Whether to update the seed
        """
        # remove any dangling objects (flows usually) passed from above but not
        # initialized
        for role in self.flexible_roles:
            roledict = getattr(self, role)
            roledict = {k: v for k, v in roledict.items() if k in self._init_flexroles}

        if update_seed and not self.as_copy:
            self.update_seed()
        if hasattr(self, 'h'):
            self.h = self.h.flatten()

    def get_flows(self, *flownames, all_if_empty=True):
        """Return a list of the model flow objects."""
        if all_if_empty and not flownames:
            flownames = self.flows
        return {flowname: self.flows[flowname] for flowname in flownames}

    def flowtypes(self):
        """Return the set of flow types used in the model."""
        return {obj.__class__.__name__ for f, obj in self.flows.items()}

    def flows_of_type(self, ftype):
        """Return the set of flows for each flow type."""
        return {flow for flow, obj in self.flows.items()
                if obj.__class__.__name__ == ftype}

    def return_mutables(self):
        sim_mutes = Simulable.return_mutables(self)
        muts = [*sim_mutes]
        role_objs = self.get_flex_role_objs()
        for obj in role_objs.values():
            if hasattr(obj, 'return_mutables'):
                muts.extend(obj.return_mutables())
        return tuple(muts)

    def get_flex_role_objs(self):
        role_objs = {}
        for role in self.flexible_roles:
            roledict = getattr(self, role)
            role_objs.update(roledict)
        return role_objs

    def update_seed(self, seed=[]):
        """
        Update model seed and the seed in all contained roles.

        Must have an associated Rand role.

        Parameters
        ----------
        seed : int, optional
            Seed to use. The default is [].
        """
        if hasattr(self, 'r'):
            super().update_seed(seed)
            role_objs = self.get_flex_role_objs()
            for obj in role_objs.values():
                if hasattr(obj, 'update_seed'):
                    obj.update_seed(self.r.seed)

    def get_rand_states(self, auto_update_only=False):
        """Get dictionary of random states throughout the model objs."""
        rand_states = {}
        role_objs = self.get_flex_role_objs()
        for objname, obj in role_objs.items():
            if hasattr(obj, 'get_rand_states'):
                rand_state = obj.get_rand_states(auto_update_only=auto_update_only)
                if rand_state:
                    rand_states[objname] = rand_state
        return rand_states

    def get_faults(self):
        """Get faults from contained roles."""
        return {obj.name+"_"+f for obj in self.get_flex_role_objs().values()
                if hasattr(obj, 'm') for f in obj.m.faults}

    def reset(self):
        """Reset the architecture and its contained objects."""
        super().reset()
        for obj in self.get_flex_role_objs().values():
            if hasattr(obj, 'reset'):
                obj.reset()

    def add_obj_modes(self, obj):
        """Add modes from an object to .faultmodes."""
        modes_to_add = {obj.name+'_'+f: val
                        for f, val in obj.m.faultmodes.items()}
        fmode_intersect = set(modes_to_add).intersection(self.faultmodes)
        if any(fmode_intersect):
            raise Exception("Action " + obj.name +
                            " overwrites existing fault modes: "+str(fmode_intersect) +
                            ". Rename the faults")
        self.faultmodes.update({obj.name+'_'+modename: obj.name
                                for modename in obj.m.faultmodes})

    def inject_faults(self, flexible_role, faults):
        """
        Inject faults in the ComponentArchitecture/ASG object obj.

        Parameters
        ----------
        obj : TYPE
            DESCRIPTION.
        faults : TYPE
            DESCRIPTION.
        """
        for fault in faults:
            if fault in self.faultmodes:
                compdict = getattr(self, flexible_role)
                comp = compdict[self.faultmodes[fault]]
                comp.m.add_fault(fault[len(comp.name)+1:])

    def copy(self, flows={}):
        """
        Copy the architecture at the current state.

        Parameters
        ----------
        flows : dict
            Dict of flows to use in the copy.

        Returns
        -------
        copy : Architecture
            Copy of the curent architecture.
        """
        cargs = dict(p=getattr(self, 'p', {}),
                     sp=getattr(self, 'sp', {}),
                     track=getattr(self, 'track', {}),
                     h=self.h.copy(),
                     as_copy=True)
        # send role dicts in to be copied via as_copy param.
        for flex_role in self.flexible_roles:
            cargs[flex_role] = getattr(self, flex_role)
        # if flows provided from above, use those flows. Otherwise copy own.
        if hasattr(self, 'flows'):
            cargs['flows'] = {f: flows[f] if f in flows else obj.copy()
                              for f, obj in self.flows.items()}

        if hasattr(self, 'r'):
            cargs['r'] = self.r.copy()
        cop = self.__class__(**cargs)
        cop.assign_roles('container', self)
        return cop

    def get_all_possible_track(self):
        return super().get_all_possible_track() + self.flexible_roles


def check_model_pickleability(model, try_pick=False):
    """
    Check to see which attributes of a model object will pickle.

    Provides more detail about functions/flows.
    """
    print('FLOWS ')
    for flowname, flow in model.flows.items():
        print(flowname)
        check_pickleability(flow, try_pick=try_pick)
    print('FUNCTIONS ')
    for fxnname, fxn in model.fxns.items():
        print(fxnname)
        check_pickleability(fxn, try_pick=try_pick)
    time.sleep(0.2)
    print('MODEL')
    unpickleable = check_pickleability(model, try_pick=try_pick)
