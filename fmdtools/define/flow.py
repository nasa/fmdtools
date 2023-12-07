# -*- coding: utf-8 -*-
"""
Description: A module to define flows used to conect functions in a model. Contains:

- :class:`Flow`:        Superclass for flows to be instantiated in a model.
- :class:`MultiFlow`:   Class for flows which enable multiple copies to be instantiated within itself (e.g., for perception)
- :class:`CommsFlow`:   Class for flows which enable communications (e.g., sending/recieving messages) between functions
- :func:`init_flow`:    Flow constructor/factory method.
"""
import copy
import sys
from recordclass import asdict, astuple

from fmdtools.define.role.parameter import Parameter
from fmdtools.define.role.state import State
from fmdtools.define.common import init_obj_attr, get_obj_track, BaseObject
from fmdtools.analyze.common import get_sub_include
from fmdtools.analyze.history import History, init_indicator_hist


class Flow(BaseObject):
    """Superclass for flows."""

    __slots__ = ['p', '_args_p', 's', '_args_s', 'h', 'name', 'is_copy']
    _init_p = Parameter
    _init_s = State
    default_track = ('s', 'i')

    def __init__(self, name, s={}, p={}):
        """
        Instances the flow with given states.

        Parameters
        ----------
        s : dict
            non-default state-values to be associated with the flow
        p : dict
            non-default parameter-values to be associated with the flow
        name : str
            name of the flow
        """
        self.name = name
        init_obj_attr(self, s=s, p=p)
        self.init_roles()
        self.init_indicators()

    def __repr__(self):
        if hasattr(self,'name'):
            return getattr(self, 'name')+' '+self.__class__.__name__+' flow: '+self.s.__repr__()
        else:
            return "Uninitialized Flow"

    def reset(self):
        """Reset the flow to the initial state."""
        self.s = self._init_s(**self._args_s)

    def return_mutables(self):
        return astuple(self.s)

    def status(self):
        """Return a dict with the current states of the flow."""
        return asdict(self.s)

    def get_memory(self):
        """Return the approximate memory usage of the flow."""
        mem = 0
        for state in self.s.__fields__:
            # (*2 to account for initstates)
            mem += 2*sys.getsizeof(getattr(self.s, state))
        return mem

    def copy(self):
        """Return a copy of the flow object (used when copying the model)."""
        cop = self.__class__(self.name, p=asdict(self.p), s=asdict(self.s))
        if hasattr(self, 'h'):
            cop.h = self.h.copy()
        return cop

    def get_typename(self):
        return "Flow"

    def create_hist(self, timerange, track):
        """
        Create the history for the flow.

        Parameters
        ----------
        timerange : np.array
            Time-range to initialize the array over
        track : dict
            States to track

        Returns
        -------
        h : History
            History to initialize.
        """
        if hasattr(self, 'h'): 
            return self.h
        else:
            track = get_obj_track(self, track, all_possible = Flow.default_track)
            if track:
                h=History()
                sh = self.s.create_hist(timerange, get_sub_include('s', track))
                if sh:
                    h['s'] = sh
                init_indicator_hist(self, h, timerange, track)
                self.h = h
                return h
            else:
                return False

# Specialized Flow types

class MultiFlow(Flow):
    """
    MultiFlow class enables represenation of multiple connected copies of the same flow.

    It enables the addition of local flows in an overall flow architecture, which are
    essentially copies of the main flow which live in functions. MultiFlows are
    helpful in cases where each function has a different view of the same external
    flow (e.g., perception, etc). Notably, this class adds the methods:
        - create_local(), which can be used to add a local flow to a function/block
        - get_view(), which can be used to look at other local views of the flow
        - update(), which can be used to update one view of the flow from another

    A MultiFlow can have any number of local views (listed by name in MultiFlow.locals)
    as well as a single global view (which may represent the actual value)
    """

    slots = ['__dict__']

    def __init__(self, name, glob=[], s={}, p={}):
        self.locals = []
        super().__init__(name,  s=s, p=p)
        if not glob:
            self.glob = self
        else:
            self.glob = glob

    def __repr__(self):
        rep_str = Flow.__repr__(self)
        for loc in self.locals:
            rep_str = rep_str+"\n   "+self.get_view(loc).__repr__()
        return rep_str

    def create_local(self, name, attrs="all", p='global', s='global'):
        """
        Create a local view of the Flow.

        Parameters
        ----------
        name : str
            Name for the view (to retrieve at the Flow level)
        attrs : dict/list/str, optional
            Attributes to include in the local copy. The default is "all". Has options:
                str: to use if only using a single attribute of the local flow
                list: list of attributes to use in the local flow
                dict: dict of attributes to use in the local flow and initial values
        p : dict
            Parameters to instantiate the local version with (if params used in flow)
            Default is 'global', which uses the same parameter as the multiflow
        s : dict
            Initial values for the states. Default is 'global', which uses the
            same initial states as the multiflow

        Returns
        -------
        newflow : MultiFlow
            Local view of the MultiFlow with its own individual values
        """

        if hasattr(self, name):
            oldflow = getattr(self, name)
            newflow = oldflow.copy(glob=self)
        else:
            if p == 'global':
                p = self.p
            if s == 'global':
                s = asdict(self.s)
            newflow = self.__class__(name, glob=self, p=p, s=s)
        setattr(self, name, newflow)
        self.locals.append(name)
        return newflow

    def get_local_name(self, name):
        """Get the name of the view corresponding to the given name (enables "local" or "global" options)"""
        if name == "local":
            return self.name
        elif name == "global":
            return "glob"
        else:
            return name

    def get_view(self, name):
        """Get the view of the MultiFlow corresponding to the given name."""
        if name == "":
            raise Exception("Must provide view")
        elif name == "local": 
            view = self
        elif name == "global": 
            view = self.glob
        elif name == "out":
            view = getattr(self.glob, self.name + "_out")
        elif name in getattr(self, 'locals',[]): 
            view = getattr(self, name)
        else:
            view = getattr(self.glob, name)
        return view

    def update(self, to_update="local", to_get="global", *states):
        """
        Update a view of the MultiFlow to the values of another view.

        Parameters
        ----------
        to_update : str/list, optional
            Name of the view to update. The default is "local". If "all", updates all
            locals (or ports for commsflows).
            If a list is provided, updates the list (in locals)
        to_get : str, optional
            Name of the view to update from. The default is "global".
        *states : str
            States to update (defaults to all states)
        """
        get = self.get_view(to_get)
        if to_update=='all':            
            if hasattr(self, 'fxns'): 
                updatelist = [*self.fxns]
            else:
                updatelist = self.locals
        elif type(to_update)==str:
            updatelist = [to_update]
        elif type(to_update)==list:
            updatelist = to_update
        else: 
            raise Exception("Invalid to_update: "+str(to_update))
        for to_up in updatelist:
            up = self.get_view(to_update)
            up.s.assign(get.s, *states, as_copy=True)

    def status(self):
        stat = super().status()
        for l in self.locals:
            stat[l]=getattr(self, l).status()
        return stat

    def return_states(self):
        states = self.status()
        for l in self.locals:
            states.update({l+"."+k:v for k, v in getattr(self, l).status().items()})
        return states

    def reset(self):
        super().reset()
        for local in self.locals:
            getattr(self, local).reset()

    def copy(self, glob=[], p={}, s={}):
        if not s: 
            s = asdict(self.s)
        cop = self.__class__(self.name, glob=glob, p=p, s=s)
        for loc in self.locals:
            local = getattr(self, loc)
            cop.create_local(local.name, s=asdict(local.s), p=local.p)
        return cop

    def create_hist(self, timerange, track):
        super().create_hist(timerange, track)
        for localname in self.locals:
            local_flow = getattr(self, localname)
            local_track = get_sub_include(localname, track)
            self.h[localname] = local_flow.create_hist(timerange, local_track)
        return self.h

    def get_typename(self):
        return "MultiFlow"

    def return_mutables(self):
        local_mutes = [getattr(self, l).return_mutables() for l in self.locals]
        return (super().return_mutables(), *local_mutes)


class CommsFlow(MultiFlow):
    """
    A CommsFlow further extends the MultiFlow class to represent communications.

    It does this by giving each block's view of the flow (in CommsFlow.fxns) an
    "internal" and "out" copy, as well as an "in" (a dict of messages from other views)
    and "received" (a set of messages received).

    To enable the sending/receiving of messages between functions, it further adds:
        -create_comms, for instantiating local copies in functions
        -send, for sending messages from one function to another
        -receive, for receiving messages from other functions
        - inbox, for seeing what messages may be received
        - clear_inbox, for clearing the inbox to enable more messages to be received
    """
    slots = ['__dict__']

    def __init__(self, name, glob=[], p={}, s={}):
        self.fxns = {}
        super().__init__(name, glob=glob, p=p, s=s)

    def __repr__(self):
        rep_str = Flow.__repr__(self)
        if self.name==self.glob.name:   
            for fname, func in self.fxns.items():
                rep_str=rep_str+"\n   "+fname+": "+func["internal"].__repr__() #+"\n       out: "+func["out"].__repr__()+"\n       in: "+str(func["in"])
        elif self.name in self.glob.fxns:
            rep_str = rep_str+"\n       out: "+self.out().__repr__()+"\n       in: "+str(self.inbox())+"\n       received: "+str(self.received())
            for l in self.locals:
                rep_str=rep_str+"\n       "+l+": "+getattr(self, l).__repr__()
        return rep_str

    def create_comms(self, name, ports=[], **kwargs):
        """
        Create an individual view of the CommsFlow (e.g., for a function).

        This will have an internal view, out view, in dict, and received set.

        Parameters
        ----------
        name : str
            Name for the view (e.g., the name of the function)
        ports : list
            Ports to send the information to/from (e.g., names of external fxns/agents)

        Returns
        -------
        CommsFlow
            A local view of the CommsFlow for the function
        """
        if name not in self.fxns:
            ins = self.create_local(name)
            outs = self.create_local(name+"_out")
            for port in ports:
                ins.create_local(port)
                outs.create_local(port)
            self.fxns[name] = {"internal": ins,
                               "out": outs,
                               "in": kwargs.get("prev_in", {}),
                               "received": kwargs.get("received", {})}
        return self.fxns[name]["internal"]

    def send(self, fxn_to, fxn_from="local", *states):
        """
        Sends a function's (fxn_from) view for the CommsFlow to another function fxn_to
        by updating the function's out property and fxn_to's inbox list. Note that the
        other function must call recieve on the other end for the message to be fully
        received (update its internal view).

        Parameters
        ----------
        fxn_to : str/list
            Name/list of names of the functions to send to. The default is "all"
        fxn_from : str, optional
            Name of the function to send from. The default is "local".
        *states : strs
            Values to send from.
        """
        fxn_from = self.get_local_name(fxn_from)
        f_from = self.get_view(fxn_from)

        if fxn_to == "all":
            fxns_to = [f for f in self.glob.fxns if f != self.name]
        elif fxn_to == "ports":
            fxns_to = [f for f in f_from.locals]
        elif type(fxn_to) == str:
            fxns_to = [self.get_local_name(fxn_to)]
        else:
            fxns_to = fxn_to

        for f_to in fxns_to:
            port_internal = self.get_port(fxn_from, f_to, "internal")
            port_out = self.get_port(fxn_from, f_to, "out")
            port_out.s.assign(port_internal.s, *states, as_copy=True)

            if fxn_from not in self.glob.fxns[f_to]["received"]:
                newstates = [*self.glob.fxns[f_to]["in"].get(fxn_from, ()), *states]
                self.glob.fxns[f_to]["in"][fxn_from] = tuple(set(newstates))

    def inbox(self, fxnname="local"):
        """ Provide list of messages which have not been received by the function yet"""
        fxnname = self.get_local_name(fxnname)
        return self.glob.fxns[fxnname]["in"]

    def received(self, fxnname="local"):
        fxnname = self.get_local_name(fxnname)
        return self.glob.fxns[fxnname]["received"]

    def clear_inbox(self, fxnname="local"):
        """Clear the inbox of the function so it can recieve more messages."""
        fxnname = self.get_local_name(fxnname)
        self.glob.fxns[fxnname]["in"].clear()
        self.glob.fxns[fxnname]["received"].clear()

    def out(self, fxnname="local"):
        """Provide the view of the message that is being sent by the function."""
        fxnname = self.get_local_name(fxnname)
        return self.glob.fxns[fxnname]["out"]

    def get_port(self, fxnname, portname, box="internal"):
        """Get a port with name portname.

        If there is no port for the name, the default port is given.
        The argument is 'internal' or 'out' for the internal state or outbox,
        respectively.
        """
        port = self.glob.fxns[fxnname][box]
        if portname in port.locals:
            port = getattr(port, portname)
        return port

    def receive(self, fxn_to="local", fxn_from="all", remove_from_in=True):
        """
        Update the internal view of the flow from external functions.

        Parameters
        ----------
        fxn_to : str
            Name of the view to recieve the view. The default is "local".
        fxn_from : str/list, optional
            Name of the function to send from. The default is "all".
        remove_from_in : bool
            Whether to remove the notification from the "inbox." The default is True.
        """
        fxn_to = self.get_local_name(fxn_to)

        if fxn_from=="all":         fxn_from = self.glob.fxns[fxn_to]["in"]
        elif fxn_from=="ports":     fxn_from=[f for f in fxn_to.locals]
        elif type(fxn_from)==str:   fxn_from = {fxn_from:self.glob.fxns[fxn_to]["in"][fxn_from] for i in range(1) if fxn_from in self.glob.fxns[fxn_to]["in"]}
        elif type(fxn_from)==list:  fxn_from = {f:self.glob.fxns[fxn_to]["in"][f] for f in fxn_from if f in self.glob.fxns[fxn_to]["in"]}
        for f_from in list(fxn_from):
            if remove_from_in:  args = self.glob.fxns[fxn_to]["in"].pop(f_from)
            else:               args = self.glob.fxns[fxn_to]["in"][f_from]
            port_from = self.get_port(f_from, fxn_to, "out")
            port_to = self.get_port(fxn_to, f_from, "internal")
            port_to.s.assign(port_from.s,  *args, as_copy=True)
            self.glob.fxns[fxn_to]["received"][f_from]=args

    def status(self):
        stat = super().status()
        for f in self.fxns:
            stat[f+"_in"]=self.fxns[f]["in"]
            stat[f+"_in"]=self.fxns[f]["received"]
        return stat

    def return_states(self):
        states= super().return_states()
        for f in self.fxns:
            states.update({f+"_in"+fo:args for fo, args in self.fxns[f]["in"].items()})
        return states

    def reset(self):
        super().reset()
        for fxn in self.fxns:
            self.fxns[fxn]["in"] = {}
            self.fxns[fxn]["received"] = {}

    def copy(self, glob=[], p={}, s={}):
        cop = super().copy(glob=glob, p=p, s=s)
        for fxn in self.fxns:
            cop.create_comms(fxn, attrs=self.fxns[fxn]['internal'].status(), out_attrs=self.fxns[fxn]['out'].status(),
                             prev_in=copy.deepcopy(self.fxns[fxn]["in"]), received=copy.deepcopy(self.fxns[fxn]["received"]),
                             ports = getattr(self.fxns[fxn], "locals", []))
        return cop

    def get_typename(self):
        return "CommsFlow"

    def return_mutables(self):
        mutes = super().return_mutables()
        comms_mutes = []
        for f in self.fxns.values():
            comms_mutes.append([f['in'], f['received']])
        return (*mutes, *comms_mutes)


def init_flow(flowname, fclass=Flow, p={}, s={}, **kwargs):
    """
    Initialize a flow (factory method).

    Enables one to instantiate different types of flows with given
    states/parameters or  pass an already-constructured flow class.

    Parameters
    ----------
    flowname : str
        Name to give the flow object
    fclass : Flow/MultiFlow/Comms/CustomFlow
        Flow class to instantiate OR already-instanced object to pass
    p : dict
        Parameter values to override from defaults.
    s : dict
        State values to override from defaults.
    **kwargs :dict
        Other specialized roles to overrride
    """
    if not callable(fclass):
        fl = fclass
    else:
        fl = fclass(flowname, p=p, s=s, **kwargs)
    return fl
