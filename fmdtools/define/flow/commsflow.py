# -*- coding: utf-8 -*-
"""Defines :class:`CommsFlow` class representing perception and communications."""
import copy
from fmdtools.define.flow.base import Flow
from fmdtools.define.flow.multiflow import MultiFlow


class CommsFlow(MultiFlow):
    """
    A CommsFlow further extends the MultiFlow class to represent communications.

    It does this by giving each block's view of the flow (in CommsFlow.fxns) an
    "internal" and "out" copy, as well as an "in" (a dict of messages from other views)
    and "received" (a set of messages received).

    To enable the sending/receiving of messages between functions, it further adds:
        - create_comms, for instantiating local copies in functions
        - send, for sending messages from one function to another
        - receive, for receiving messages from other functions
        - inbox, for seeing what messages may be received
        - clear_inbox, for clearing the inbox to enable more messages to be received
    """

    slots = ['__dict__']
    check_dict_creation = False

    def __init__(self, name='', glob=[], p={}, s={}, track=['s']):
        self.fxns = {}
        super().__init__(name=name, glob=glob, p=p, s=s, track=track)

    def __repr__(self):
        rep_str = Flow.__repr__(self)
        if self.name == self.glob.name:
            for fname, func in self.fxns.items():
                rep_str=rep_str+"\n   "+fname+": "+func["internal"].__repr__()
        elif self.name in self.glob.fxns:
            rep_str = (rep_str +
                       "\n       out: " + self.out().__repr__() +
                       "\n       in: " + str(self.inbox()) +
                       "\n       received: " + str(self.received()))
            for l in self.locals:
                rep_str = rep_str+"\n       "+l+": "+getattr(self, l).__repr__()
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

        if fxn_from == "all":
            fxn_from = self.glob.fxns[fxn_to]["in"]
        elif fxn_from == "ports":
            fxn_from = [f for f in self.locals]
        elif isinstance(fxn_from, str):
            fxn_from = {fxn_from: self.glob.fxns[fxn_to]["in"][fxn_from]
                        for i in range(1) if fxn_from in self.glob.fxns[fxn_to]["in"]}
        elif isinstance(fxn_from, list):
            fxn_from = {f: self.glob.fxns[fxn_to]["in"][f]
                        for f in fxn_from if f in self.glob.fxns[fxn_to]["in"]}
        for f_from in list(fxn_from):
            if remove_from_in:
                args = self.glob.fxns[fxn_to]["in"].pop(f_from)
            else:
                args = self.glob.fxns[fxn_to]["in"][f_from]
            port_from = self.get_port(f_from, fxn_to, "out")
            port_to = self.get_port(fxn_to, f_from, "internal")
            port_to.s.assign(port_from.s,  *args, as_copy=True)
            self.glob.fxns[fxn_to]["received"][f_from]=args

    def status(self):
        stat = super().status()
        for f in self.fxns:
            stat[f+"_in"] = self.fxns[f]["in"]
            stat[f+"_in"] = self.fxns[f]["received"]
        return stat

    def return_states(self):
        states= super().return_states()
        for f in self.fxns:
            states.update({f+"_in"+fo: args for fo, args in self.fxns[f]["in"].items()})
        return states

    def reset(self):
        super().reset()
        for fxn in self.fxns:
            self.fxns[fxn]["in"] = {}
            self.fxns[fxn]["received"] = {}

    def copy(self, name='', glob=[], p={}, s={}, track=['s']):
        cop = super().copy(name=name, glob=glob, p=p, s=s, track=track)
        for fxn in self.fxns:
            cop.create_comms(fxn, attrs=self.fxns[fxn]['internal'].status(),
                             out_attrs=self.fxns[fxn]['out'].status(),
                             prev_in=copy.deepcopy(self.fxns[fxn]["in"]),
                             received=copy.deepcopy(self.fxns[fxn]["received"]),
                             ports=getattr(self.fxns[fxn]['internal'], "locals", []))
        return cop

    def get_typename(self):
        return "CommsFlow"

    def find_mutables(self):
        """Add in/received dicts to mutables."""
        mutes = super().find_mutables()
        for f in self.fxns.values():
            mutes.append([f['in'], f['received']])
        return mutes
