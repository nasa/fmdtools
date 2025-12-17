#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Representation of flows with the capability for representing a communications network.

Defines:

- :class:`CommsFlow` class which represents communications networks.
- :class:`CommsFlowGraph` class which represents `CommsFlow` in a ModelGraph structure.

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

from fmdtools.define.base import get_dict_repr
from fmdtools.define.container.state import ExampleState
from fmdtools.define.flow.multiflow import MultiFlow, MultiFlowGraph
from fmdtools.define.base import get_obj_name
from fmdtools.analyze.graph.model import add_edge, ModelGraph

import copy


class CommsFlowGraph(MultiFlowGraph):
    """Create graph representation of the CommsFlow."""

    def __init__(self, flow, role_nodes=['local'], recursive=True, **kwargs):
        ModelGraph.__init__(self, flow, role_nodes=role_nodes, recursive=recursive,
                            **kwargs)


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

    Examples
    --------
    >>> ecf = ExampleCommsFlow()
    >>> t1 = ecf.create_comms("t1")
    >>> t2 = ecf.create_comms("t2")
    >>> ecf
    examplecommsflow ExampleCommsFlow
    - s=ExampleState(x=1.0, y=1.0)
    COMMS:
    - t1=(s=(x=1.0, y=1.0))
    - t2=(s=(x=1.0, y=1.0))
    >>> t1.s.put(x=10.0, y=10.0)
    >>> t1.send("t2", "x")
    >>> t1
    t1 ExampleCommsFlow
    - s=ExampleState(x=10.0, y=10.0)
    - out(): t1_out(s=(x=10.0, y=1.0))
    - inbox(): {}
    - received(): {}
    >>> t2
    t2 ExampleCommsFlow
    - s=ExampleState(x=1.0, y=1.0)
    - out(): t2_out(s=(x=1.0, y=1.0))
    - inbox(): {'t1': ('x',)}
    - received(): {}
    >>> t2.receive()
    >>> t2
    t2 ExampleCommsFlow
    - s=ExampleState(x=10.0, y=1.0)
    - out(): t2_out(s=(x=1.0, y=1.0))
    - inbox(): {}
    - received(): {'t1': ('x',)}
    """

    __slots__ = ['fxns']
    check_dict_creation = False

    def __init__(self, name='', glob=[], track=['s'], **kwargs):
        """Initialize CommsFlow object."""
        self.fxns = {}
        super().__init__(name=name, glob=glob, track=track, **kwargs)

    def base_type(self):
        """Return fmdtools type of the model class."""
        return CommsFlow

    def __repr__(self):
        """Print console representation of CommsFlow."""
        rep_str = self.create_repr(one_line=False)
        if self.name == self.glob.name and self.locals:
            rep_str += "\nCOMMS:"
            comms = {f: func['internal'] for f, func in self.fxns.items()}
            rep_str += get_dict_repr(comms, with_classname=False, one_line=False)
        elif self.name in self.glob.fxns:
            rep_str = (rep_str +
                       "\n- out(): " + self.out().create_repr(with_classname=False, one_line=True) +
                       "\n- inbox(): " + str(self.inbox()) +
                       "\n- received(): " + str(self.received()))
            rep_str += self.create_local_repr()
        return rep_str

    def create_comms(self, name, ports=[], as_copy=False, **kwargs):
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
            ins = self.create_local(name, as_copy=as_copy)
            outs = self.create_local(name+"_out", as_copy=as_copy)
            for port in ports:
                ins.create_local(port, as_copy=as_copy)
                outs.create_local(port, as_copy=as_copy)
            self.fxns[name] = {"internal": ins,
                               "out": outs,
                               "in": kwargs.get("prev_in", {}),
                               "received": kwargs.get("received", {})}
        return self.fxns[name]["internal"]

    def send(self, fxn_to, *states, fxn_from="local"):
        """
        Send a function's view to another function.

        Sends function's (fxn_from) view for the CommsFlow to another function fxn_to
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
        elif isinstance(fxn_to, str):
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
        """Provide list of messages which have not been received by the function yet."""
        fxnname = self.get_local_name(fxnname)
        return self.glob.fxns[fxnname]["in"]

    def received(self, fxnname="local"):
        """Get received property for external function."""
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
            port_to.s.assign(port_from.s, *args, as_copy=True)
            self.glob.fxns[fxn_to]["received"][f_from] = args

    def reset(self):
        """Reset the CommsFlow (and all subflows)."""
        super().reset()
        for fxn in self.fxns:
            self.fxns[fxn]["in"] = {}
            self.fxns[fxn]["received"] = {}

    def copy(self, name='', glob=[], p={}, s={}, track=['s']):
        """Copy the CommsFlow (and all subflows)."""
        cop = super().copy(name=name, glob=glob, p=p, s=s, track=track)
        for fxn in self.fxns:
            cop.create_comms(fxn,
                             prev_in=copy.deepcopy(self.fxns[fxn]["in"]),
                             received=copy.deepcopy(self.fxns[fxn]["received"]),
                             ports=getattr(self.fxns[fxn]['internal'], "locals", []),
                             as_copy=True)
        return cop

    def find_mutables(self, **kwargs):
        """Add in/received dicts to mutables."""
        mutes = super().find_mutables(**kwargs)
        for f in self.fxns.values():
            mutes.append([f['in'], f['received']])
        return mutes

    def add_subgraph_edges(self, g, with_flowedges=True, **kwargs):
        """Add subgraph edges that account for the CommsFlow's comms structure."""
        super().add_subgraph_edges(g, **kwargs)
        if with_flowedges:
            self.add_subgraph_flowedges(g, **kwargs)

    def add_subgraph_flowedges(self, g, **kwargs):
        """Add in/out edges between connected commsflows in the graph."""
        for f in self.fxns:
            int_flow, out_flow = self.get_vars(f, f+"_out")
            int_ports = int_flow.locals
            out_ports = out_flow.locals
            # add internal ports going out
            for portname, portobj in int_flow.get_roles_as_dict('local').items():
                if portname in out_ports:
                    out_port = out_flow.get_vars(portname)
                else:
                    out_port = out_flow
                out_name = out_port.get_full_name()
                pname = portobj.get_full_name()
                add_edge(g, pname, out_name, portname, "connection")
            # add external ports going in
            for f2 in self.fxns:
                f2_out = self.get_vars(f2+"_out")
                f2_out_ports = f2_out.locals
                if int_flow.name in f2_out_ports:
                    out_port = f2_out.get_vars(int_flow.name)
                else:
                    out_port = f2_out
                if f2 in int_ports:
                    in_port = int_flow.get_vars(f2)
                else:
                    in_port = int_flow
                in_name = get_obj_name(in_port, in_port.name, basename=int_flow.root)
                out_name = get_obj_name(out_port, out_port.name, basename=int_flow.root)
                add_edge(g, in_name, out_name, "in", "connection")

    def as_modelgraph(self, gtype=CommsFlowGraph, **kwargs):
        """Create and return the corresponding ModelGraph for the Object."""
        return gtype(self, **kwargs)


class ExampleCommsFlow(CommsFlow):
    """Extension of ExampleFlow to CommsFlow case."""

    container_s = ExampleState
    __slots__ = ()


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
