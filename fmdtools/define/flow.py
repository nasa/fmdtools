# -*- coding: utf-8 -*-
"""
Description: A module to define flows used to conect functions in a model. Contains:
    
- :class:`Flow`:        Superclass for flows to be instantiated in a model.
- :class:`MultiFlow`:   Class for flows which enable multiple copies to be instantiated within itself (e.g., for perception)
- :class:`CommsFlow`:   Class for flows which enable communications (e.g., sending/recieving messages) between functions
- :func:`init_flow`:    Flow constructor/factory method.
"""
import warnings
import copy
import sys
import networkx as nx
from recordclass import asdict, astuple

from .parameter import Parameter
from .state import State
from fmdtools.sim.result import History, get_sub_include


class Flow(object):
    __slots__ = ['p', '_args_p', 's', '_args_s', 'h', 'name' ,'is_copy']
    _init_p = Parameter
    _init_s = State
    """
    Superclass for flows. Instanced by Model.add_flow but can also be used as a flow superclass if flow attributes are not easily definable as a dict.
    """
    def __init__(self, name, s={}, p={}):
        """
        Instances the flow with given states.

        Parameters
        ----------
        states : dict
            states and their values to be associated with the flow
        name : str
            name of the flow
        """
        self.name=name
        if not type(s)==dict: s=asdict(s)
        if not type(p)==dict: p=asdict(p)
        self._args_s = s
        self._args_p = p
        self.p=self._init_p(**p)
        self.s=self._init_s(**s)
        # TODO : add to module for safety-checking. Alternatively, run these checks prior to use
        #if type(self)!=Flow and not suppress_warnings:
        #    if type(self).reset == Flow.reset:      warnings.warn("Custom reset() Method Not Implemented--model protection between methods may not work")
        #    if type(self).status == Flow.status:    warnings.warn("Custom status() Method Not Implemented--custom flow attributes may not be saved")
        #    if type(self).get_memory == Flow.get_memory:    warnings.warn("Custom get_memory() method not implement--memory estimates may be innaccurate")
        #    if type(self).copy == Flow.copy:        warnings.warn("Custom copy() method not implemented--Staged Execution may not copy custom model states")
    def __repr__(self):
        if hasattr(self,'name'):    
            return getattr(self, 'name')+' '+self.__class__.__name__+' flow: '+self.s.__repr__()
        else: return "Uninitialized Flow"
    def reset(self):
        """ Resets the flow to the initial state"""
        self.s=self._init_s(**self._args_s)
    def return_mutables(self):
        return astuple(self.s)
    def status(self):
        """
        Returns a dict with the current states of the flow.
        """
        return asdict(self.s)
    def get_memory(self):
        """
        Returns the approximate memory usage of the flow.
        """
        mem = 0
        for state in self.s.__fields__:
            mem+=2*sys.getsizeof(getattr(self.s, state)) # (*2 to account for initstates)
        return mem
    def copy(self):
        """
        Returns a copy of the flow object (used when copying the model)
        """
        cop = self.__class__(self.name, p=asdict(self.p), s=asdict(self.s))
        if hasattr(self, 'h'): cop.h =self.h.copy()
        return cop
    def get_typename(self):
        return "Flow"
    def create_hist(self, timerange, track):
        if hasattr(self, 'h'): return self.h
        else:
            flow_track = get_sub_include('s', track)
            if flow_track:
                h=History()
                h['s'] = self.s.create_hist(timerange, flow_track)
            else: 
                h = History()
            self.h = h
            return h
#Specialized Flow types
class MultiFlow(Flow):
    """
    The MultiFlow class extends the Flow class by adding local flows, which are
    essentially copies of the main flow which live in functions. MultiFlows are
    helpful in cases where each function has a different view of the same external
    flow (e.g., perception, etc). Notably, this class adds the methods:
        - create_local(), which can be used to add a local flow to a function/block
        - get_view(), which can be used to look at other local views of the flow
        - update(), which can be used to update one view of the flow from another
    
    A MultiFlow can have any number of local views (listed by name in MultiFlow.locals)
    as well as a single global view (which may represent the actual value)
    """
    slots= ['__dict__']
    def __init__(self, name, glob=[], s={}, p={}):
        self.locals=[]
        super().__init__(name,  s=s, p=p)
        if not glob: self.glob=self
        else:        self.glob=glob
    def __repr__(self):
        rep_str = Flow.__repr__(self)
        for l in self.locals:
            rep_str=rep_str+"\n   "+self.get_view(l).__repr__()
        return rep_str
    def create_local(self, name, attrs = "all", p={}, s={}):
        """
        Creates a local view of the Flow

        Parameters
        ----------
        name : str
            Name for the view (to retrieve at the Flow level)
        attrs : dict/list/str, optional
            Attributes to include in the local copy. The default is "all". Has options:
                str: to use if only using a single attribute of the local flow
                list: list of attributes to use in the local flow
                dict: dict of attributes to use in the local flow and their initial values
        p : dict
            Parameters to instantiate the local version with (if params used in the flow)

        Returns
        -------
        newflow : MultiFlow
            Local view of the MultiFlow with its own individual values
        """
        #TODO: Need a way of having local flows that are subsets of the global
        # I think this would be done by passing a different class with a subset of attributes
        
        #if attrs == "all":      atts = default_states
        #elif type(attrs)==str:  attrs = [attrs]
        #if type(attrs)==list:   atts = {k:v for k,v in default_states if k in attrs}
        #elif type(attrs)==dict: atts = {k:v for k,v in attrs.items() if k in default_states}
        
        if hasattr(self, name): newflow = getattr(self, name).copy(glob=self, p=p, s=s)
        else:                   newflow = self.__class__(name, glob=self, p=p, s=s)
        setattr(self, name, newflow)
        self.locals.append(name)
        return newflow
    def get_local_name(self, name):
        """Gets the name of the view corresponding to the given name (enables "local" or "global" options)"""
        if name=="local":       return self.name
        elif name=="global":    return "glob"
        else:                   return name
    def get_view(self, name):
        """Gets the view of the MultiFlow corresponding to the given name"""
        if name=="":                                raise Exception("Must provide view")
        elif name=="local":                         view = self
        elif name=="global":                        view=self.glob
        elif name in getattr(self, 'locals',[]):    view = getattr(self, name)
        else:                                       view = getattr(self.glob, name)
        return view
    def update(self, to_update="local", to_get="global", *states):
        """
        Updates a view of the MultiFlow to the values of another view.

        Parameters
        ----------
        to_update : str/list, optional
            Name of the view to update. The default is "local". If "all", updates all locals (or ports for commsflows). 
            If a list is provided, updates the list (in locals)
        to_get : str, optional
            Name of the view to update from. The default is "global".
        *states : str
            States to update (defaults to all states)
        """
        get = self.get_view(to_get)
        if to_update=='all':            
            if hasattr(self, 'fxns'):   updatelist = [*self.fxns]
            else:                       updatelist = self.locals
        elif type(to_update)==str:      updatelist = [to_update]
        elif type(to_update)==list:     updatelist = to_update
        else: raise Exception("Invalid to_update: "+str(to_update))
        for to_up in updatelist:
            up = self.get_view(to_update)
            up.s.assign(get.s, *states, as_copy=True)
    def status(self):
        stat = super().status()
        for l in self.locals:
            stat[l]=getattr(self, l).status()
        return stat
    def return_states(self):
        states = super().status()
        for l in self.locals:
            states.update({l+"."+k:v for k, v in getattr(self, l).status().items()})
        return states
    def reset(self):
        super().reset()
        for local in self.locals:
            getattr(self, local).reset()
    def copy(self, glob=[], params={}):
        states = super().status()
        cop = self.__class__(states, self.name, glob=glob, params=params)
        for loc in self.locals:
            local = getattr(self, loc)
            cop.create_local(local.name, attrs=local.status())
        return cop
    def create_multigraph(self, include_glob=False,
                               send_connections={"closest":"base"},
                               connections_as_tags=True,
                               include_states=False,
                               get_states=False):
        """
        Creates a networkx graph corresponding to the MultiFlow.
    
        Parameters
        ----------
        include_glob : bool, optional
            Whether to include the base flow (if used). The default is False.
        send_connections : dict/list, optional
            Tags/edges to create as send/recieve connections between local views of the 
            flow without explicit containment relationships.
            
            With structure {in_tag : out_tag}. The default is {}.
            Or structure [(in_node : out_node)]
        include_states:
            whether to include states in the graph
        get_states:
            whether to attach state information as node attributes
    
        Returns
        -------
        g : nx.DiGraph
            Networkx graph corresponding to the MultiFlow
        """
        g = nx.DiGraph()
        if include_glob:
            add_g_nested(g,self,self.name, include_states=include_states, get_states=get_states)
        else:
            for loc in self.locals:
                local_flow = getattr(self, loc)
                add_g_nested(g, local_flow, loc, include_states=include_states, get_states=get_states)
        if type(send_connections)==dict:   send_iter = send_connections.items();    connections_as_tags=True
        elif type(send_connections)==list: send_iter = send_connections;            connections_as_tags=False
        
        for in_tag, out_tag in send_iter:
            for in_node in g.nodes:
                if node_is_tagged(connections_as_tags, in_tag, in_node):
                    for out_node in g.nodes:
                        if (node_is_tagged(connections_as_tags, out_tag, out_node)
                            and not((in_node, out_node) in g.edges) and in_node!=out_node):
                            g.add_edge(in_node, out_node, label="sends")
        return g
    def return_stategraph(self, **kwargs):
        g = self.create_multigraph(**kwargs, get_states=True)
        return g
    def create_hist(self, timerange, track):
        super().create_hist(timerange, track)
        for localname in self.locals:
            local_flow = getattr(self, localname)
            local_track = get_sub_include(localname, track)
            self.h[localname] = local_flow.create_hist(timerange, local_track)
        return self.h
def node_is_tagged(connections_as_tags, tag, node):
    return (connections_as_tags and (tag in node or (tag=="base" and not("_" in node)))) or tag==node


def add_g_nested(g, multiflow, base_name, include_states=False, get_states=False):
    """
    Helper function for MultiFlow.create_multigraph. Iterates recursively
    through multigraph locals to construct the containment tree.

    Parameters
    ----------
    g : networkx.graph
        Existing graph
    multiflow : MultiFlow
        Multiflow Structure
    base_name : str
        Name at the current level of recursion
    include_states : bool, optional
        Whether to include state attributes in the plot. The default is False.
    get_states : bool, optional
        Whether to attach states as attributes to the graph. The default is False
    """
    if not get_states:  kwargs={}
    else:               kwargs={"states":multiflow.return_states()}
    g.add_node(base_name, label=multiflow.get_typename(), **kwargs)
    if include_states:
        for state in multiflow.s.__fields__:
            if get_states:  kwargs={"states":getattr(multiflow.s, state)}
            g.add_node(base_name+"_"+state, label="state", **kwargs)
            g.add_edge(base_name, base_name+"_"+state, label="contains")
    for loc in multiflow.locals:
        local_flow = getattr(multiflow, loc)
        local_name = base_name+"_"+loc
        if get_states:  kwargs={"states":local_flow.return_states()}
        
        g.add_node(local_name, label=local_flow.get_typename(), **kwargs)
        g.add_edge(base_name, local_name, label="contains")
        if local_flow.locals:
            add_g_nested(g, local_flow, local_name)
        if include_states:
            for state in local_flow.s.__fields__:
                if get_states:  kwargs={"states":getattr(multiflow.s, state)}
                g.add_node(local_name+"_"+state, label="state", **kwargs)
                g.add_edge(local_name, local_name+"_"+state, label="contains")
    def get_typename(self):
        return "MultiFlow"

class CommsFlow(MultiFlow):
    """
    A CommsFlow further extends the MultiFlow class to represent communications. 
    It does this by giving each block's view of the flow (in CommsFlow.fxns) an "internal" and "out" copy, 
    as well as an "in" (a dict of messages from other views) and "received" (a set of messages received)
    To enable the sending/receiving of messages between functions, it further adds:
        -create_comms, for instantiating local copies in functions
        -send, for sending messages from one function to another
        -receive, for receiving messages from other functions
        - inbox, for seeing what messages may be received
        - clear_inbox, for clearing the inbox to enable more messages to be received
    """
    slots= ['__dict__']
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
        Creates an individual view of the CommsFlow (e.g., for a function), with an
        internal view, out view, in dict, and received set.

        Parameters
        ----------
        name : str
            Name for the view (e.g., the name of the function)
        ports : list
            Ports to send the information to/from (e.g., names of external functions/agents)

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
            self.fxns[name]={"internal":    ins, 
                                "out":      outs, 
                                "in":       kwargs.get("prev_in", {}),
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
        fxn_from=self.get_local_name(fxn_from)
        f_from = self.get_view(fxn_from)
        
        if fxn_to=="all":       fxns_to=[f for f in self.glob.fxns if f!=self.name]
        elif fxn_to=="ports":   fxns_to=[f for f in f_from.locals]
        elif type(fxn_to)==str: fxns_to = [self.get_local_name(fxn_to)]
        else:                   fxns_to = fxn_to
        
        for f_to in fxns_to:
            port_internal = self.get_port(fxn_from, f_to, "internal")
            port_out = self.get_port(fxn_from, f_to, "out")
            port_out.s.assign(port_internal.s, *states, as_copy=True)
            
            if fxn_from not in self.glob.fxns[f_to]["received"]:
                newstates = tuple(set([*self.glob.fxns[f_to]["in"].get(fxn_from,()), *states]))
                self.glob.fxns[f_to]["in"][fxn_from]=newstates
    def inbox(self, fxnname="local"):
        """ Provides a list of messages which have not been received by the function yet"""
        fxnname = self.get_local_name(fxnname)
        return self.glob.fxns[fxnname]["in"]
    def received(self, fxnname="local"):
        fxnname = self.get_local_name(fxnname)
        return self.glob.fxns[fxnname]["received"]
    def clear_inbox(self, fxnname="local"):
        """ Clears the inbox of the function so it can recieve more messages"""
        fxnname = self.get_local_name(fxnname)
        self.glob.fxns[fxnname]["in"].clear()
        self.glob.fxns[fxnname]["received"].clear()
    def out(self, fxnname="local"):
        """ Provies the view of the message that is being sent by the function"""
        fxnname = self.get_local_name(fxnname)
        return self.glob.fxns[fxnname]["out"]
    def get_port(self, fxnname, portname, box="internal"):
        """Gets a port with name portname (if it exists), otherwise the default port is riven. 
        The argument is 'internal' or 'out' for the internal state or outbox, respectively"""
        port = self.glob.fxns[fxnname][box]
        if portname in port.locals:
            port = getattr(port, portname)
        return port
    def receive(self, fxn_to="local", fxn_from="all", remove_from_in=True): #need to add something for resolving errors
        """ Updates the internal view of the flow from external functions 

        Parameters
        ----------
        fxn_to : str
            Name of the view to recieve the view. The default is "local".
        fxn_from : str/list, optional
            Name of the function to send from. The default is "all".
        remove_from_in : bool
            Whether to remove the notification from the "inbox." The default is True
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
    def copy(self, glob=[], p={}):
        states = super().status()
        cop = self.__class__(self.name, glob=glob, p=p, s=states)
        for fxn in self.fxns:
            cop.create_comms(fxn, attrs=self.fxns[fxn]['internal'].status(), out_attrs=self.fxns[fxn]['out'].status(),
                             prev_in=copy.deepcopy(self.fxns[fxn]["in"]), received=copy.deepcopy(self.fxns[fxn]["received"]),
                             ports = getattr(self.fxns[fxn], "locals", []))
        return cop
    def create_commsgraph(self, include_glob=False, ports_only=False, get_states=False):
        """
        Creates a graph representation of the CommsFlow (assuming no additional locals)
    
        Parameters
        ----------
        include_glob : bool, optional
            Whether to include the base (root) node. The default is False.
        ports_only : bool, optional
            Whether to only include the explicit port connections betwen flows. The default is False
        with_internal: bool, optional
            Whether to include the internal aspect of the commsflow in the commsflow.
    
        Returns
        -------
        g : networkx.DiGraph
            Graph of the commsflow connections.
        """
        send_connections=[]
        for f in self.fxns:
            int_flow = getattr(self, f)
            int_ports = int_flow.locals
            out_flow = getattr(self, f+"_out")
            out_ports = out_flow.locals
            send_connections.append((f, f+"_out"))
            for port in int_ports:
                portname = f+"_"+port
                if port in out_ports:   send_connections.append((portname, f+"_out_"+port))
                else:                   send_connections.append((portname, f+"_out"))
            for f2 in self.fxns:
                f2_int = getattr(self,f2)
                if f2 in out_ports:
                    for port in out_ports:
                        portname = f+"_out: "+port
                        if port in f2_int.locals:
                            send_connections.append((portname, f2+": "+port))
                        elif port==f2:
                            send_connections.append((portname,f2))
                else:
                    if f in f2_int.locals:
                        send_connections.append((f+"_out", f2+"_"+f))
                    elif not(ports_only):
                        send_connections.append((f+"_out", f2))
                        
        g = self.create_multigraph(include_glob=include_glob, 
                                   send_connections=send_connections, 
                                   get_states=get_states)
        return g
    def return_stategraph(self, include_glob=False, ports_only=False):
        g= self.create_commsgraph(get_states=True, include_glob=include_glob, ports_only=ports_only)
        return g
    def get_typename(self):
        return "CommsFlow"

def init_flow(flowname, fclass=Flow, p={}, s={}):
    """Factory method for flows. Enables one to instantiate different types of flows with given states/parameters
    or  pass an already-constructured flow class.

    Parameters
    ----------
    flowname : str
        Name to give the flow object
    fclass : Flow/MultiFlow/Comms/CustomFlow
        Flow class to instantiate OR already-instanced object to pass
    p : dict
        Parameter values to override from defaults.
    p : dict
        State values to override from defaults.
    """
    if not callable(fclass):        fl = fclass
    else:                           fl = fclass(flowname, p=p, s=s)
    return fl


