# -*- coding: utf-8 -*-
"""
Module for action architectures.

Classes
-------
:class:`ActionArchitecture`: Architecture of multiple Actions.
"""
import networkx as nx
import copy
from fmdtools.define.flow.base import Flow, init_flow
from fmdtools.analyze.common import get_sub_include
from fmdtools.analyze.history import History, init_indicator_hist
from recordclass import asdict


class ActionArchitecture(object):
    """
    Construct the Action Sequence Graph with the given parameters.

    Parameters
    ----------
    initial_action : str/list
        Initial action to set as active. Default is 'auto'
            - 'auto' finds the starting node of the graph and uses it
            - 'ActionName' sets the given action as the first active action
            - providing a list of actions will set them all to active
            (if multi-state rep is used)
    state_rep : 'finite-state'/'multi-state'
        How the states of the system are represented. Default is 'finite-state'
            - 'finite-state' means only one action in the system can be active at once
            (i.e., a finite state machine)
            - 'multi-state' means multiple actions can be performed at once
    max_action_prop : 'until_false'/'manual'/int
        How actions progress. Default is 'until_false'
            - 'until_false' means actions are simulated until all outgoing conditions
            are false
            - providing an integer places a limit on the number of actions that can be
            performed per timestep
    proptype : 'static'/'dynamic'/'manual'
        Which propagation step to execute the Action Sequence Graph in.
        Default is 'dynamic'
            - 'manual' means that the propagation is performed manually
            (defined in a behavior method)
    per_timestep : bool
        Defines whether the action sequence graph is reset to the initial state each
        time-step (True) or stays in the current action (False). Default is False.
    """

    initial_action = "auto"
    state_rep = "finite-state"
    max_action_prop = "until_false"
    proptype = 'dynamic'
    per_timestep = False
    default_track = ('actions', 'active_actions', 'i')

    def __init__(self, flows={}, is_copy=False):
        self.actions = {}
        self.flows = flows
        self.conditions = {}
        self.action_graph = nx.DiGraph()
        self.flow_graph = nx.DiGraph()
        self.conditions = {}
        self.faultmodes = {}
        self.is_copy = is_copy
        self.active_actions = set()

    def build(self):
        if self.initial_action == 'auto':
            initial_action = [act for act, in_degree in self.action_graph.in_degree
                              if in_degree == 0]
            if not initial_action:
                raise Exception("Cannot set initial action--no starting node")
        elif type(self.initial_action) == str:
            initial_action = [self.initial_action]
        self.set_active_actions(initial_action)
        if self.state_rep == 'finite-state' and len(self.active_actions) > 1:
            raise Exception("Cannot have more than one initial action with" +
                            " finite-state representation")

    def add_flow(self, flowname, fclass=Flow, p={}, s={}):
        """
        Add a flow with given attributes to ASG.

        Used to enable a flexible internal flow architecture in the ASG.

        Parameters
        ----------
        flowname : str
            Unique flow name to give the flow in the model
        fclass : Class, optional
            Class to instantiate (e.g. CommsFlow, MultiFlow). Default is Flow.
            Class must take flowname, flowdict, flowtype as input to __init__()
            May alternatively provide already-instanced object.
        p : dict, optional
            Parameter dictionary to instantiate the flow with
        s : dict, optional
            State dictionary to overwrite Flow default state values with
        """
        if flowname not in self.flows:
            self.flows[flowname] = init_flow(flowname, fclass, p=p, s=s)

    def add_act(self, name, actclass, *flownames, duration=0.0, **params):
        """
        Associate an Action with the architecture. Called after add_flow.

        Parameters
        ----------
        name : str
            Internal Name for the Action
        actclass : Action
            Action class to instantiate
        *flownames : flow
            Flows (optional) which connect the actions
        duration:
            Duration of the action. Default is 0.0
        **params : any
            parameters to instantiate the Action with.
        """
        flows = {fl: self.flows[fl] for fl in flownames}
        action = actclass(name=name, flows={**flows}, **params)

        self.actions[name] = action
        self.actions[name].duration = duration

        self.action_graph.add_node(name)
        self.flow_graph.add_node(name, bipartite=0)
        for flow in flows:
            self.flow_graph.add_node(flow, bipartite=1)
            self.flow_graph.add_edge(name, flow)

        modes_to_add = {action.name+'_'+f: val
                        for f, val in action.m.faultmodes.items()}
        fmode_intersect = set(modes_to_add).intersection(self.faultmodes)
        if any(fmode_intersect):
            raise Exception("Action "+name +
                            " overwrites existing fault modes: "+str(fmode_intersect) +
                            ". Rename the faults")
        self.faultmodes.update({action.name+'_'+modename: name
                                for modename in action.m.faultmodes})

    def cond_pass(self): # noqa
        return True

    def add_cond(self, start_action, end_action, name='auto', condition='pass'):
        """
        Associate a Condition with the ActionArchitecture.

        Conditions specify when to precede from one action to the next.

        Parameters
        ----------
        start_action : str
            Action where the condition is checked
        end_action : str
            Action that the condition leads to.
        name : str
            Name for the condition.
            Defaults to numbered conditions if none are provided.
        condition : method
            Method in the class to use as a condition.
            Defaults to self.condition_pass if none are provided.
        """
        if name == 'auto':
            name = str(len(self.conditions)+1)
        if condition == 'pass':
            condition = self.cond_pass
        self.conditions[name] = condition
        self.action_graph.add_edge(start_action,
                                   end_action,
                                   **{'name': name, name: 'name', 'arrow': True})

    def set_active_actions(self, actions):
        """Set given action(s) as active."""
        if type(actions) == str:
            if actions in self.actions:
                actions = [actions]
            else:
                raise Exception("initial_action=" + actions +
                                " not in self.actions: "+str(self.actions))
        if type(actions) == list:
            self.active_actions = set(actions)
            if any(self.active_actions.difference(self.actions)):
                raise Exception("Initial actions not associated with model: " +
                                str(self.active_actions.difference(self.actions)))
        else:
            raise Exception("Invalid option for initial_action.")

    def __call__(self, time, run_stochastic, proptype, dt):
        """
        Propagates behaviors through the ActionArchitecture.

        Parameters
        ----------
        time : float, optional
            Model time. The default is 0.
        run_stochastic : bool/str
            Whether to run the simulation using stochastic or deterministic behavior
        proptype : str
            Type of propagation step to update
            ('behavior', 'static_behavior', or 'dynamic_behavior')
        dt : float
            Timestep to propagate over.
        """
        if self.per_timestep:
            self.set_active_actions(self.initial_action)
            for action in self.active_actions:
                self.actions[action].t.t_loc = 0.0

        if proptype == self.proptype:
            active_actions = self.active_actions
            num_prop = 0
            while active_actions:
                new_active_actions = set(active_actions)
                for action in active_actions:
                    self.actions[action](time, run_stochastic, proptype=proptype, dt=dt)
                    action_cond_edges = self.action_graph.out_edges(action, data=True)
                    for act_in, act_out, atts in action_cond_edges:
                        try:
                            cond = self.conditions[atts['name']]()
                        except TypeError as e:
                            raise TypeError("Poorly specified condition " +
                                            str(atts['name'])+": ") from e
                        if cond and getattr(self.actions[action], 'duration', 0.0)+dt <= self.actions[action].t.t_loc:
                            self.actions[action].t.t_loc = 0.0
                            new_active_actions.add(act_out)
                            new_active_actions.discard(act_in)
                if len(new_active_actions) > 1 and self.state_rep == 'finite-state':
                    raise Exception("Multiple active actions in a finite-state " +
                                    "representation: "+str(new_active_actions))
                num_prop += 1
                if type(self.proptype) == int and num_prop >= self.proptype:
                    break
                if new_active_actions == set(active_actions):
                    break
                else:
                    active_actions = new_active_actions
                if num_prop > 10000:
                    raise Exception("Undesired looping in Function ASG for: "+self.name)
            self.active_actions = active_actions

    def get_faults(self):
        return {act.name+"_"+f for act in self.actions.values() for f in act.m.faults}

    def update_seed(self, seed=[]):
        if seed:
            for act in self.actions.values():
                act.update_seed(seed)

    def copy(self, flows={}, **kwargs):
        newflows = {}
        for flowname, flow in self.flows.items():
            if flowname in flows:
                newflows[flowname] = flows[flowname]
            else:
                newflows[flowname] = self.flows[flowname].copy()

        cop = self.__class__(flows=newflows, is_copy=True, **kwargs)
        for flowname, flow in newflows.items():
            if flow.__hash__() != cop.flows[flowname].__hash__():
                raise Exception("Flow not associated w- lower level of ASG: " +
                                flowname)

        for actname, action in self.actions.items():
            cop_act = cop.actions[actname]
            cop_act.duration = action.duration
            cop_act.s = action.s.copy()
            cop_act.m = action.m.copy()
            cop_act.t = action.t.copy()
            if hasattr(action, 'h'):
                cop_act.h = action.h.copy()

        cop.active_actions = copy.deepcopy(self.active_actions)
        return cop

    def reset(self):
        for name, action in self.actions.items():
            action.reset()
        self.build()

    def create_hist(self, timerange, track):
        """
        Create a history corresponding to ASG attributes.

        Parameters
        ----------
        timerange : iterable, optional
            Time-range to initialize the history over. The default is None.
        track : list/str/dict, optional
            argument specifying attributes for :func:`get_sub_include'.
            The default is None.

        Returns
        -------
        h : History
            History corresponding to the ASG.
        """
        if track == 'default':
            track = self.default_track
        h = History()
        if 'i' in track or track == 'all':
            init_indicator_hist(self, h, timerange, track)
        actions_track = get_sub_include('actions', track)
        if actions_track:
            ha = History()
            for a, act in self.actions.items():
                act_track = get_sub_include(a, actions_track)
                if act_track:
                    ha[a] = act.create_hist(timerange, act_track)
            h['actions'] = ha
        h.init_att('active_actions', self.active_actions,
                   timerange=timerange, track=track)
        return h

    def return_mutables(self):
        am = []
        for a in self.actions.values():
            am.extend(a.return_mutables())
        for f in self.flows.values():
            am.extend(f.return_mutables())
        am.append(copy.copy(self.active_actions))
        return am
