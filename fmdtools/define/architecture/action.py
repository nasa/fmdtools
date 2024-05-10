# -*- coding: utf-8 -*-
"""
Module for action architectures.

Classes
-------
:class:`ActionArchitecture`: Architecture of multiple Actions.
"""
import networkx as nx
from fmdtools.define.architecture.base import Architecture


class ActionArchitecture(Architecture):
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
            - 'finite-state' means only one action in the system can be active at once (i.e., a finite state machine)
            - 'multi-state' means multiple actions can be performed at once
    max_action_prop : 'until_false'/'manual'/int
        How actions progress. Default is 'until_false'
            - 'until_false' means actions are simulated until all outgoing conditions are false
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
    __slots__ = ['acts', 'conds', 'action_graph', 'flow_graph', 'faultmodes', 'active_actions']
    initial_action = "auto"
    state_rep = "finite-state"
    max_action_prop = "until_false"
    proptype = 'dynamic'
    per_timestep = False
    default_track = ('acts', 'flows', 'active_actions', 'i')
    flexible_roles = ['flows', 'acts', 'conds']
    rolename = 'aa'

    def __init__(self, **kwargs):
        self.action_graph = nx.DiGraph()
        self.flow_graph = nx.DiGraph()
        self.faultmodes = {}
        self.active_actions = set()
        Architecture.__init__(self, **kwargs)

    def copy(self, **kwargs):
        cop = super().copy(**kwargs)
        cop.active_actions = {*self.active_actions}
        return cop

    def reset(self):
        super().reset()
        self.set_initial_active_action()

    def set_initial_active_action(self):
        if self.initial_action == 'auto':
            initial_action = [act for act, in_degree in self.action_graph.in_degree
                              if in_degree == 0]
            if not initial_action:
                raise Exception("Cannot set initial action--no starting node")
        elif type(self.initial_action) == str:
            initial_action = [self.initial_action]
        self.set_active_actions(initial_action)

    def build(self, **kwargs):
        super().build(**kwargs)
        self.set_initial_active_action()
        if self.state_rep == 'finite-state' and len(self.active_actions) > 1:
            raise Exception("Cannot have more than one initial action with" +
                            " finite-state representation")

    def add_act(self, name, actclass, *flownames, **fkwargs):
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
        **kwargs : any
            kwargs to instantiate the Action with.
        """
        self.add_sim('acts', name, actclass, *flownames, **fkwargs)

        # TODO: maybe functions should work like this also?
        self.action_graph.add_node(name)
        self.flow_graph.add_node(name, bipartite=0)
        flows = {fl: self.flows[fl] for fl in flownames}
        for flow in flows:
            self.flow_graph.add_node(flow, bipartite=1)
            self.flow_graph.add_edge(name, flow)

        if hasattr(self.acts[name], 'm'):
            self.add_obj_modes(self.acts[name])

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
            name = str(len(self.conds)+1)
        if condition == 'pass':
            condition = self.cond_pass
        self.conds[name] = condition
        self.action_graph.add_edge(start_action,
                                   end_action,
                                   **{'name': name, name: 'name', 'arrow': True})

    def set_active_actions(self, actions):
        """Set given action(s) as active."""
        if type(actions) == str:
            if actions in self.acts:
                actions = [actions]
            else:
                raise Exception("initial_action=" + actions +
                                " not in self.acts: "+str(self.acts))
        if type(actions) == list:
            self.active_actions = set(actions)
            if any(self.active_actions.difference(self.acts)):
                raise Exception("Initial actions not associated with model: " +
                                str(self.active_actions.difference(self.acts)))
        else:
            raise Exception("Invalid option for initial_action.")

    def inject_faults(self, faults):
        Architecture.inject_faults(self, 'acts', faults)

    def __call__(self, proptype, time, run_stochastic, dt):
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
                self.acts[action].t.t_loc = 0.0

        if proptype == self.proptype:
            active_actions = self.active_actions
            num_prop = 0
            while active_actions:
                new_active_actions = set(active_actions)
                for action in active_actions:
                    self.acts[action](time, run_stochastic, proptype=proptype, dt=dt)
                    action_cond_edges = self.action_graph.out_edges(action, data=True)
                    for act_in, act_out, atts in action_cond_edges:
                        try:
                            cond = self.conds[atts['name']]()
                        except TypeError as e:
                            raise TypeError("Poorly specified condition " +
                                            str(atts['name'])+": ") from e
                        if cond and getattr(self.acts[action], 'duration', 0.0)+dt <= self.acts[action].t.t_loc:
                            self.acts[action].t.t_loc = 0.0
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


