# -*- coding: utf-8 -*-
"""
Description: A module to define Functions.

- :class:`Function`: Class for defining model Functions.
"""
from decimal import Decimal
from fmdtools.define.block.base import Block
from fmdtools.define.container.state import ExampleState
from fmdtools.define.container.parameter import ExampleParameter
from fmdtools.define.container.mode import ExampleMode
from fmdtools.define.architecture.base import inject_faults_internal


class Function(Block):
    """
    Superclass for representing system functions.

    Additional roles
    ----------
    ca : ComponentArchitecture
        Component Architecture fulfilling function.
    aa : ActionArchitecture
        Action Architecture performed by function.
    """

    __slots__ = ["ca", "aa", "args_f"]
    default_track = ["ca", "aa"]+Block.default_track

    def __init__(self, name='', flows={}, ca=dict(), aa=dict(), local=dict(),
                 args_f=dict(), **kwargs):
        """
        Instantiate the function superclass with the relevant parameters.

        Parameters
        ----------
        flows :dict
            Flow objects passed to use instead of instantiating locally.
        ca : dict, optional
            Internal ComponentArchitecture fields/arguments. The default is {}.
        aa : dict, optional
            Internal ASG fields/arguments. The default is {}.
        args_f : dict, optional
            arguments to pass to custom __init__ function.
        """
        super().__init__(name=name, flows=flows.copy(), **kwargs)
        self.args_f = args_f

        for at in ['ca', 'aa']:  # NOTE: similar to init_obj_attr()
            at_arg = eval(at)
            at_init = getattr(self, 'arch_'+at, False)
            if at_init:
                try:
                    if at == "aa":
                        setattr(self, at, at_init(flows=self.flows.copy(), **at_arg))
                    elif at == "ca":
                        setattr(self, at, at_init(**at_arg))
                except TypeError as e:
                    invalid_args = [a for a in at_arg if a not in at_init.__fields__]
                    if invalid_args:
                        argstr = ", Invalid args: "+', '.join(invalid_args)
                    else:
                        argstr = ''
                    raise TypeError("Poor specification for : " + str(at_init) +
                                    " with kwargs: " + str(at_arg) + argstr) from e
                self.update_contained_modes(at)
            elif at_arg:
                raise Exception(at + " argument provided: " + str(at_arg) +
                                "without associating an archiecture to arch_" + at)
        self.update_seed()

    def update_contained_modes(self, at):
        """
        Add contained faultmodes for the container at to the Function model.

        Parameters
        ----------
        at : str ('ca' or 'aa')
            Role to update (for ComponentArchitecture or ActionArchitecture roles)
        """
        if at == 'ca':
            compacts = self.ca.components
        elif at == 'aa':
            compacts = self.aa.actions
        for ca in compacts.values():
            self.m.faultmodes.update({ca.name + "_" + f: vals
                                      for f, vals in ca.m.faultmodes.items()})

    def get_typename(self):
        return "Function"

    def return_faultmodes(self):
        """
        Get the fault modes present in the simulation (for propagate/model).

        Returns
        -------
        ms : list
            List of faults present.
        modeprops : dict
            Dict of corresponding fault mode properties.
        """
        ms = [m for m in self.m.faults.copy() if m != 'nom']
        modeprops = dict.fromkeys(ms)
        for mode in ms:
            modeprops[mode] = self.m.faultmodes.get(mode)
            if mode not in self.m.faultmodes:
                raise Exception("Mode " + mode + " not in m.faultmodes for fxn " +
                                self.__class__.__name__+" and may not be tracked.")
        return ms, modeprops

    def update_seed(self, seed=[]):
        """
        Update seed and propogates update to contained actions/components.

        (keeps seeds in sync)

        Parameters
        ----------
        seed : int, optional
            Random seed. The default is [].
        """
        super().update_seed(seed)

        if hasattr(self, 'ca'):
            self.ca.update_seed(self.r.seed)
        if hasattr(self, 'aa'):
            self.aa.update_seed(self.r.seed)

    def copy(self, *args, **kwargs):
        """
        Create a copy of the function object.

        Adds newflows and arbitrary parameters to be associated with the copy. Used when
        copying the model.

        Returns
        -------
        copy : Function
            Copy of the given function with new flows
        """
        cop = super().copy(*args, **kwargs)
        if hasattr(self, 'ca'):
            cop.ca = self.ca.copy()
            cop.update_contained_modes('ca')
        if hasattr(self, 'aa'):
            cop.aa = self.aa.copy(flows=cop.flows.copy())
            cop.update_contained_modes('aa')
        if hasattr(self, 'h'):
            if hasattr(self, 'ca'):
                for compname, comp in cop.ca.components.items():
                    ex_hist = cop.h.get("ca.components." + compname)
                    if ex_hist:
                        comp.h = ex_hist.copy()
                        for k, v in comp.h.items():
                            cop.h["ca.components." + compname + "." + k] = v
            if hasattr(self, 'aa'):
                # if "a.active_actions" in self.h.keys():
                #     cop.h["a.active_actions"] = self.h['a.active_actions'].copy()
                for actname, act in cop.aa.actions.items():
                    ex_hist = cop.h.get("aa.actions." + actname)
                    if ex_hist:
                        act.h = ex_hist.copy()
                        for k, v in act.h.items():
                            cop.h["aa.actions." + actname + "." + k] = v
        return cop

    def return_mutables(self):
        bm = super().return_mutables()
        cm, am = (), ()
        if hasattr(self, 'ca'):
            cm = self.ca.return_mutables()
        if hasattr(self, 'aa'):
            am = self.aa.return_mutables()
        return *bm, *cm, *am

    def __call__(self, proptype, faults=[], time=0, run_stochastic=False):
        """
        Update the state of the function at a given time and injects faults.

        Parameters
        ----------
        proptype : str
            Type of propagation step to update
            ('behavior', 'static_behavior', or 'dynamic_behavior')
        faults : list, optional
            Faults to inject in the function. The default is [].
        time : float, optional
            Model time. The default is 0.
        run_stochastic : book
            Whether to run the simulation using stochastic or deterministic behavior
        """
        if hasattr(self, 'r'):
            self.r.run_stochastic = run_stochastic
        if faults:
            self.m.add_fault(*faults)  # if there is a fault, it is instantiated
        if hasattr(self, 'mode_state_dict') and any(faults):
            self.update_modestates()
        if hasattr(self, 'condfaults'):
            self.condfaults(time)    # conditional faults and behavior are then run
        if time > self.t.time:
            if hasattr(self, 'r'):
                self.r.update_stochastic_states()
        if hasattr(self, 'ca'):
            inject_faults_internal(self.ca, faults, self.ca.components)
        if hasattr(self, 'aa'):
            inject_faults_internal(self.aa, faults, self.aa.actions)
            try:
                self.aa(time, run_stochastic, proptype, self.t.dt)
            except TypeError as e:
                raise Exception("Poorly specified ActionArchitecture: "
                                + str(self.a.__class__)) from e

        if proptype == 'static' and hasattr(self, 'behavior'):
            self.behavior(time)     # generic behavioral methods are run at all steps
        if proptype == 'static' and hasattr(self, 'static_behavior'):
            self.static_behavior(time)
        elif proptype == 'dynamic' and hasattr(self, 'dynamic_behavior') and time > self.t.time:
            if self.t.run_times >= 1:
                for i in range(self.t.run_times):
                    self.dynamic_behavior(time)
            elif not Decimal(str(time)) % Decimal(str(self.t.dt)):
                self.dynamic_behavior(time)

        # propagate faults from action/component level to function level
        if hasattr(self, 'aa') and self.aa.actions:
            self.m.faults.difference_update(self.aa.faultmodes)
            self.m.faults.update(self.aa.get_faults())
        comps = getattr(self, 'ca', {'components': {}})['components']
        if comps:
            self.m.faults.difference_update(self.ca.faultmodes)
            self.m.faults.update(self.ca.get_faults())
        self.t.time = time
        if run_stochastic == 'track_pdf':
            if hasattr(self, 'r'):
                self.r.probdens = self.r.return_probdens()
        if self.m.exclusive is True and len(self.m.faults) > 1:
            raise Exception("More than one fault present in " + self.name +
                            "\n at t= " + str(time) +
                            "\n faults: " + str(self.m.faults) +
                            "\n Is the mode representation nonexclusive?")
        return

    def reset(self):
        super().reset()
        if hasattr(self, 'ca'):
            self.ca.reset()
        if hasattr(self, 'aa'):
            self.aa.reset()


class ExampleFunction(Function):
    """Example Function block for testing."""

    container_p = ExampleParameter
    container_s = ExampleState
    container_m = ExampleMode

    def dynamic_behavior(self, time):
        """Increment x if nominal, else increment y."""
        if not self.m.any_faults():
            self.s.x += self.p.x
        else:
            self.s.y += self.p.y
        if time < 1.0:
            self.s.put(x=0.0, y=0.0)

    def find_classification(self, scen, hist):
        """Classify via metric xy = s.x + s.y."""
        return {"xy": self.s.x + self.s.y}


class GenericFxn(Function):
    """
    Generic function block.

    For use when the user has not yet defined a class for the given (to be implemented)
    function block. Acts as a placeholder that enables simulation.
    """

    def __init__(self, name='', flows={}, args_f={}, **kwargs):
        super().__init__(name=name, flows=flows, **kwargs)
