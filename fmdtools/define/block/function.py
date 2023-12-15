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


class Function(Block):
    """
    Superclass for representing system functions.

    Additional role types
    ----------
    arch : Architecture
        component, action, function architectures at ca, aa, fa, etc.
    """

    __slots__ = ["ca", "aa", "fa", "args_f", "archs"]
    default_track = ["ca", "aa", "fa"]+Block.default_track
    roletypes = ['container', 'flow', 'arch']

    def __init__(self, args_f=dict(), **kwargs):
        """
        Instantiate the function superclass with the relevant parameters.

        Parameters
        ----------
        args_f : dict, optional
            arguments to pass to custom __init__ function.
        """
        super().__init__(**kwargs)
        self.args_f = args_f
        self.update_contained_modes()

    def update_contained_modes(self):
        """
        Add contained faultmodes for the container at to the Function model.

        Parameters
        ----------
        at : str ('ca' or 'aa')
            Role to update (for ComponentArchitecture or ActionArchitecture roles)
        """
        for at in self.get_roles('arch'):
            arch = getattr(self, at)
            self.m.faultmodes.update(arch.faultmodes)

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
        for at in self.get_roles('arch'):
            arch = getattr(self, at)
            arch.update_seed(self.r.seed)

    def prop_arch_behaviors(self, proptype, faults, time, run_stochastic):
        """Propagate behaviors into contained architectures."""
        for at, obj in self.get_roles('arch'):
            try:
                obj.inject_faults(faults)
                # TODO: this should be more general
                if at == 'aa':
                    obj(proptype, time, run_stochastic, self.t.dt)
                elif at == 'fa':
                    obj.propagate(proptype, time=time, run_stochastic=run_stochastic)
            except TypeError as e:
                raise Exception("Poorly specified Architecture: "
                                + str(self.at.__class__)) from e

    def prop_arch_faults_up(self):
        """Get faults from contained components and add to .m."""
        for at, obj in self.get_roles('arch'):
            self.m.faults.difference_update(obj.faultmodes)
            self.m.faults.update(obj.get_faults())

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
        self.prop_arch_behaviors(proptype, faults, time, run_stochastic)

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

        self.prop_arch_faults_up()

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
