# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 09:52:57 2021

@author: dhulse
"""
from fmdtools.faultsim import propagate
class CommonTests():
    def check_model_copy_same(self, mdl, mdl2, inj_times, copy_time, max_time=55, run_stochastic=False):
        """ Tests to see that two models have the same states and that a copied model
        has the same states as the others given the same inputs"""
        faultscens = [{fname: [*f.faultmodes][0]} for fname, f in mdl.fxns.items()] 
        for faultscen in faultscens:
            for inj_time in inj_times:
                for t in range(max_time):
                    if t==inj_time:   scen=faultscen
                    else:       scen={}
                    propagate.propagate(mdl,scen,t,run_stochastic=run_stochastic)       
                    propagate.propagate(mdl2,scen,t,run_stochastic=run_stochastic) 
                    self.check_same_model(mdl, mdl2)
                    if t==copy_time: mdl_copy = mdl.copy()
                    if t>copy_time: 
                        propagate.propagate(mdl_copy, scen, t,run_stochastic=run_stochastic)
                        self.check_same_model(mdl, mdl_copy)
    def check_model_reset(self, mdl, mdl_reset, inj_times, max_time=55, run_stochastic=False):
        """ Tests to see if model attributes reset with the reset() method such that
        reset models simulate the same as newly-created models. """
        faultscens = [{fname: [*f.faultmodes][0]} for fname, f in mdl.fxns.items()]
        mdls = [mdl.copy() for i in range(len(faultscens)*len(inj_times))]
        for faultscen in faultscens:
            for inj_time in inj_times:
                for t in range(max_time):
                    if t==inj_time:     scen=faultscen
                    else:               scen={}
                    propagate.propagate(mdl_reset,scen,t,run_stochastic=run_stochastic)       
                mdl_reset.reset()
                mdl = mdls.pop()
                for t in range(max_time):
                    propagate.propagate(mdl_reset,{},t,run_stochastic=run_stochastic)  
                    propagate.propagate(mdl,{},t,run_stochastic=run_stochastic)  
                    self.check_same_model(mdl, mdl_reset)
    def check_model_copy_different(self,mdl, inj_times, max_time=55, run_stochastic=False):
        """ Tests to see that a copied model has different states from the model
        it was copied from after fault injection/etc"""
        faultscens = [{fname: [*f.faultmodes][0]} for fname, f in mdl.fxns.items()] 
        for faultscen in faultscens:
            for inj_time in inj_times:
                for t in range(max_time):
                    propagate.propagate(mdl,{},t,run_stochastic=run_stochastic)       
                    if t==inj_time: mdl_copy = mdl.copy()
                    if t>inj_time: 
                        propagate.propagate(mdl_copy, faultscen, t)
                        self.check_diff_model(mdl, mdl_copy)
    def check_same_model(self, mdl, mdl2):
        for flname, fl in mdl.flows.items():
            for state in fl._states:
                self.assertEqual(getattr(fl, state), getattr(mdl2.flows[flname], state))
        for fxnname, fxn in mdl.fxns.items():
            for state in fxn._states:
                self.assertEqual(getattr(fxn, state), getattr(mdl2.fxns[fxnname], state))
            self.assertEqual(fxn.faults, mdl2.fxns[fxnname].faults)
    def check_diff_model(self, mdl, mdl2):
        same=1
        for flname, fl in mdl.flows.items():
            for state in fl._states:
                if getattr(fl, state)==getattr(mdl2.flows[flname], state): same = same*1
                else:                                                       same=0
        for fxnname, fxn in mdl.fxns.items():
            for state in fxn._states:
                if getattr(fxn, state)== getattr(mdl2.fxns[fxnname], state): same= same*1
                else:                                                       same=0
            if fxn.faults==mdl2.fxns[fxnname].faults:                   same=same*1
            else:                                                       same=0
        if same==1:
            a=1
        self.assertEqual(same,0)
