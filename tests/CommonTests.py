# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 09:52:57 2021

@author: dhulse
"""
import os
import shutil
import numpy as np
from fmdtools.faultsim import propagate
from fmdtools.resultdisp import process as proc
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
    def check_save_load_nominal(self, mdl, mfile, ecfile):
        if os.path.exists(mfile):   os.remove(mfile)
        if os.path.exists(ecfile):  os.remove(ecfile)

        endresult, resgraph, mdlhist=propagate.nominal(mdl, \
                                                        save_args={'mdlhist':{'filename':mfile},\
                                                                   'endclass':{'filename':ecfile}})
        
        self.check_same_file(mdlhist,mfile, check_link=True)
        self.check_same_file(endresult['classification'], ecfile)

        os.remove(mfile), os.remove(ecfile)
    def check_same_file(self, result, resfile, check_link=False):
        result_flattened = proc.flatten_hist(result)
        result_saved = proc.load_result(resfile)
        result_saved_flattened = proc.flatten_hist(result_saved)
        self.assertCountEqual([*result_flattened.keys()], [*result_saved_flattened.keys()])
        for hist_key in result_flattened: # test to see that all values of the arrays in the hist are the same
            if  not isinstance(result_flattened[hist_key], (np.ndarray,list)):
                if isinstance(result_flattened[hist_key], (float, np.number)):
                    self.assertAlmostEqual(result_flattened[hist_key], result_saved_flattened[hist_key])
                else:
                    self.assertEqual(result_flattened[hist_key], result_saved_flattened[hist_key])
            else:
                np.testing.assert_array_equal(result_flattened[hist_key],result_saved_flattened[hist_key])
        if check_link:
            result_flattened['time'][0]=100 #check to see that they aren't linked somehow
            self.assertNotEqual(result_flattened['time'][0], result_saved_flattened['time'][0])
    def check_same_files(self, result, resfolder, filetype, check_link=False):
        result_flattened = proc.flatten_hist(result)
        result_saved = proc.load_results(resfolder, filetype)
        result_saved_flattened = proc.flatten_hist(result_saved)
        self.assertCountEqual([*result_flattened.keys()], [*result_saved_flattened.keys()])
        for hist_key in result_flattened: # test to see that all values of the arrays in the hist are the same
            if  not isinstance(result_flattened[hist_key], (np.ndarray,list)):
                if isinstance(result_flattened[hist_key], (float, np.number)):
                    self.assertAlmostEqual(result_flattened[hist_key], result_saved_flattened[hist_key])
                else:
                    self.assertEqual(result_flattened[hist_key], result_saved_flattened[hist_key])
            else:
                np.testing.assert_array_equal(result_flattened[hist_key],result_saved_flattened[hist_key])
        if check_link:
            result_flattened['time'][0]=100 #check to see that they aren't linked somehow
            self.assertNotEqual(result_flattened['time'][0], result_saved_flattened['time'][0])
    def check_save_load_singlefaults(self,mdl, mfile, ecfile):
        if os.path.exists(mfile):   os.remove(mfile)
        if os.path.exists(ecfile):  os.remove(ecfile)
        endclasses, mdlhists = propagate.single_faults(mdl, showprogress=False, \
                                                        save_args={'mdlhist':{'filename':mfile},\
                                                                   'endclass':{'filename':ecfile}})
        self.check_same_file(mdlhists,mfile)
        self.check_same_file(endclasses, ecfile)
        os.remove(mfile), os.remove(ecfile)
    def check_save_load_singlefaults_indiv(self,mdl, mfolder, ecfolder, ext):
        if os.path.exists(mfolder):   shutil.rmtree(mfolder)
        if os.path.exists(ecfolder):  shutil.rmtree(ecfolder)
        endclasses, mdlhists = propagate.single_faults(mdl, showprogress=False, \
                                                        save_args={'mdlhist':{'filename':mfolder+"."+ext},\
                                                                   'endclass':{'filename':ecfolder+"."+ext},
                                                                   'indiv':True})
        if ext=='pkl': ext="pickle"
        self.check_same_files(mdlhists,mfolder, ext)
        self.check_same_files(endclasses, ecfolder, ext)
        shutil.rmtree(mfolder), shutil.rmtree(ecfolder)
    def check_save_load_nominal_approach(self, mdl, app, mfile, ecfile):
        if os.path.exists(mfile):   os.remove(mfile)
        if os.path.exists(ecfile):  os.remove(ecfile)
        endclasses, mdlhists = propagate.nominal_approach(mdl, app, showprogress=False, \
                                                        save_args={'mdlhist':{'filename':mfile},\
                                                                   'endclass':{'filename':ecfile}})
        self.check_same_file(mdlhists,mfile)
        self.check_same_file(endclasses, ecfile)
        os.remove(mfile), os.remove(ecfile)
    def check_save_load_nominal_approach_indiv(self,mdl, app, mfolder, ecfolder, ext):
        if os.path.exists(mfolder):   shutil.rmtree(mfolder)
        if os.path.exists(ecfolder):  shutil.rmtree(ecfolder)
        endclasses, mdlhists = propagate.nominal_approach(mdl, app, showprogress=False, \
                                                        save_args={'mdlhist':{'filename':mfolder+"."+ext},\
                                                                   'endclass':{'filename':ecfolder+"."+ext},\
                                                                   'indiv':True})
        if ext=='pkl': ext="pickle"
        self.check_same_files(mdlhists,mfolder, ext)
        self.check_same_files(endclasses, ecfolder, ext)
        shutil.rmtree(mfolder), shutil.rmtree(ecfolder)
    def check_save_load_nested_approach(self, mdl, nomapp, mfile, ecfile):
        if os.path.exists(mfile):   os.remove(mfile)
        if os.path.exists(ecfile):  os.remove(ecfile)
        endclasses, mdlhists = propagate.nested_approach(mdl, nomapp, showprogress=False, \
                                                        save_args={'mdlhist':{'filename':mfile},\
                                                                   'endclass':{'filename':ecfile}})
        self.check_same_file(mdlhists,mfile)
        self.check_same_file(endclasses, ecfile)
        os.remove(mfile), os.remove(ecfile)
    def check_save_load_nested_approach_indiv(self, mdl, nomapp, mfolder, ecfolder, ext):
        if os.path.exists(mfolder):   shutil.rmtree(mfolder)
        if os.path.exists(ecfolder):  shutil.rmtree(ecfolder)
        endclasses, mdlhists = propagate.nested_approach(mdl, nomapp, showprogress=False, \
                                                        save_args={'mdlhist':{'filename':mfolder+"."+ext},\
                                                                   'endclass':{'filename':ecfolder+"."+ext},\
                                                                   'indiv':True})
        if ext=='pkl': ext="pickle"
        self.check_same_files(mdlhists,mfolder, ext)
        self.check_same_files(endclasses, ecfolder, ext)
        shutil.rmtree(mfolder), shutil.rmtree(ecfolder)
        
        

