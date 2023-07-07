# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 09:52:57 2021

@author: dhulse
"""
import os
import shutil
import numpy as np
from fmdtools import sim
from fmdtools.analyze import tabulate as tabulate
from fmdtools.analyze.result import load, load_folder, History, Result
class CommonTests():
    def check_var_setting(self,mdl, statenames, newvalues):
        """ Tests to see that given variable values are set to new values"""
        mdl.set_vars(statenames, newvalues)
        values_to_check = mdl.get_vars(*statenames)
        np.testing.assert_array_equal(values_to_check, newvalues)
    def check_var_setting_dict(self,mdl, new_val_dict):
        mdl.set_vars(**new_val_dict)
        values_to_check = mdl.get_vars(*new_val_dict)
        np.testing.assert_array_equal(values_to_check, [*new_val_dict.values()])
    def check_model_copy_same(self, mdl, mdl2, inj_times, copy_time, max_time=55, run_stochastic=False):
        """ Tests to see that two models have the same states and that a copied model
        has the same states as the others given the same inputs"""
        faultscens = [{fname: [*f.m.faultmodes][0]} for fname, f in mdl.fxns.items()] 
        for faultscen in faultscens:
            for inj_time in inj_times:
                for t in range(max_time):
                    if t == inj_time:   
                        scen = faultscen
                    else:       
                        scen = {}
                    mdl.propagate(t, run_stochastic=run_stochastic, fxnfaults=scen)       
                    mdl2.propagate(t, run_stochastic=run_stochastic, fxnfaults=scen) 
                    self.check_same_model(mdl, mdl2)
                    if t == copy_time: 
                        mdl_copy = mdl.copy()
                    if t > copy_time: 
                        mdl_copy.propagate(t, run_stochastic=run_stochastic, fxnfaults=scen)
                        self.check_same_model(mdl, mdl_copy)
    def check_approach_parallelism(self, mdl, app, track="all"):
        """Test whether the model simulates the same when simulated using parallel or staged options"""
        from multiprocessing import Pool
        endclasses, mdlhists = sim.propagate.approach(mdl, app, showprogress=False, pool=False, track=track)
        mdlhists_by_scen = mdlhists.nest(1)
        
        endclasses_staged, mdlhist_staged = sim.propagate.approach(mdl, app, showprogress=False, pool=False, staged=True, track=track)
        staged_by_scen = mdlhist_staged.nest(1)
        
        endclasses_par, mdlhists_par = sim.propagate.approach(mdl, app, showprogress=False, pool=Pool(4), staged=False, track=track)
        par_by_scen = mdlhists_par.nest(1)
        
        endclasses_staged_par, mdlhists_staged_par = sim.propagate.approach(mdl, app, showprogress=False, pool=Pool(4), staged=True, track=track)
        par_staged_by_scen = mdlhists_staged_par.nest(1)
        
        for scen in mdlhists_by_scen.keys():
            mdlhist = mdlhists_by_scen[scen]
            try: 
                staged = staged_by_scen[scen]
                self.check_same_hist(mdlhist, staged, hist1name="staged")
                
                par = par_by_scen[scen]
                self.check_same_hist(mdlhist, par, hist1name="parallel")
                
                par_staged = par_staged_by_scen[scen]
                self.check_same_hist(mdlhist, par_staged, hist1name="staged-parallel")
            except AssertionError as e:
                raise AssertionError("Problem with scenario: " + scen) from e
        self.check_same_res(endclasses, endclasses_staged, res1name="staged")
        self.check_same_res(endclasses, endclasses_par, res1name="par")
        self.check_same_res(endclasses, endclasses_staged_par, res1name="staged-par")
    def check_same_res(self, res, res1, res1name = "res1"):
        for k in res:
            if res[k] != res1[k]:
                raise AssertionError("Results inconsistent at key k=" + k
                                     + "\n res=" + str(res[k]) + "\n " + res1name + "=" + str(res1[k]))
    def check_same_hist(self, hist, hist1, hist1name="hist1"):
        earliest = np.inf
        err_key = ''
        err_keys = []
        for k in hist:
            err = np.where(hist[k] != hist1[k])[0]
            if any(err):
                err_keys.append(k)
                if err[0] <= earliest:
                    earliest = err[0]
                    err_key = k
                
        if err_key:
            raise AssertionError("Histories inconsistent starting at key k=" + err_key 
                                 + " t=" + str(earliest) + " \n hist = " + str(hist[err_key])
                                 + "\n" + hist1name + "= " + str(hist1[err_key]) +
                                 " \n see also: " + "\n".join(err_keys))
        
    def check_model_reset(self, mdl, mdl_reset, inj_times, max_time=55, run_stochastic=False):
        """ Tests to see if model attributes reset with the reset() method such that
        reset models simulate the same as newly-created models. """
        faultscens = [{fname: [*f.m.faultmodes][0]} for fname, f in mdl.fxns.items()]
        mdls = [mdl.copy() for i in range(len(faultscens)*len(inj_times))]
        for faultscen in faultscens:
            for inj_time in inj_times:
                for t in range(max_time):
                    if t == inj_time:     
                        scen = faultscen
                    else:               
                        scen = {}
                    mdl_reset.propagate(t, run_stochastic=run_stochastic, fxnfaults=scen)       
                mdl_reset.reset()
                mdl = mdls.pop()
                for t in range(max_time):
                    mdl_reset.propagate(t,run_stochastic=run_stochastic)  
                    mdl.propagate(t,run_stochastic=run_stochastic)  
                    try:
                        self.check_same_model(mdl, mdl_reset)
                    except AssertionError as e:
                        raise AssertionError("Problem at time: " + str(t) +
                                             " and faultscen: " + str(faultscen) +
                                             " injected at t=" + str(inj_time)) from e
    def check_model_copy_different(self,mdl, inj_times, max_time=55, run_stochastic=False):
        """ Tests to see that a copied model has different states from the model
        it was copied from after fault injection/etc"""
        faultscens = [{fname: [*f.m.faultmodes][0]} for fname, f in mdl.fxns.items()] 
        for faultscen in faultscens:
            for inj_time in inj_times:
                for t in range(max_time):
                    mdl.propagate(t,run_stochastic=run_stochastic)       
                    if t == inj_time: 
                        mdl_copy = mdl.copy()
                    if t > inj_time: 
                        mdl_copy.propagate(t, fxnfaults=faultscen)
                        self.check_diff_model(mdl, mdl_copy)
    def check_same_model(self, mdl, mdl2):
        """Checks if models mdl and mdl2 have the same attributes"""
        for flname, fl in mdl.flows.items():
            try:
                self.assertEqual(fl.return_mutables(), mdl2.flows[flname].return_mutables())
            except AssertionError as e:
                raise AssertionError("Problem in flow " + flname) from e
        for fxnname, fxn in mdl.fxns.items():
            try:
                self.assertEqual(fxn.return_mutables(), mdl2.fxns[fxnname].return_mutables())
            except AssertionError as e:
                raise AssertionError("Problem in fxn " + fxnname) from e
    def check_diff_model(self, mdl, mdl2):
        """Checks if models mdl and mdl2 have different attributes"""
        same=1
        for flname, fl in mdl.flows.items():
            if fl.return_mutables()==mdl2.flows[flname].return_mutables(): 
                same=same*1
            else:
                same=0
        for fxnname, fxn in mdl.fxns.items():
            if fxn.return_mutables()==mdl2.fxns[fxnname].return_mutables(): 
                same=same*1
            else:
                same=0
        self.assertEqual(same,0)
    def check_save_load_onerun(self, mdl, mfile, ecfile, runtype, faultscen={}):
        """
        Checks if the results from the mdl are the same when saved from a given one-run propagate method as 
        the direct output.

        Parameters
        ----------
        mdl : Model
        mfile : name of file to save mdlhists in
        ecfile : name of file to save endclasses in
        runtype : propagate method to test ('nominal', 'one_fault', 'sequence')
        faultscen : dict/tuple, optional
            - for one_fault, the (functionname, faultname, faulttime)
            - for sequence, the faultseq dict input
        """
        if os.path.exists(mfile):   os.remove(mfile)
        if os.path.exists(ecfile):  os.remove(ecfile)
        check_link=False
        if runtype=='nominal':
            endresult,  mdlhist=sim.propagate.nominal(mdl, 
                                                      save_args={'mdlhist': {'filename': mfile},
                                                                 'endclass': {'filename': ecfile}})
            check_link=True
        elif runtype=='one_fault':
            fxnname, faultmode, faulttime = faultscen
            endresult, mdlhist=sim.propagate.one_fault(mdl, fxnname, faultmode, faulttime,
                                                      save_args={'mdlhist': {'filename': mfile},
                                                                 'endclass': {'filename': ecfile}})
        elif runtype=='sequence':
            endresult, mdlhist=sim.propagate.sequence(mdl, faultscen, {},
                                                      save_args={'mdlhist': {'filename': mfile},
                                                                 'endclass': {'filename': ecfile}})
        else: raise Exception("Invalid Run Type" + runtype)
        
        self.check_same_file(mdlhist, mfile, check_link=check_link)
        self.check_same_file(endresult, ecfile)

        os.remove(mfile), os.remove(ecfile)
    def check_same_fmea(self, app, endclasses, mdl):
        """Tests to ensure results from the fmea are the same over all options"""
        fmea = tabulate.fmea(endclasses,app, group_by='none', mdl=mdl, mode_types={'short'})
        none_exp_cost = sum(fmea['expected cost'])
        for group_by in ['none', 'phase', 'fxnfault', 'mode', 'modetype', 'functions', 'times', 'fxnclassfault','fxnclass']:
            fmea = tabulate.fmea(endclasses,app, group_by=group_by, mdl=mdl, mode_types={'short'})
            exp_cost = sum(fmea['expected cost'])
            self.assertAlmostEqual(none_exp_cost,exp_cost)
    def check_same_file(self, result, resfile, check_link=False):
        """ Checks if the mdlhist/endclass result is the same as the result loaded from resfile """
        result_flattened = result.flatten()
        if isinstance(result, History): Rclass=History
        else:                           Rclass=Result
        
        result_saved = Rclass.load(resfile)
        result_saved_flattened = result_saved.flatten()
        self.assertCountEqual([*result_flattened.keys()], [*result_saved_flattened.keys()])
        self.compare_results(result_flattened, result_saved_flattened)
        if check_link and isinstance(result_saved_flattened['time'], (np.ndarray, list)):
            result_flattened['time'][0]=100 #check to see that they aren't linked somehow
            self.assertNotEqual(result_flattened['time'][0], result_saved_flattened['time'][0])
    def compare_results(self,result_true, result_check):
        """Checks if two flattened endclass/mdlhist results dictionaries are the same"""
        for hist_key in result_true: # test to see that all values of the arrays in the hist are the same
            if  not isinstance(result_true[hist_key], (np.ndarray,list)):
                if isinstance(result_true[hist_key], (float, np.number)) and not np.isnan(result_true[hist_key]) :   
                    self.assertAlmostEqual(result_true[hist_key], result_check[hist_key], 4)
                else:
                    np.testing.assert_array_equal(result_true[hist_key], result_check[hist_key])
            elif isinstance(result_true[hist_key], np.ndarray) and np.issubdtype(result_true[hist_key].dtype, np.number):
                np.testing.assert_allclose(result_true[hist_key],result_check[hist_key])
            else:
                np.testing.assert_array_equal(result_true[hist_key],result_check[hist_key])
    def check_same_files(self, result, resfolder, filetype, check_link=False):
        """Checks to see if the given endclass/mdlhist result is the same as the results
        loaded from resfolder. filetype is the type of file in resfolder, while check_link
        checks if modifying one modifies the other (set to False--usually not applicable)"""
        result_flattened = result.flatten()
        if isinstance(result, History): Rclass=History
        else:                           Rclass=Result
        
        result_saved = Rclass.load_folder(resfolder, filetype, renest_dict=False)
        
        self.assertCountEqual([*result_flattened.keys()], [*result_saved.keys()])
        self.compare_results(result_flattened, result_saved)
        if check_link:
            result_flattened['time'][0]=100 #check to see that they aren't linked somehow
            self.assertNotEqual(result_flattened['time'][0], result_saved['time'][0])
    def start_approach_test(self, mfile, ecfile):
        if os.path.exists(mfile): 
            os.remove(mfile)
        if os.path.exists(ecfile): 
            os.remove(ecfile)
    def end_approach_test(self, mdlhists, mfile, endclasses, ecfile):
        self.check_same_file(mdlhists,mfile)
        self.check_same_file(endclasses, ecfile)
        
        os.remove(mfile), os.remove(ecfile)
        
    def check_save_load_approach(self, mdl, mfile, ecfile, app={}, **kwargs):
        """
        Checks to see if saved results are the same as the direct outputs of a given propagate method

        Parameters
        ----------
        mdl : Model
        mfile : name of file to save mdlhists in
        ecfile : name of file to save endclasses in
        runtype : propagate method to test ('single_faults', 'nominal_approach', 'nested_approach', 'approach')
        app : nominal/sampleapproach to send to the method (if any)
        **kwargs : kwargs to send to the propagate method (if any)
        """
        self.start_approach_test(mfile, ecfile)
        
        endclasses, mdlhists = sim.propagate.approach(mdl, app, showprogress=False, \
                                                  save_args={'mdlhist':{'filename':mfile},\
                                                             'endclass':{'filename':ecfile}}, **kwargs)
        
        self.end_approach_test(mdlhists, mfile, endclasses, ecfile)
    def check_save_load_singlefaults(self, mdl, mfile, ecfile, app={}, **kwargs):
        self.start_approach_test(mfile, ecfile)
        
        endclasses, mdlhists = sim.propagate.single_faults(mdl, showprogress=False, \
                                                    save_args={'mdlhist':{'filename':mfile},\
                                                               'endclass':{'filename':ecfile}}, **kwargs)
        
        self.end_approach_test(mdlhists, mfile, endclasses, ecfile)
    def check_save_load_nomapproach(self, mdl, mfile, ecfile, app={}, **kwargs):
        self.start_approach_test(mfile, ecfile)
        
        endclasses, mdlhists = sim.propagate.nominal_approach(mdl, app, showprogress=False, \
                                                        save_args={'mdlhist':{'filename':mfile},\
                                                                   'endclass':{'filename':ecfile}}, **kwargs)
        
        self.end_approach_test(mdlhists, mfile, endclasses, ecfile)
    def check_save_load_nestapproach(self, mdl, mfile, ecfile, app={}, **kwargs):
        self.start_approach_test(mfile, ecfile)
        endclasses, mdlhists, apps = sim.propagate.nested_approach(mdl, app, showprogress=False, \
                                                        save_args={'mdlhist':{'filename':mfile},\
                                                                   'endclass':{'filename':ecfile}}, **kwargs)
        self.end_approach_test(mdlhists, mfile, endclasses, ecfile)
    def start_save_load_indiv(self, mfolder, ecfolder):
        if os.path.exists(mfolder): 
            shutil.rmtree(mfolder)
        if os.path.exists(ecfolder):
            shutil.rmtree(ecfolder)
    def end_save_load_indiv(self, mdlhists, mfolder, endclasses, ecfolder, ext):
        if ext=='pkl': 
            ext="pickle"
        self.check_same_files(mdlhists,mfolder, ext)
        self.check_same_files(endclasses, ecfolder, ext)
        shutil.rmtree(mfolder), shutil.rmtree(ecfolder)
        

    def check_save_load_singlefaults_indiv(self, mdl, mfolder, ecfolder, ext, **kwargs):
        """
        Checks to see if saved results are the same as the direct outputs of a given propagate method when using individual saving option

        Parameters
        ----------
        mdl : Model
        mfolder : name of folder to save mdlhists in
        ecfolder : name of folder to save endclasses in
        ext : file extension ('pkl','csv','json')
        runtype : propagate method to test ('single_faults', 'nominal_approach', 'nested_approach', 'approach')
        app : nominal/sampleapproach to send to the method (if any)
        **kwargs : kwargs to send to the propagate method (if any)
        """
        self.start_save_load_indiv(mfolder, ecfolder)
        endclasses, mdlhists = sim.propagate.single_faults(mdl, showprogress=False, \
                                                        save_args={'mdlhist':{'filename':mfolder+"."+ext},\
                                                                   'endclass':{'filename':ecfolder+"."+ext},
                                                                   'indiv':True}, **kwargs)
        self.end_save_load_indiv(mdlhists, mfolder, endclasses, ecfolder, ext)
    
    def check_save_load_approach_indiv(self, mdl, mfolder, ecfolder, ext, app={}, **kwargs):
        self.start_save_load_indiv(mfolder, ecfolder)
        endclasses, mdlhists = sim.propagate.approach(mdl, app, showprogress=False, \
                                                  save_args={'mdlhist':{'filename':mfolder+"."+ext},\
                                                             'endclass':{'filename':ecfolder+"."+ext},\
                                                            'indiv':True}, **kwargs)
        self.end_save_load_indiv(mdlhists, mfolder, endclasses, ecfolder, ext)
        
    def check_save_load_nomapproach_indiv(self, mdl, mfolder, ecfolder, ext, app={}, **kwargs):
        self.start_save_load_indiv(mfolder, ecfolder)
        endclasses, mdlhists = sim.propagate.nominal_approach(mdl, app, showprogress=False, \
                                                        save_args={'mdlhist':{'filename':mfolder+"."+ext},\
                                                                   'endclass':{'filename':ecfolder+"."+ext},\
                                                                   'indiv':True}, **kwargs)
        self.end_save_load_indiv(mdlhists, mfolder, endclasses, ecfolder, ext)
        
    def check_save_load_nestapproach_indiv(self, mdl, mfolder, ecfolder, ext, app={}, **kwargs):
        self.start_save_load_indiv(mfolder, ecfolder)
        endclasses, mdlhists, apps = sim.propagate.nested_approach(mdl, app, showprogress=False, \
                                                                   save_args={'mdlhist':{'filename':mfolder+"."+ext},\
                                                                              'endclass':{'filename':ecfolder+"."+ext},\
                                                                            'indiv':True}, **kwargs)
        self.end_save_load_indiv(mdlhists, mfolder, endclasses, ecfolder, ext)
        

        

