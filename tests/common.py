# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 09:52:57 2021

@author: dhulse
"""
import os
import shutil
import numpy as np
from fmdtools.sim import propagate as prop
from fmdtools.analyze import tabulate as tabulate
from fmdtools.analyze.result import History, Result


class CommonTests():
    """Some basic tests which can be run accross different models."""

    def check_var_setting(self, mdl, statenames, newvalues):
        """Test to see that given variable values are set to new values."""
        mdl.set_vars(statenames, newvalues)
        values_to_check = mdl.get_vars(*statenames)
        np.testing.assert_array_equal(values_to_check, newvalues)

    def check_var_setting_dict(self, mdl, new_val_dict):
        mdl.set_vars(**new_val_dict)
        values_to_check = mdl.get_vars(*new_val_dict)
        np.testing.assert_array_equal(values_to_check, [*new_val_dict.values()])

    def check_model_copy_same(self, mdl, mdl2, inj_times, copy_time, max_time=55,
                              run_stochastic=False):
        """
        Model copying tests.

        Tests that:
            - Two copied models have the same states during fault injection
            - If given the same inputs, a copied model will run the same as the original
        """
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
                        mdl_copy.propagate(t, run_stochastic=run_stochastic,
                                           fxnfaults=scen)
                        self.check_same_model(mdl, mdl_copy)

    def check_fs_parallel(self, mdl, fs, track="all"):
        """
        Chack results are the same for propagate.fault_sample staged/parallel options.

        Checks:
            - History and Result consistent in staged option
            - History and Result consistent in parallel option
            - History and Result consistent with both staged and parallel = True
        """
        from multiprocessing import Pool
        res, hist = prop.fault_sample(mdl, fs, showprogress=False, pool=False,
                                      track=track)
        hist_by_scen = hist.nest(1)

        res_stage, hist_stage = prop.fault_sample(mdl, fs, showprogress=False,
                                                  pool=False, staged=True, track=track)
        stage_by_scen = hist_stage.nest(1)

        res_par, hist_par = prop.fault_sample(mdl, fs, showprogress=False, pool=Pool(4),
                                              staged=False, track=track)
        par_by_scen = hist_par.nest(1)

        res_stage_par, hist_stage_par = prop.fault_sample(mdl, fs, showprogress=False,
                                                          pool=Pool(4), staged=True,
                                                          track=track)
        stage_par_by_scen = hist_stage_par.nest(1)

        for scen in hist_by_scen.keys():
            mdlhist = hist_by_scen[scen]
            try:
                stage = stage_by_scen[scen]
                self.check_same_hist(mdlhist, stage, hist1name="staged")

                par = par_by_scen[scen]
                self.check_same_hist(mdlhist, par, hist1name="parallel")

                stage_par = stage_par_by_scen[scen]
                self.check_same_hist(mdlhist, stage_par, hist1name="staged-parallel")
            except AssertionError as e:
                raise AssertionError("Problem with scenario: " + scen) from e
        self.check_same_res(res, res_stage, res1name="staged")
        self.check_same_res(res, res_par, res1name="par")
        self.check_same_res(res, res_stage_par, res1name="staged-par")

    def check_same_res(self, res, res1, res1name="res1"):
        """Check that two Results have the same values."""
        for k in res:
            if res[k] != res1[k]:
                raise AssertionError("Results inconsistent at key k=" + k
                                     + "\n res=" + str(res[k])
                                     + "\n " + res1name + "=" + str(res1[k]))

    def check_same_hist(self, hist, hist1, hist1name="hist1"):
        """Check that two Historys have the same values."""
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
                                 + " t=" + str(earliest)
                                 + " \n hist = " + str(hist[err_key])
                                 + "\n" + hist1name + "= " + str(hist1[err_key])
                                 + " \n see also: " + "\n".join(err_keys))

    def check_model_reset(self, mdl, mdl_reset, res_times, max_time=55,
                          run_stochastic=False):
        """
        Check if model attributes reset with the reset() method.

        Reset models should simulate the same as newly-created models.
        """
        faultscens = [{fname: [*f.m.faultmodes][0]} for fname, f in mdl.fxns.items()]
        mdls = [mdl.copy() for i in range(len(faultscens)*len(res_times))]
        for faultscen in faultscens:
            for inj_time in res_times:
                for t in range(max_time):
                    if t == inj_time:
                        scen = faultscen
                    else:
                        scen = {}
                    mdl_reset.propagate(t, run_stochastic=run_stochastic,
                                        fxnfaults=scen)
                mdl_reset.reset()
                mdl = mdls.pop()
                for t in range(max_time):
                    mdl_reset.propagate(t, run_stochastic=run_stochastic)
                    mdl.propagate(t, run_stochastic=run_stochastic)
                    try:
                        self.check_same_model(mdl, mdl_reset)
                    except AssertionError as e:
                        raise AssertionError("Problem at time: " + str(t) +
                                             " and faultscen: " + str(faultscen) +
                                             " injected at t=" + str(inj_time)) from e

    def check_model_copy_different(self, mdl, cop_times, max_time=55,
                                   run_stochastic=False):
        """
        Check that a copied model has different states from the original.

        Tests in a fault injection time copying scenario specifed by inj_times.
        """
        faultscens = [{fname: [*f.m.faultmodes][0]} for fname, f in mdl.fxns.items()]
        for faultscen in faultscens:
            for inj_time in cop_times:
                for t in range(max_time):
                    mdl.propagate(t, run_stochastic=run_stochastic)
                    if t == inj_time:
                        mdl_copy = mdl.copy()
                    if t > inj_time:
                        mdl_copy.propagate(t, fxnfaults=faultscen)
                        self.check_diff_model(mdl, mdl_copy)

    def check_same_model(self, mdl, mdl2):
        """Check if models mdl and mdl2 have the same attributes."""
        for flname, fl in mdl.flows.items():
            try:
                self.assertEqual(fl.return_mutables(),
                                 mdl2.flows[flname].return_mutables())
            except AssertionError as e:
                raise AssertionError("Problem in flow " + flname) from e
        for fxnname, fxn in mdl.fxns.items():
            try:
                self.assertEqual(fxn.return_mutables(),
                                 mdl2.fxns[fxnname].return_mutables())
            except AssertionError as e:
                raise AssertionError("Problem in fxn " + fxnname) from e

    def check_diff_model(self, mdl, mdl2):
        """Check if models mdl and mdl2 have different attributes."""
        same = 1
        for flname, fl in mdl.flows.items():
            if fl.return_mutables() == mdl2.flows[flname].return_mutables():
                same = same*1
            else:
                same = 0
        for fxnname, fxn in mdl.fxns.items():
            if fxn.return_mutables() == mdl2.fxns[fxnname].return_mutables():
                same = same*1
            else:
                same = 0
        self.assertEqual(same, 0)

    def check_onerun_save(self, mdl, runtype, mfile, ecfile, faultscen={}):
        """
        Check if the results from the mdl are the same when saved.

        Works only for one-run propagate methods.

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
        if os.path.exists(mfile):
            os.remove(mfile)
        if os.path.exists(ecfile):
            os.remove(ecfile)
        check_link = False
        save_args = {'mdlhist': {'filename': mfile}, 'endclass': {'filename': ecfile}}
        if runtype == 'nominal':
            res, hist = prop.nominal(mdl, save_args=save_args)
            check_link = True
        elif runtype == 'one_fault':
            fxn, mode, time = faultscen
            res, hist = prop.one_fault(mdl, fxn, mode, time, save_args=save_args)
        elif runtype == 'sequence':
            res, hist = prop.sequence(mdl, faultscen, {}, save_args=save_args)
        else:
            raise Exception("Invalid Run Type" + runtype)

        self.check_same_file(hist, mfile, check_link=check_link)
        self.check_same_file(res, ecfile)

        os.remove(mfile),
        os.remove(ecfile)

    def check_same_fmea(self, fs, res, mdl):
        """Test to ensure results from the fmea are the same over all options."""
        fmea = tabulate.fmea(res, fs, mdl=mdl)
        none_exp_cost = sum(fmea['expected cost'])
        for group_by in ['phase', 'function', 'fault']:
            fmea = tabulate.fmea(res, fs, group_by=[group_by], mdl=mdl)
            exp_cost = sum(fmea['expected cost'])
            self.assertAlmostEqual(none_exp_cost, exp_cost)

    def check_same_file(self, result, resfile, check_link=False):
        """Check if the result.history is the same as the result loaded from resfile."""
        res_flattened = result.flatten()
        if isinstance(result, History):
            Rclass = History
        else:
            Rclass = Result

        res_saved = Rclass.load(resfile)
        res_saved_flattened = res_saved.flatten()
        # check to see that they are the same size
        self.assertCountEqual([*res_flattened.keys()],
                              [*res_saved_flattened.keys()])
        # check that they are the same value
        self.compare_results(res_flattened, res_saved_flattened)
        if check_link and isinstance(res_saved_flattened['time'], (np.ndarray, list)):
            # check to see that they aren't linked somehow
            res_flattened['time'][0] = 100
            self.assertNotEqual(res_flattened['time'][0],
                                res_saved_flattened['time'][0])

    def compare_results(self, res_true, res_check):
        """Check if two flattened endclass/mdlhist results dictionaries are the same."""
        # test to see that all values of the arrays in the hist are the same
        for key in res_true:
            val = res_true[key]
            if not isinstance(val, (np.ndarray, list)):
                if isinstance(val, (float, np.number)) and not np.isnan(val):
                    self.assertAlmostEqual(val, res_check[key], 4)
                else:
                    np.testing.assert_array_equal(val, res_check[key])
            elif isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.number):
                np.testing.assert_allclose(val, res_check[key])
            else:
                np.testing.assert_array_equal(val, res_check[key])

    def check_same_files(self, result, resfolder, filetype, check_link=False):
        """
        Check to see if the given result is the same as one loaded from a set of files.

        Parameters
        ----------
        result : Result
            Result to check
        resfolder : str
            Folder to check with the files
        filetype : str
            File type
        check_link : bool
            Whether to check if the results aren't linked somehow. Default is False.
        """
        result_flattened = result.flatten()
        if isinstance(result, History):
            Rclass = History
        else:
            Rclass = Result

        result_saved = Rclass.load_folder(resfolder, filetype, renest_dict=False)

        self.assertCountEqual([*result_flattened.keys()], [*result_saved.keys()])
        self.compare_results(result_flattened, result_saved)
        if check_link:
            # check to see that they aren't linked somehow
            result_flattened['time'][0] = 100
            self.assertNotEqual(result_flattened['time'][0], result_saved['time'][0])

    def start_sample_test(self, histfile, resfile):
        """Create paths for save/load tests."""
        if os.path.exists(histfile):
            os.remove(histfile)
        if os.path.exists(resfile):
            os.remove(resfile)

    def end_sample_test(self, hist, histfile, res, resfile):
        """Check files and remove from folder."""
        self.check_same_file(hist, histfile)
        self.check_same_file(res, resfile)

        os.remove(histfile)
        os.remove(resfile)

    def check_fs_save(self, mdl, fs, histfile='file', resfile='file', **kwargs):
        """
        Check to see if saved results are the same as the direct sim outputs.

        Parameters
        ----------
        mdl : Model
            Model to simulate
        fs : FaultSample
            sample to send to the method (if any)
        histfile : str
            name of file to save history in
        resfile : str
            name of file to save result in

        **kwargs : kwargs to send to the propagate method (if any)
        """
        self.start_sample_test(histfile, resfile)
        save_args = {'mdlhist': {'filename': histfile},
                     'endclass': {'filename': resfile}}
        loc_kwargs = {'save_args': save_args, 'showprogress': False, **kwargs}
        res, hist = prop.fault_sample(mdl, fs, **loc_kwargs)

        self.end_sample_test(hist, histfile, res, resfile)

    def check_sf_save(self, mdl, resfile='file', histfile='file', **kwargs):
        """Check that prop.single_faults results save/load correctly."""
        self.start_sample_test(histfile, resfile)
        save_args = {'mdlhist': {'filename': histfile},
                     'endclass': {'filename': resfile}}
        loc_kwargs = {'save_args': save_args, 'showprogress': False, **kwargs}
        res, hist = prop.single_faults(mdl, **loc_kwargs)

        self.end_sample_test(hist, histfile, res, resfile)

    def check_ps_save(self, mdl, ps, resfile='file', histfile='file', **kwargs):
        """Check that prop.parameter_sample results save/load correctly."""
        self.start_sample_test(histfile, resfile)
        save_args = {'mdlhist': {'filename': histfile},
                     'endclass': {'filename': resfile}}
        loc_kwargs = {'save_args': save_args, 'showprogress': False, **kwargs}
        res, hist = prop.parameter_sample(mdl, ps, **loc_kwargs)

        self.end_sample_test(hist, histfile, res, resfile)

    def check_ns_save(self, mdl, ps, faultdomains, faultsamples,
                      histfile='file', resfile='file', **kwargs):
        """Check that prop.nested_sample results save/load correctly."""
        self.start_sample_test(histfile, resfile)
        save_args = {'mdlhist': {'filename': histfile},
                     'endclass': {'filename': resfile}}
        loc_kwargs = {'save_args': save_args, 'showprogress': False,
                      'faultdomains': faultdomains, 'faultsamples': faultsamples,
                      **kwargs}
        res, hist, apps = prop.nested_sample(mdl, ps, **loc_kwargs)
        self.end_sample_test(hist, histfile, res, resfile)

    def start_indiv_save(self, resfolder, histfolder):
        """Remove existing folders for test."""
        if os.path.exists(histfolder):
            shutil.rmtree(histfolder)
        if os.path.exists(resfolder):
            shutil.rmtree(resfolder)

    def end_indiv_save(self, res, hist, resfolder, histfolder, ext):
        """Remove existing folders for test."""
        if ext == 'pkl':
            ext = "pickle"
        self.check_same_files(hist, histfolder, ext)
        self.check_same_files(res, resfolder, ext)
        shutil.rmtree(resfolder)
        shutil.rmtree(histfolder)

    def check_sf_isave(self, mdl, resfolder, histfolder, ext, **kwargs):
        """
        Check to see if individually saved results are the same as the direct outputs.

        Parameters
        ----------
        mdl : Model
        resfolder : str
            name of folder to save res in
        histfolder : str
            name of folder to save hist in
        ext : str
            file extension ('pkl','csv','json')
        **kwargs : **kwargs
            kwargs to send to the propagate method (if any)
        """
        self.start_indiv_save(resfolder, histfolder)
        save_args = {'mdlhist': {'filename': histfolder+"."+ext},
                     'endclass': {'filename': resfolder+"."+ext},
                     'indiv': True}
        loc_kwargs = {'save_args': save_args, 'showprogress': False, **kwargs}
        res, hist = prop.single_faults(mdl, **loc_kwargs)
        self.end_indiv_save(res, hist, resfolder, histfolder, ext)

    def check_fs_isave(self, mdl, fs, resfolder, histfolder, ext, **kwargs):
        """Check that individually saved results for prop.fault_sample match outputs."""
        self.start_indiv_save(resfolder, histfolder)
        save_args = {'mdlhist': {'filename': histfolder+"."+ext},
                     'endclass': {'filename': resfolder+"."+ext},
                     'indiv': True}
        loc_kwargs = {'save_args': save_args, 'showprogress': False, **kwargs}
        res, hist = prop.fault_sample(mdl, fs, **loc_kwargs)
        self.end_indiv_save(res, hist, resfolder, histfolder, ext)

    def check_ps_isave(self, mdl, ps, resfolder, histfolder, ext, **kwargs):
        """Check that individually saved prop.parameter_sample results match outputs."""
        self.start_indiv_save(resfolder, histfolder)
        save_args = {'mdlhist': {'filename': histfolder+"."+ext},
                     'endclass': {'filename': resfolder+"."+ext},
                     'indiv': True}
        loc_kwargs = {'save_args': save_args, 'showprogress': False, **kwargs}
        res, hist = prop.parameter_sample(mdl, ps, **loc_kwargs)
        self.end_indiv_save(res, hist, resfolder, histfolder, ext)

    def check_ns_isave(self, mdl, ps, faultdomains, faultsamples,
                       resfolder, histfolder, ext, **kwargs):
        """Check that individually saved prop.nested_sample results match outputs."""
        self.start_indiv_save(resfolder, histfolder)
        save_args = {'mdlhist': {'filename': histfolder+"."+ext},
                     'endclass': {'filename': resfolder+"."+ext},
                     'indiv': True}
        loc_kwargs = {'save_args': save_args, 'showprogress': False,
                      'faultdomains': faultdomains, 'faultsamples': faultsamples,
                      **kwargs}
        res, hist, apps = prop.nested_sample(mdl, ps, **loc_kwargs)
        self.end_indiv_save(res, hist, resfolder, histfolder, ext)
