# -*- coding: utf-8 -*-
"""
Description: Functions and Classes to enable optimization and search of fault model states and parameters.

Classes:
    - :class:`Problem`:         Creates an interface for model simulations for optimization methods
    - :class:`DynamicProblem`:  Creates an interface for model simulations for dynamic optimization of a single sim
"""


import sys, os
sys.path.insert(0,os.path.join('..','..'))
sys.path.insert(1,os.path.join('..','..', "example_pump"))

import copy
import fmdtools.faultsim.propagate as prop
import fmdtools.resultdisp.process as proc
import fmdtools.resultdisp.plot as plot
import matplotlib.pyplot as plt
import numpy as np
import warnings

class Problem(): 
    """
    Interface for resilience optimization problems. 
    
    Attributes
    ----------
        simulations : dict
            Dictionary of simulations and their corresponding arguments
        variables : list
            List of variables and their properties
        objectives : dict
            Dictionary of objectives and their times/arguments
        constraints : dict
            Dictionary of constraints and their times/arguments
        current_iter : dict
            Dictionary of current values for variables/objectives/constraints/etc.
    """
    def __init__(self, name, mdl, default_params={}, negative_form=True, **kwargs):
        """
        Instantiates the Problem object.
        
        Parameters
        ----------
        
        name : str
            Name for the problem
        mdl : Model
            Model to optimize
        negative_form : bool
            Whether constraints are negative when feasible (True) or positive when feasible (False)
        default_params : dict
            Default parameters for the model
        **kwargs : kwargs
            Default run kwargs. See :data:`sim_kwargs`, :data:`run_kwargs`, :data:`mult_kwargs`
        """
        self.name=name
        self.mdl=mdl
        self.default_params=mdl.params
        
        self.default_sim_kwargs = {k:kwargs[k] if k in kwargs else v for k,v in prop.sim_kwargs.items()}
        self.default_run_kwargs = {k:kwargs[k] if k in kwargs else v for k,v in prop.run_kwargs.items()}
        self.default_mult_kwargs = {k:kwargs[k] if k in kwargs else v for k,v in prop.mult_kwargs.items()}
        
        self.negative_form =negative_form
        self.simulations={}
        self.variables=[]
        self.objectives=dict()
        self.constraints=dict()
        self.var_mapping={}
        self.obj_const_mapping={}
        self.log={}
        self.current_iter={}
        self._sims={}
    def __repr__(self):
        var_str = "\n -"+"\n -".join('{:<63}{:>20.4f}'.format(var[2]+" "+var[0]+" at t="+str(var[3])+": "+str(var[1]), self.current_iter.get('vars', [np.NaN for j in range(i+1)])[i]) for i,var in enumerate(self.variables))
        con_str = "\n -"+"\n -".join('{:<63}{:>20.4f}'.format(name+": "+get_pos_str(greater_to_factor(con[4][0])*n_to_factor(self.negative_form))+"("+str(con[0])+" "+str(con[2])+" "+str(con[1])+" at t="+str(con[3])+" -"+str(con[4][1])+")", self.current_iter.get('consts', {name:np.NaN})[name]) for name, con in self.constraints.items() if not con[0]=="set_const")
        obj_str = "\n -"+"\n -".join('{:<63}{:>20.4f}'.format(name+": "+con[4][0]+con[4][1]+"("+con[0]+" "+str(con[2])+" "+str(con[1])+" at t="+str(con[3])+")", self.current_iter.get('objs', {name:np.NaN})[name]) for name, con in self.objectives.items())
        sim_str = "\n -"+"\n -".join(name+": "+sim[0]+" scen: "+str(sim[2].get('sequence', "")) for name, sim in self.simulations.items() if sim[0] !="set_const")
        str_repr = '{:<65}{:>20}'.format("Problem "+self.name, "current value")
        return str_repr+"\n Variables"+var_str+"\n Objectives"+obj_str+"\n Constraints"+con_str+"\n Simulations"+sim_str
    def add_simulation(self, simname, simtype, *args, **kwargs):
        """
        Defines a simulation to be used with the model

        Parameters
        ----------
        simname : str
            Name/identifier for the simulation.
        simtype : str, optional
            Type of simulation(s) to run (aligns with propagate methods):
                - single:      simulates a single scenario (the default)
                    - args: sequence: sequence defining fault scenario {time:{'faults':(fxn:mode), 'disturbances':{'Fxn1.var1'}})}
                - multi:       simulates multiple scenarios (provided approach or nominalapproach)
                    - args: scenlist: dict with structure {"scenname":{'sequence':{"faults":{}, "disturbances":{}}, "properties":{"params":params}}}
                            (can be gotten from prop.list_init_faults, SampleApproach, or NominalApproach)
                - nested:      simulates nested scenarios (provided approach and nominalapproach)
                    - args: see prop.nested_approach
                - external_func:    calls an external function (rather than a simulation)
                    - args: callable
                - custom_sim:       calls an external function with arguments (mdl) or (mdl, mdlhist)
                    - args: callable
                - set_const:        used for set constraints
        *args : args
            Custom arguments for the given simulation type (see above)
        **kwargs : dict
            run, sim, and mult_kwargs from prop. 
        """
        kwargs = {**self.default_run_kwargs,**self.default_sim_kwargs, **self.default_mult_kwargs, **kwargs}
        self.simulations[simname]=(simtype, args, kwargs)
    def add_variables(self, simnames, *args, vartype='vars', t=0): 
        """
        Adds variable of specified type ('params', 'vars', or 'faults') to a given problem. 
        Also adds variable set constraints (if given)
        
        Parameters
        ----------
        simnames : str/list
            identifier for the simulation(s) the variable is optimized over
        *args : tuples
            variables to add, where each tuple has the form:
            (varname, set_const (optional), vartype (optional), t (optional)), where
            - varname is:
                an element of mdl.params (if vartype='params')
                a model variable (if vartype='vars')
                a function name (if vartype='faults')
            - set_const defines the variable set constraints, which may be:
                None (for none/inf)
                A two-element tuple (for continuous variable bounds)
                A multi-element list (for discrete variables)
            - vartype is the individual variable type (overrides vartype)
            - t is the individual time (overrides t)
        vartype : str
            overall variable type defining the variable(s). The default is 'vars'
                - `param`: element(s) of mdl.params (set pre-simulation)
                - `vars`: function/flow variables (set during the simulation)
                - 'faults': fault scenario variables (set during the simulation)
                - 'external': variables for external func
        """
        if type(simnames)==str: simnames=[simnames]
        for arg in args:
            self._add_var(simnames, arg,vartype=vartype, t=t)
    def _add_var(self, simnames, arg, vartype='vars', t=None):
        if type(arg)==str: arg = (arg, None, None, None)
        arg = [*arg, *(None, None, None, None)][0:4]
        if not arg[2]: arg[2]=vartype
        if not arg[3]: arg[3]=t
        if arg[2] not in {'param', 'vars', 'faults', 'external'}:
            raise Exception(arg[2]+" not a legal legal variable type (param, vars, faults, or external)")
        self.variables.append(arg)
        if arg[2] in ['var', 'vars']:       vartype='disturbances'
        elif arg[2] in ['fault', 'faults']: vartype='faults'
        else:                               vartype=arg[2]
        
        ind = len(self.variables)-1
        
        if arg[3]!=None: 
            if 'set_const' not in self.simulations: self.simulations['set_const'] = ('set_const', [ind],[arg[1]])
            else:                                   
                self.simulations['set_const'][1].append(ind)
                self.simulations['set_const'][2].append(arg[1])
            self._add_obj_const('set_const', 'set_var_'+str(ind)+'_lb', (ind, 'external', 'na', ('greater',arg[1][0])), obj_const="constraints")
            self._add_obj_const('set_const', 'set_var_'+str(ind)+'_ub', (ind, 'external', 'na', ('less',arg[1][1])), obj_const="constraints")
        
        if vartype not in ['faults', 'disturbances']:   label=vartype
        else:                                           label=arg[3]
        for simname in simnames:
            self._make_mapping(self.var_mapping, simname, label, vartype, arg[0],ind)
    def add_objectives(self, simname, *args, objtype='endclass', t='end', obj_const='objectives', agg=("+",'sum'), **kwargs):
        """
        Adds objective to the given problem.
        
        Parameters
        ----------
        simname : str
            identifier for the simulation
        *args : strs/tuples
            variables to use as objectives (auto-named to f1, f2...)
            may take form: (variablename, objtype (optional), t (optional), agg (optional)) 
            or variablename, where variablename is the name of the variable (from params, mdlparams)
            or index of the callable (for external)and objtype, t, and agg may override
            the default objtype and t (see below)
        objtype : str (optional)
            default type of objective: `vars`, `endclass`, or `external`. Default is 'endclass'
        t : int (optional)
            default time to get objective: 'end' or set time t
        agg : tuple
            Specifies the aggregation of the objective/constraint:
            - for objectives: ('+'/'-','sum'/'difference'/'mult'/'max'/'min'), specifying 
            (1) whether the objective is:
                    - "+": positive (for minimization of the variable if the algorithm minimizes)
                    - "-": negative (for maximization of the variable if the algorithm minimizes)
            (2) how to aggregate objectives over scenarios 
            - for constraints: (less'/'greater', val) where value is the threshold value
        **kwargs : str=strs/tuples
            Named objectives with their corresponding arg values (see args)
            objectivename = variablename
        """
        if obj_const not in {'objectives', 'constraints'}:
            raise Exception("Invalid obj_const: "+obj_const+" (should be 'objectives' or 'constraints')")
        unnamed_objs = {'f'+str(i+len(self.obj_const)):j for i,j in enumerate(args)}
        all_objs = {**unnamed_objs, **kwargs}
        for obj in all_objs:
            self._add_obj_const(simname, obj, all_objs[obj], objtype=objtype, t=t, agg=agg, obj_const=obj_const)
    def _add_obj_const(self, simname, objname, arg, objtype='endclass', t='end',agg=("-",'sum'), obj_const='objectives'):
        if simname not in self.simulations: raise Exception("Undefined simulation: "+simname)
        if type(arg)==str: arg=[arg, objtype, t, agg]
        arg = [*arg, *(None,None,None,None)][:4]
        if not arg[1]: arg[1]=objtype
        if arg[1] not in ['vars', 'endclass', 'external']: raise Exception("Invalid objtype: "+arg[1])
        if not arg[2]:  arg[2]=t
        if not arg[3]:  arg[3]=agg
        if hasattr(self, objname): warnings.warn("Objective already defined: "+objname)
        def newobj(x):
            return self.x_to_obj_const(x)[0][objname]
        def newconst(x):
            return self.x_to_obj_const(x)[1][objname]
        if obj_const=='objectives':     objfunc = newobj
        elif obj_const=='constraints':  objfunc = newconst
        setattr(self, objname, objfunc)
        getattr(self, obj_const)[objname]=[simname,*arg]
        
        vname, objtype, t, _ = arg
        self._make_mapping(self.obj_const_mapping, simname, t,objtype,vname, exclusive_vars=False)

    def _make_mapping(self, mapping, *args, exclusive_vars=True):
        if len(args)>2:
            if args[0] not in mapping:  mapping[args[0]]=self._make_nested_dict(*args[1:])
            else:                       self._make_mapping(mapping[args[0]], *args[1:], exclusive_vars=exclusive_vars)
        elif len(args)==2:
            if args[0] not in mapping:  mapping[args[0]]=args[1]
            elif exclusive_vars:        raise Exception("Overwriting variable: "+str(args))
            elif type(mapping[args[0]])==list: mapping[args[0]].append(args[1])
            else:                               mapping[args[0]] = [mapping[args[0]], args[1]]
        else:                           raise Exception("Not enough args: "+str(args))
    def _make_nested_dict(self, *args):
        if len(args)==1:    return args[0]
        else:               return {args[0]:self._make_nested_dict(*args[1:])}
    def add_constraints(self, simname, *args, objtype='endclass', t='end', threshold=('less', 0.0), **kwargs):
        """
        Adds constraints to the given problem.
        
        Parameters
        ----------
        simname : str
            identifier for the simulation
        *args : strs/tuples
            variables to use as constraints (auto-named to f1, f2...)
            may take form: (variablename, objtype (optional), t (optional)) or variablename, where
            variablename is the name of the variable (from params, mdlparams) or index of the callable (for external)
            and objtype and t may override the default objtype and t (see below)
        objtype : str (optional)
            default type of constraint: `vars`, `endclass`, or `external`. Default is 'endclass'
        t : int (optional)
            default time to get constraint: 'end' or set time t
        threshold : tuple
            Specifies the threshold for the constraint ('less'/'greater', val) where value is the threshold value
        **kwargs : str=strs/tuples
            Named objectives with their corresponding arg values (see args)
            constraintname = variablename
        """
        self.add_objectives(simname, *args, objtype=objtype, t=t, obj_const='constraints', agg=threshold, **kwargs)
    def _get_var_obj_time(self, simname):
        var_times = [v[3] for v in self.variables]
        if 'start' in var_times:    var_time=0
        else:                       var_time = min(var_times) 
        obj_times = [v[3] for v in [*self.objectives.values(),*self.constraints.values()] if v[3]!='na']
        if 'end' in obj_times:      obj_time=self.mdl.times[-1]
        else:                       obj_time=max(obj_times)
        return var_time, obj_time
    def _prep_single_sim(self, simname, **kwargs):
        var_time, obj_time = self._get_var_obj_time(simname)
        mdl = prop.new_mdl(self.mdl, {'modelparams':{'times':self.mdl.times[:-1]+[obj_time]}})
        result, nomhist, nomscen, c_mdls, t_end = prop.nom_helper(mdl, [var_time], **{**self.simulations[simname][2], 'scen':{}})
        if self.simulations[simname][2]['sequence']:
            mdl = prop.new_mdl(self.mdl, {'modelparams':{'times':self.mdl.times[:-1]+[obj_time]}})
            scen=prop.create_faultseq_scen(mdl,  rate=1.0, sequence=self.simulations[simname][2]['sequence'])
            kwargs = {**self.simulations[simname][2], 'desired_result':{}, 'staged':False}
            kwargs.pop("sequence")
            _, prevhist, c_mdls, _  = prop.prop_one_scen(mdl, scen, ctimes = [var_time], **kwargs)
        else: prevhist = nomhist
        self._sims[simname] = {'var_time':var_time, 'nomhist':nomhist, 'prevhist':prevhist, 'obj_time': obj_time, 'mdl':mdl, 'c_mdls':c_mdls}
    def _prep_multi_sim(self, simname, **kwargs):
        var_time, obj_time = self._get_var_obj_time(simname)
        mdl = prop.new_mdl(self.mdl, {'modelparams':{'times':self.mdl.times[:-1]+[obj_time]}})
        result, nomhist, nomscen, c_mdls, t_end = prop.nom_helper(mdl, [var_time], **{**self.simulations[simname][2], 'scen':{}})
        
        scenlist = self.simulations[simname][1][0]
        for scen in scenlist: scen['properties']['time']=var_time
        
        prevhists = dict(); c_mdls = dict()
        for scen in scenlist:
            kwargs = {**self.simulations[simname][2], 'desired_result':{}, 'staged':False}
            scenname=scen['properties']['name']
            mdl = prop.new_mdl(self.mdl, {'modelparams':{'times':self.mdl.times[:-1]+[obj_time]}})
            _, prevhists[scenname], c_mdls[scenname], _  = prop.prop_one_scen(mdl, scen, ctimes = [var_time], **kwargs)
        self._sims[simname] = {'var_time':var_time, 'nomhist':nomhist, 'prevhists':prevhists, 'obj_time': obj_time, 'mdl':mdl, 'c_mdls':c_mdls}
    def _check_new_mdl(self,simname, var_time, mdl, x, c_mdl):
        if var_time==0 or not self.simulations[simname][2]['staged']: # set model parameters that are a part of the sim
            paramvars = self.var_mapping[simname].get('param',{'param':{}})
            params = {param: x[ind] for param, ind in paramvars['param'].items()}
            mdl = prop.new_mdl(mdl, {'params':params})
        elif self.simulations[simname][2]['staged']:  mdl = c_mdl[var_time].copy()
        return mdl
    def _run_single_sim(self, simname, x):
        sim = self._sims[simname]
        var_time, prevhist, nomhist, obj_time, mdl, c_mdl = sim['var_time'], sim['prevhist'], sim['nomhist'], sim['obj_time'], sim['mdl'], sim['c_mdls']
        
        mdl = self._check_new_mdl(simname, var_time, mdl, x, c_mdl)
        # set model faults/disturbances as elements of scenario 
        ##NOTE: need to make sure scenarios don't overwrite each other
        scen=prop.construct_nomscen(mdl)
        scen['sequence'] = self._update_sequence(self.simulations[simname][2]['sequence'], simname, x)
        scen['properties']['time'] = var_time
        #propagate scenario, get results
        des_r=copy.deepcopy(self.obj_const_mapping[simname])
        kwargs = {**self.simulations[simname][2], "desired_result":des_r, "nomhist":nomhist, "prevhist":prevhist}
        kwargs.pop("sequence")
        result, mdlhist, _, _ = prop.prop_one_scen(mdl, scen, **kwargs)
        self._sims[simname]['mdlhists'] = {"faulty":mdlhist, "nominal":nomhist}
        self._sims[simname]['result'] = result
        # log, return objectives, etc
        objs = self._get_obj_from_result(simname, result, "objectives")
        consts = self._get_obj_from_result(simname, result, "constraints")
        return objs, consts
    def _run_multi_sim(self, simname, x):
        sim = self._sims[simname]
        var_time, prevhists, nomhist, obj_time, mdl, c_mdls = sim['var_time'], sim['prevhists'], sim['nomhist'], sim['obj_time'], sim['mdl'], sim['c_mdls']
        
        pool = self.simulations[simname][2]['pool']
        scenlist = self.simulations[simname][1][0]
        staged = self.simulations[simname][2]['staged']
        kwargs = self.simulations[simname][2]
        kwargs = {**kwargs, 'desired_result':copy.deepcopy(self.obj_const_mapping[simname]), "pool":True}
        results, objs, consts, mh = {}, {}, {}, {}
        new_scenlist = []
        for scen in scenlist:
            newscen = copy.deepcopy(scen)
            newscen['sequence']= self._update_sequence(newscen['sequence'], simname, x)
            new_scenlist.append(newscen)
        scenlist = new_scenlist
        if pool: 
            if staged:  inputs = [(self._check_new_mdl(simname, var_time, mdl, x, c_mdls[scen['properties']['name']][var_time]), scen, {**kwargs, 'prevhist':prevhists[scen['properties']['name']]},  str(i)) for i, scen in enumerate(scenlist)]
            else:       inputs = [(self._check_new_mdl(simname, var_time, mdl, x, mdl), scen,  kwargs, str(i)) for i, scen in enumerate(scenlist)]
            res_list = list(pool.imap(prop.exec_scen_par, inputs))
            results, mh = prop.unpack_res_list(scenlist, res_list)
        else:
            for i, scen in enumerate(scenlist):
                name = scen['properties']['name']
                prevhist = prevhists[name]
                kwargs['nomhist'] = nomhist
                kwargs['desired_result'] = copy.deepcopy(self.obj_const_mapping[simname])
                mdl = self._check_new_mdl(simname, var_time, mdl, x, c_mdls[name][var_time])
                results[name], mh[name], t_end = prop.exec_scen(mdl, scen, indiv_id=str(i), **kwargs, prevhist=prevhist)
        objs = self._get_obj_from_results(simname, results, "objectives")
        consts = self._get_obj_from_results(simname, results, "constraints")
        self._sims[simname]['mdlhists'] = mh
        return objs, consts
    def _update_sequence(self, existing_sequence, simname, x):
        new_sequence = {t:{k:{var: x[ind] for var,ind in v.items()} for k,v in v.items()} for t,v in self.var_mapping[simname].items() if type(t) in [int, float]}
        return {**existing_sequence, **new_sequence}
    def _get_obj_from_results(self,simname, results, obj_const):
        objs={}
        obj_dict = getattr(self, obj_const)
        for objname, var_to_get in obj_dict.items():
            if simname ==var_to_get[0]:
                values = [result[var_to_get[3]][var_to_get[2]][var_to_get[1]] if var_to_get[3] in result else result[var_to_get[2]][var_to_get[1]] for result in results.values()]
                if obj_const == 'objectives':    
                    pos_fact = get_pos_negative(self.objectives[objname][-1][0])
                    aggfunc = getattr(np, self.objectives[objname][-1][1])
                    objs[objname] = pos_fact*aggfunc(values)
                elif obj_const == 'constraints': 
                    values = [eval_con(value,var_to_get[4][1], var_to_get[4][0], self.negative_form) for value in values]
                    if self.negative_form:  objs[objname]=np.min(values)
                    else:                   objs[objname]=np.max(values)
                else: raise Exception("Invalid type: "+obj_const)
        return objs
    def _get_obj_from_result(self, simname, result, obj_const):
        objs={}
        obj_dict = getattr(self, obj_const)
        for objname, var_to_get in obj_dict.items():
            if simname ==var_to_get[0]:
                if var_to_get[3] in result:     value = result[var_to_get[3]][var_to_get[2]][var_to_get[1]]
                elif var_to_get[2] in result:   value = result[var_to_get[2]][var_to_get[1]]
                if obj_const == 'objectives':       objs[objname] = get_pos_negative(self.objectives[objname][-1][0])*value
                elif obj_const == 'constraints':    objs[objname] = eval_con(value, var_to_get[4][1], var_to_get[4][0], self.negative_form)
        return objs
    def _get_set_const(self,x):
        ubs = {"set_var_"+str(i)+"_ub": eval_con(x[i], self.simulations['set_const'][2][i][1], "less", self.negative_form) for i in self.simulations['set_const'][1]}
        lbs = {"set_var_"+str(i)+"_lb": eval_con(x[i], self.simulations['set_const'][2][i][0], "greater", self.negative_form) for i in self.simulations['set_const'][1]}
        return {}, {**ubs, **lbs}
    def _prep_sim_type(self, simtype, simname):
        if simtype=='single': self._prep_single_sim(simname)
        if simtype=='multi': self._prep_multi_sim(simname)
    def _run_sim_type(self, simtype, simname, x):
        if simtype=='set_const': return self._get_set_const(x)
        elif simtype=='single':  return self._run_single_sim(simname, x)
        elif simtype=='multi':   return self._run_multi_sim(simname,x)
        else: raise Exception("Invalid simulation type: "+simtype)
    def x_to_obj_const(self, x):
        """
        Calculates objectives and constraints for a given variable value

        Parameters
        ----------
        x : list/array
            Variable values corresponding to self.variables

        Returns
        -------
        objectives : dict
            Dictionary of objectives and their values
        constraints
            Dictionary of constraints and their values

        """
        if type(x)==list: x=np.array(x)
        if not self.current_iter: 
            for simname in self.simulations:
                self._prep_sim_type(self.simulations[simname][0], simname)
            self.current_iter = {'vars':np.array([np.nan]), 'objs':dict(), 'consts':dict()}
        if any(self.current_iter['vars']!=x):
            self.current_iter['vars'] = x
            for simname in self.simulations:
                objs, consts = self._run_sim_type(self.simulations[simname][0], simname, x)
                self.current_iter['objs'].update(objs)
                self.current_iter['consts'].update(consts)
        return self.current_iter['objs'], self.current_iter['consts']
    def _get_plot_vals(self, simname, obj_con_var):
        vals = {n:o[1] for n,o in obj_con_var.items() if o[0]==simname and o[2]=='vars'}
        return vals
    def _get_plot_vars(self, simname, variables):
        vals = {"x_"+str(i):o[0] for i,o in enumerate(variables) if  o[2]=='vars'}
        return vals
    def _get_plot_times(self, simname, obj_con_var):
        vals = {n:get_text_time(o[3], end=self.mdl.times[-1]) for n,o in obj_con_var.items() if o[0]==simname and o[2]=='vars'}
        return vals
    def _get_var_times(self):
        vals = {"x_"+str(i):get_text_time(o[3]) for i,o in enumerate(self.variables) if o[2]=='vars'}
        return vals
    def get_var_obj_con(self):
        variables = {"x_"+str(i): self.current_iter['vars'][i] for i, var in enumerate(self.current_iter['vars'])}
        return {**variables, **self.current_iter['objs'], **self.current_iter['consts']}
    def plot_obj_const(self, simname, **kwargs):
        """
        Plots the objectives, variables, and constraints over time at the current variable value. Note that simulation tracking must be turned on.
        
        Parameters
        ----------
        simname : str
            Name of the simulation.
        kwargs : kwargs
            Keyword arguments for plot.mdlhists
        """
        objs_to_plot = self._get_plot_vals(simname, {**self.objectives, **self.constraints})
        vars_to_plot = self._get_plot_vars(simname, self.variables)
        all_to_plot = {**objs_to_plot, **vars_to_plot}
        
        fxnflowvals = proc.nest_flattened_hist({tuple(objname.split(".")):'all' for objname in all_to_plot.values()})
            
        if self.simulations[simname][0]=='multi':      f_times = {get_text_time(t)  for seq in self.simulations[simname][1][0] for t in seq['sequence'].keys()}
        elif self.simulations[simname][0]=='single':   f_times = {get_text_time(t) for t in self.simulations[simname][2]['sequence'].keys()}
        
        fig, axs = plot.mdlhists(self._sims[simname]['mdlhists'], fxnflowvals=fxnflowvals, time_slice=f_times, **kwargs)
        
        vars_ordered = [".".join(ax.get_title().split(": ")) for ax in axs]
        rev_all_to_plot = {v:k for k,v in all_to_plot.items()}
        objnames_ordered = [rev_all_to_plot[v] for v in vars_ordered if v in rev_all_to_plot]
        
        vartimes = self._get_var_times()
        objcontimes = self._get_plot_times(simname, {**self.objectives, **self.constraints})
        times = {**vartimes, **objcontimes}
        current_vars = self.get_var_obj_con()
        for i, val in enumerate(objnames_ordered):
            #axs[i].vlines([times[val]], *axs[i].get_ylim())
            axs[i].axvline([times[val]], color="grey", ls="--")
            mid = np.mean(axs[i].get_ylim())
            axs[i].text(times[val], mid, val+"="+'{0:.2f}'.format(current_vars[val]),horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.5, edgecolor='white'))
        return fig, axs
    def get_constraint_dict(self, include_set_consts=True):
        constraint_dict = {}
        for con in self.constraints:
            if (not "set_var" in con) or include_set_consts:
                if self.constraints[con][4][0] in ["greater", "less"]:  con_type="ineq"
                else:                                                   con_type="eq"
                constraint_dict[con]={"type":con_type, "fun":getattr(self, con)}
        return constraint_dict
    def get_constraint_list(self, include_set_consts=True):
        return [*self.get_constraint_dict().values()]

def eval_con(value, threshold, greater_less, negative_form):
    g_factor = greater_to_factor(greater_less)
    n_factor = n_to_factor(negative_form)
    return g_factor*n_factor*(value-threshold)

def greater_to_factor(greater_less):
    if greater_less =="greater":    g_factor = 1
    elif greater_less=="less":      g_factor = -1
    else: raise Exception("Invalid option for greater_less: "+str(greater_less))
    return g_factor
def n_to_factor(n):
    if n:          f_factor = -1
    else:          f_factor = 1
    return f_factor
def get_pos_negative(pos_str):
    if pos_str=="-":    pos_fact= -1
    elif pos_str=="+":  pos_fact= +1
    else: raise Exception("Invalid aggregation: "+pos_str)
    return pos_fact
def get_pos_str(factor):
    if factor>=0:   pos_str = "+"
    else:           pos_str = "-"
    return pos_str
def get_text_time(time,start=0,end=0):
    if time=='start':   t=start
    elif time=='end':   t=end
    else:               t=time
    return t
class DynamicProblem():
    """ 
    Interface for dynamic search of model states (e.g., AST)
    
    Attributes:
        t : float
            time
        t_max : float
            max time
        t_ind : int
            time index in log
        desired_result : list
            variables to get from the model at each time-step
        log : dict
            mdlhist for simulation
    """
    def __init__(self, mdl, paramdict={}, t_max=False, track="all", run_stochastic="track_pdf", desired_result=[], use_end_condition=None):
        """
        Initializing the problem

        Parameters
        ----------
        mdl : Model
            Model defining the simulation.
        paramdict : dict, optional
            Parameters to run the model at. The default is {}.
        t_max : float, optional
            Maximum simulation time. The default is False.
        track : str/dict, optional
            Properties of the model to track over time. The default is "all".
        run_stochastic : bool/str, optional
            Whether to run stochastic behaviors (True/False) and/or return pdf "track_pdf". The default is "track_pdf".
        desired_result : list, optional
            List of desired results to return at each update. The default is [].
        use_end_condition : bool, optional
            Whether to use model end-condition. The default is None.
        """
        self.t=0.0
        self.t_ind=0
        if not t_max:   self.t_max=mdl.times[-1]
        else:           self.t_max = t_max
        if type(desired_result)==str:   self.desired_result=[desired_result]
        else:                           self.desired_result = desired_result
        self.mdl = prop.new_mdl(mdl, paramdict)
        self.log = prop.init_mdlhist(mdl, np.arange(self.t, self.t_max+2*mdl.tstep, self.mdl.tstep), track=track)
        self.run_stochastic=run_stochastic
        if use_end_condition==None and hasattr(mdl, "end_condition"): 
            self.use_end_condition = mdl.use_end_condition
        else:                       
            self.use_end_condition = use_end_condition
    def update(self, seed={}, faults={}, disturbances={}):
        """
        Updates the model states at the simulation time and iterates time

        Parameters
        ----------
        seed : seed, optional
            Seed for the simulation. The default is {}.
        faults : dict, optional
            faults to inject in the model, with structure {fxn:[faults]}. The default is {}.
        disturbances : dict, optional
            Variables to change in the model, with structure {fxn.var:value}. The default is {}.

        Returns
        -------
        returns : dict
            dictionary of returns with values corresponding to desired_result
        """
        if seed: self.mdl.update_seed(seed)
        prop.propagate(self.mdl,self.t, fxnfaults=faults, disturbances=disturbances, run_stochastic=self.run_stochastic)
        prop.update_mdlhist(self.mdl, self.log, self.t_ind)
        
        returns = {}
        for result in self.desired_result:      returns[result] = self.mdl.get_vars(result)
        if self.run_stochastic=="track_pdf":    returns['pdf'] = self.mdl.return_probdens()

        self.t += self.mdl.tstep
        self.t_ind +=1
        if returns: return returns
    def check_sim_end(self, external_condition=False):
        """
        Checks the model end-condition (and sim time) and clips the simulation log if necessary
        
        Parameters
        ----------
        external_condition : bool, optional
            External end-condition to trigger simulation end. The default is False.
        
        Returns
        ----------
        end : bool
            Whether the simulation is finished
        """
        if self.t>=self.t_max:                                      end = True
        elif self.use_end_condition and self.mdl.end_condition():   end = True
        elif external_condition:                                    end = True
        else:                                                       end = False
        if end: prop.cut_mdlhist(self.log, self.t_ind)
        return end
        
    
    

if __name__=="__main__":

    
    from fmdtools.modeldef import SampleApproach
    import multiprocessing as mp
    from scipy.optimize import minimize
    
    from pump_stochastic import *
    
    mdl=Pump()
    app = SampleApproach(mdl, faults="ExportWater", phases=["on"], defaultsamp={'samp':'evenspacing','numpts':4})
    multi_problem = Problem("multi_problem", mdl, staged=True) #, track='valparams')
    multi_problem.add_simulation("test_multi", "multi", app.scenlist)
    multi_problem.add_variables("test_multi", ("delay", [0,15]), vartype='param')
    multi_problem.add_variables("test_multi", ("MoveWater.eff", [0,15]), t=10)
    multi_problem.add_objectives("test_multi", cost="expected cost", objtype='endclass')
    
    multi_problem.cost([10,1])
    
    
    #plot.mdlhists(multi_problem._sims['test_multi']['mdlhists'], fxnflowvals={"MoveWater":['total_flow', 'eff'], "EE_1":{"voltage"}, "Wat_2":{"area"}}, time_slice=[2, 10, 27,29, 52], legend_loc=False)
    
    
    