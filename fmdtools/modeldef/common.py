# -*- coding: utf-8 -*-
"""
Description: A module to define base data structures for simulation. Contains:
    
- :class:`States`:      Class operations for states (inherited by FxnClass, Flow, etc)
"""
import numpy as np
from operator import attrgetter
import warnings
from recordclass import dataobject, asdict, recordclass
from collections.abc import Iterable
import dill
import copy
import inspect   
from scipy import stats 


#def container(name="Container", **kwargs):
#    """Simple factory method for creating a container for states, modes, components, etc"""
#    return recordclass(name, tuple([(k,type(v)) for k, v in kwargs.items()]), tuple([v for v in kwargs.values()]), mapping=True)

def get_true_fields(dataobject, *args,  force_kwargs = False, **kwargs):
    """
    Resolves the args to pass to a dataobject given certain defaults, *args and **kwargs
    
    NOTE: must be used for pickling, since pickle passes arguments as *args and not **kwargs.
    """
    true_args = list(dataobject.__defaults__)
    for i, n in enumerate(dataobject.__fields__):
        if force_kwargs:    true_args[i]=kwargs[n]
        if i<len(args):     true_args[i]=args[i]
        elif n in kwargs:   true_args[i]=kwargs[n]
    return true_args
def get_true_field(dataobject, fieldname, *args, **kwargs):
    """Gets the value that will be set to fieldname given *args and **kwargs"""
    if fieldname in kwargs:                         return kwargs[fieldname]
    field_ind = dataobject.__fields__.index(fieldname)
    if args and len(args)>field_ind:                return args[field_ind]
    else:                                           return dataobject.__defaults__[field_ind]

class State(dataobject, mapping=True):
    """ """
    def set_atts(self, **kwargs):
        """Sets the given arguments to a given value. Mainly useful for 
        reducing length/adding clarity to assignment statements in __init__ methods
        (self.put is reccomended otherwise so that the iteration is on function/flow *states*)
        e.g., self.set_attr(maxpower=1, maxvoltage=1) is the same as saying
              self.maxpower=1; self.maxvoltage=1
        """
        for name, value in kwargs.items():
            setattr(self, name, value)
    def put(self, as_copy=True, **kwargs):
        """Sets the given arguments to a given value. Mainly useful for 
        reducing length/adding clarity to assignment statements.
        e.g., self.EE.put(v=1, a=1) is the same as saying
              self.EE.v=1; self.EE.a=1
        
        as_copy: bool, set to True for dicts/sets to be copied rather than referenced
        """
        for name, value in kwargs.items():
            if name not in self.__fields__: raise Exception(name+" not a property of "+str(self.__class__))
            if as_copy: value=copy.copy(value)
            setattr(self, name, value)
    def assign(self,obj,*states, as_copy=True, **statedict):
        """ Sets the same-named values of the current flow/function object to those of a given flow. 
        Further arguments specify which values.
        e.g. self.EE1.assign(EE2, 'v', 'a') is the same as saying
            self.EE1.a = self.EE2.a; self.EE1.v = self.EE2.v
        Can also be used to assign list values to a variable
        e.g. self.Pos.assign([1,2,3],'x','y','z')
        Can also provide dict in case value names don't match
        e.g. self.Pos_out.assign(self.Pos_in, x='dx',y='dy')
        as_copy: bool, set to True for dicts/sets to be copied rather than referenced
        """
        if type(obj)==list or isinstance(obj, np.ndarray):
            for i, state in enumerate(states):  
                if as_copy: val=copy.copy(obj[i])
                else:       val=obj[i]
                setattr(self, state, val)
        else:
            if not statedict:
                if len(states)==0:    statedict = {s:s for s in obj.__fields__}
                else:                 statedict = {s:s for s in states}
            elif len(states)>0: raise Exception("Can only provide positional states or keyword states, not both")
            for set_state, get_state in statedict.items():
                if set_state not in self.__fields__: raise Exception(set_state+" not a property of "+self.name)
                val = getattr(obj,get_state)
                if as_copy: val=copy.copy(val)
                setattr(self, set_state, val)
    def get(self, *attnames, **kwargs):
        """Returns the given attribute names (strings). Mainly useful for reducing length
        of lines/adding clarity to assignment statements.
        e.g., x,y = self.Pos.get('x','y') is the same as
              x,y = self.Pos.x, self.Pos.y, or
              z = self.Pos.get('x','y') is the same as
              z = np.array([self.Pos.x, self.Pos.y])
        """
        if len(attnames)==1:    states = getattr(self,attnames[0])
        else:                   states = [getattr(self,name) for name in attnames]
        if not is_iter(states):                 return states
        elif len(states)==1:                    return states[0]
        elif kwargs.get('as_array', True):      return np.array(states)
        else:                                   return states
    def values(self):
        return self.gett(*self.__fields__)
    def gett(self, *attnames):
        """Alternative to self.get that returns the given constructs as a tuple instead
        of as an array. Useful when a numpy array would translate the underlying data types
        poorly (e.g., np.array([1,'b'] would make 1 a string--using a tuple instead preserves
        the data type)"""
        states = self.get(*attnames,as_array=False)
        if not is_iter(states):                 return states
        elif len(states)==1:                    return states[0]
        else:                                   return tuple(states)
    def inc(self,**kwargs):
        """Increments the given arguments by a given value. Mainly useful for
        reducing length/adding clarity to increment statements.
        e.g., self.Pos.inc(x=1,y=1) is the same as
             self.Pos.x+=1; self.Pos.y+=1, or
             self.Pos.x = self.Pos.x + 1; self.Pos.y = self.Pos.y +1
             
        Can additionally be provided with a second value denoting a limit on the increments
        e.g. self.Pos.inc(x=(1,10)) will increment x by 1 until it reaches 10
        """
        for name, value in kwargs.items():
            if name not in self.__fields__: raise Exception(name+" not a property of "+self.name)
            if type(value)==tuple:  
                current = getattr(self,name)
                sign = np.sign(value[0])
                newval = current + value[0]
                if sign*newval <= sign*value[1]:    setattr(self, name, newval)
                else:                               setattr(self,name,value[1])
            else:                   setattr(self, name, getattr(self,name)+ value)
    def roundto(self, **kwargs):
        """
        Rounds the given arguments to a given resolution.
        e.g., self.Pos.roundto(x=0.1) will round Pos.x to the nearest 0.1.
        """
        for name, value in kwargs.items():
            current = getattr(self,name)
            setattr(self, name, round(current/value)*value)
    def limit(self,**kwargs):
        """Enforces limits on the value of a given property. Mainly useful for
        reducing length/adding clarity to increment statements.
        e.g., self.EE.limit(a=(0,100), v=(0,12)) is the same as
            self.EE.a = min(100, max(0,self.EE.a));
            self.EE.v = min(12, max(0,self.EE.v))
        """
        for name, value in kwargs.items():
            if name not in self.__fields__: raise Exception(name+" not a property of "+self.name)
            setattr(self, name, min(value[1], max(value[0], getattr(self,name))))
    def mul(self,*states):
        """Returns the multiplication of given attributes of the model construct.
        e.g.,   a = self.mul('x','y','z') is the same as
                a = self.x*self.y*self.z
        """
        a= self.get(states[0])
        for state in states[1:]:
            a = a * self.get(state)
        return a
    def div(self,*states):
        """Returns the division of given attributes of the model construct
        e.g.,   a = self.div('x','y','z') is the same as
                a = (self.x/self.y)/self.z
        """
        a= self.get(states[0])
        for state in states[1:]:
            a = a / self.get(state)
        return a
    def add(self,*states):
        """Returns the addition of given attributes of the model construct
        e.g.,   a = self.add('x','y','z') is the same as
                a = self.x+self.y+self.z
        """
        a= self.get(states[0])
        for state in states[1:]:
            a += self.get(state)
        return a
    def sub(self,*states):
        """Returns the subtraction of given attributes of the model construct
        e.g.,   a = self.div('x','y','z') is the same as
                a = (self.x-self.y)-self.z
        """
        a= self.get(states[0])
        for state in states[1:]:
            a -= self.get(state)
        return a
    def same(self,values, *states):
        """Tests whether a given iterable values has the same value as each
        give state in the model construct.
        e.g.,   self.same([1,2],'a','b') is the same as
                all([1,2]==[self.a, self.b])"""
        test = values==self.get(*states)
        if is_iter(test):   return all(test)
        else:               return test
    def different(self,values, *states):
        """Tests whether a given iterable values has any different value the
        given states in the model construct.
        e.g.,   self.same([1,2],'a','b') is the same as
                any([1,2]!=[self.a, self.b])"""
        test = values!=self.get(*states)
        if is_iter(test):   return any(test)
        else:               return test
    def set_var(self,var, val):
        """
        Sets variable of the object to a given value

        Parameters
        ----------
        var : list/tuple of strings
            list of nested attributes
        val : attr
            attribute to set the value to

        Returns
        -------
        flowdict : dict
            dict of flows indexed by flownames
        """
        if type(var)==str: var=var.split(".")
        #if not attrgetter(".".join(var))(self): raise Exception("Attibute does not exist: "+str(var))
        
        if len(var)==1: setattr(self, var[0], val)
        else: 
            if getattr(self, var[0]): 
                subattr = getattr(self, var[0])
                if hasattr(subattr, 'set_var'): subattr.set_var(var[1:], val)
                elif type(subattr)==dict:  
                    if var[1] not in subattr:
                        subattr[eval(var[1])]=val
                    else:                       
                        subattr[var[1]]=val
                else: raise Exception("Model sub-attribute "+str(subattr)+" does not inherit from Common")
            else: raise Exception("Invalid variables :"+str(var))
    def get_var(self, var):
        """
        Gets the variable value of the object

        Parameters
        ----------
        var : str/list
            list specifying the attribute (or sub-attribute of the object
        Returns
        -------
        var_value: any
            value of the variable
        """
        if type(var)==str: var=var.split(".")
        return attrgetter(".".join(var))(self)
    def warn(self, *messages, stacklevel=2):
        """
        Prints warning message(s) when called.

        Parameters
        ----------
        *messages : str
            Strings to make up the message (will be joined by spaces)
        stacklevel : int
            Where the warning points to. The default is 2 (points to the place in the model)
        """
        warnings.warn(' '.join(messages), stacklevel=stacklevel)


def is_iter(data):
    """ Checks whether a data type should be interpreted as an iterable or not and returned
    as a single value or tuple/array"""
    if isinstance(data, Iterable) and type(data)!=str:  return True
    else:                                               return False

def check_pickleability(obj, verbose=True):
    """ Checks to see which attributes of an object will pickle (and thus parallelize)"""
    unpickleable = []
    for name, attribute in vars(obj).items():
        if not dill.pickles(attribute):
            unpickleable = unpickleable + [name]
    if verbose:
        if unpickleable: print("The following attributes will not pickle: "+str(unpickleable))
        else:           print("The object is pickleable")
    return unpickleable


class Parameter(dataobject, readonly=True):
    """
    The Parameter class defines model/function/flow values which are immutable,
    that is, the same from model instantiation through a simulation. Parameters 
    inherit from recordclass, giving them a low memory footprint, and use type
    hints and ranges to ensure parameter values are valid.
    
    e.g.,
    class Param(Parameter, readonly=True):
        x:          float = 30.0
        y:          float = 30.0
        x_lim = (0.0,100.0)
        y_set = (0.0,30.0,100.0)
    defines a parameter with float x and y fields with default values of 30 and
    x_lim minimum/maximum values for x and y_set possible values for y. Note that
    readonly=True should be set to ensure fields are not changed.
    
    This parameter can then be instantiated using:
        p = Param(x=1.0, y=0.0)
    """
    def __init__(self, *args, strict_immutability=True,**kwargs):
        """
        Initializes the parameter with given kwargs.

        Parameters
        ----------
        strict_immutability : bool
            Performs basic checks to ensure fields are immutable
        
        **kwargs : kwargs
            Fields to set to non-default values.
        """
        if not self.__doc__: 
            raise Exception("Please provide docstring")
            #self.__doc__=Parameter.__doc__
        for k, v in kwargs.items():
            self.check_lim(k,v)
        try:
            super().__init__(*args, **kwargs)
        except TypeError:
            raise Exception("Invalid args/kwargs: "+str(args)+" , "+str(kwargs))
        if strict_immutability: self.check_immutable()
        self.check_type()
        self.check_pickle()
    def check_lim(self, k, v):
        """
        Checks to ensure the value v for field k is within the defined limits
        self.k_lim or set constraints self.k_set

        Parameters
        ----------
        k : str
            Field to check
        v : mutable
            Value for the field to check

        Raises
        ------
        Exception
            Notification that the field is outside limits/set constraints.
        """
        var_lims = getattr(self, k+"_lim", False)
        if var_lims:
            if not(var_lims[0]<=v<=var_lims[1]):
                raise Exception("Variable "+k+" ("+str(v)+") outside of limits: "+str(var_lims))
        var_set = getattr(self, k+"_set", False)
        if var_set:
            if not(v in var_set):
                raise Exception("Variable "+k+" ("+str(v)+") outside of set constraints: "+str(var_set))
    def check_immutable(self):
        """
        (woefully incomplete) check to ensure defined field values are immutable.
        Checks if a known/common mutable or a known/common immutable, otherwise 
        gives a warning.

        Raises
        ------
        Exception
            Throws exception if a known mutable (e.g., dict, set, list, etc)
        """
        for f in self.__fields__:
            attr = getattr(self, f)
            attr_type = type(attr)
            if isinstance(attr, (list, set, dict)):
                raise Exception("Parameter "+f+" type "+str(attr_type)+" is mutable")
            elif not isinstance(attr, (int, float, tuple, str, Parameter, np.number)):
                warnings.warn("Parameter "+f+" type "+str(attr_type)+" may be mutable")
    def check_type(self):
        """
        Checks to ensure Parameter type-hints are being followed.

        Raises
        ------
        Exception
            Raises exception if a field is not the same as its defined type.
        """
        for typed_field in self.__annotations__:
            attr_type = type(getattr(self, typed_field))
            true_type = self.__annotations__.get(typed_field, False)
            if ((true_type and not attr_type==true_type) and
                str(true_type).split("'")[1] not in str(attr_type)): # weaker, but enables use of np.str, np.float, etc
                raise Exception(typed_field+" assigned incorrect type: "+str(attr_type)+" (should be "+str(true_type)+")")
    def copy_with_vals(self, **kwargs):
        """Creates a copy of itself with modified values given by kwargs"""
        return self.__class__(**{**asdict(self), **kwargs})
    def check_pickle(self):
        """Checks to make sure pickled object will get *args and **kwargs"""
        signature = str(inspect.signature(self.__init__))
        if not ('*args' in signature) and ('**kwargs' in signature):
            raise Exception("*args and **kwargs not in __init__()--will not pickle.")
    def get_true_field(self, fieldname, *args, **kwargs):
        return get_true_field(self, fieldname, *args, **kwargs)
    def get_true_fields(self, *args, **kwargs):
        return get_true_field(self, *args, **kwargs)

class Rand(dataobject, mapping=True):
    rng:            np.random.default_rng
    probs:          list = list()
    seed:           int =   42
    run_stochastic: bool=False
    def __init__(self, *args, seed=42, run_stochastic=False, probs = list(), s_kwargs={}):
        args = get_true_fields(self, *args, seed=seed, run_stochastic=run_stochastic, probs=probs)
        super().__init__(*args)
        self.rng = np.random.default_rng(self.seed)
        if 's' in self.__fields__:
            self.s.set_atts(**s_kwargs)
    def get_rand_states(self, auto_update_only=False):
        rand_states = asdict(self.s)
        if auto_update_only:
            rand_states = {state:vals for state,vals in  rand_states if hasattr(self.s, state+"_update")}
        return rand_states
    def set_rand(self,statename,methodname, args):
        """
        Update the given random state with a given method and arguments (if in run_stochastic mode)

        Parameters
        ----------
        statename : str
            name of the random state defined in assoc_rand_state(s)
        methodname : 
            str name of the numpy method to call in the rng
        *args : args
            arguments for the numpy method
        """
        if getattr(self, 'run_stochastic', True):
            gen_method = getattr(self.rng, methodname)
            newvalue = gen_method(*args)
            setattr(self.s, statename, newvalue)
            if self.run_stochastic == 'track_pdf':
                value_pds = get_pdf_for_rand(newvalue, methodname, args)
                self.pds.extend(value_pds)
    def return_probdens(self):
        if self.pds: state_pd= np.prod(self.pds)
        else:        state_pd= 1.0
        return state_pd
    def update_stochastic_states(self):
        """Updates the defined stochastic states defined to auto-update."""
        if hasattr(self,'s'):
            if self.run_stochastic == 'track_pdf': self.pds.clear()
            for state in self.s.__fields__:
                if hasattr(self.s, state+"_update"):
                    self.set_rand(state, *getattr(self.s, state+'_update'))
    def reset(self):
        self.s.reset()
        self.rng = np.random.default_rng(self.seed)
    def assign(self, other_rand):
        if hasattr(self,'s'):
            self.s.assign(other_rand.s)
        self.rng.__setstate__(other_rand.rng.__getstate__())
    def get_true_field(self, fieldname, *args, **kwargs):
        return get_true_field(self, fieldname, *args, **kwargs)
    def get_true_fields(self, *args, **kwargs):
        return get_true_field(self, *args, **kwargs)

def get_pdf_for_rand(x, randname, args):
    """
    Gets the corresponding probability mass/density for  
    for random sample x from 'randname' function in numpy.
    
    Parameters
    ----------
    x : int/float/array
        samples to get probability mass/density of
    randname : str
        Name of numpy.random distribution
    args : tuple
        Arguments sent to numpy.random distribution

    Returns
    -------
    prob: float/array of probability densities
    """
    if type(x) not in [np.ndarray, list]: x=[x]
    if randname=='integers':
        if len(args)==1:        pd= [1/args[0] for x in x]
        elif len(args)>=2:      pd= [1/(args[1]-args[0]) for x in x]
    elif randname=='random':    pd= [1 for x in x]
    elif randname=='bytes':     raise Exception("Not able to calculate pdf for bytes")
    elif randname=='choice':    
        if type(args[0])==int:  options = [*np.arange(args[0])]
        else:                   options = args[0]
        if len(args)==4:        p = args[3]
        else:                   p = [1/len(options) for i in options]
        pd= [p[options.index(i)]  for i in x]
    elif randname in ['shuffle', 'permutation']:
        pd= [1/np.math.factorial(len(args[0]))]
    elif randname=='permuted':
        if len(args)>1 and type(args[0])==np.ndarray:
            pd= [1/np.math.factorial(args[0].shape(args[1]))]
        else:
            pd= [1/np.math.factorial(len(args[0]))]
    else:
        pd = get_pdf_for_dist(x,randname,args)
    if type(pd)==list:              pd=np.array(pd)
    elif type(pd) != np.ndarray:    pd=np.array([pd]) 
    return pd

def get_scipy_pdf_helper(x, randname, args,pmf=False):
    """
    Gets probability mass/density for the outcome x from the distribution "randname" with arguments "args".
    Used as a helper function in determining stochastic model state probability

    Parameters
    ----------
    x : int/float/array
        samples to get probability mass/density of
    randname : str
        Name of scipy.stats probability distribution
    args : tuple
        Arguments to send to scipy.stats.randname.pdf
    pmf : Bool, optional
        Whether the distribution uses a probability mass function instead of a pdf. The default is False.

    Returns
    -------
    prob: float/array of probability densities

    """
    if randname=='dirichlet':
        a=1
    if pmf:     return getattr(stats, randname).pmf(x, *args)
    else:       return getattr(stats, randname).pdf(x, *args)
def get_pdf_for_dist(x, randname, args): # note: when python 3.10 releases, this should become match/case
    """
    Gets the corresponding probability mass/density (from scipy) for outcome x 
    for probability distributions with name 'randname' in numpy.
    
    Parameters
    ----------
    x : int/float/array
        samples to get probability mass/density of
    randname : str
        Name of numpy.random distribution
    args : tuple
        Arguments sent to numpy.random distribution

    Returns
    -------
    prob: float/array of probability densities
    
    """
    if type(x) in [np.ndarray, list] and len(x)>1 and len(args)>0: args=args[:-1]
    
    same_funcs = ['beta', 'dirichlet', 'f', 'gamma', 'laplace', 'logistic', 'multivariate_normal', 'pareto', 'uniform', 'wald']
    same_funcs_pmf = ['multinomial', 'poisson', 'zipf']
    different_funcs_pmf = {'binomial':'binom', 'geometric':'geom', 'logseries':'logser',\
                           'multivariate_hypergeometric':'multivariate_hypergeom',\
                           'negative_binomial':'nbinom'}
    different_funcs = {'chisquare':'chi2', 'gumbel':'gumbel_r', 'noncentral_chisquare':'ncx2',\
                       'noncentral_f':'ncf', 'normal':'norm', 'power':'powerlaw', 'standard_cauchy':'cauchy',\
                       'standard_gamma':'gamma', 'standard_normal':'norm', 'weibull':'weibull_min'}
    if randname in same_funcs:            
        return get_scipy_pdf_helper(x,randname, args)
    elif randname in same_funcs_pmf:
        return get_scipy_pdf_helper(x, randname, args, pmf=True)
    elif randname in different_funcs:
        return get_scipy_pdf_helper(x,different_funcs[randname], args)
    elif randname in different_funcs_pmf:
        return get_scipy_pdf_helper(x,different_funcs_pmf[randname], args, pmf=True)       
    elif randname in ['exponential', 'rayleigh']:   
        if len(args)==0:            return getattr(stats, randname).pdf(x)
        elif len(args)==1:          return getattr(stats, randname).pdf(x, scale=args[0]) 
        elif len(args)==2:          return getattr(stats, randname).pdf(x, loc=args[1], scale=args[0]) 
        else: raise Exception("Too many arguments for "+randname+" distribution")
    elif randname=='hypergeometric': 
        n_pop = args[0]+args[1]
        n_good = args[0]
        n_sample = args[2]
        return stats.hypergeom.pmf(x,n_pop, n_good, n_sample)
    elif randname=='lognormal':
        s=args[1]
        scale=np.exp(args[0])
        return stats.lognormal.pdf(x, s, scale=scale) 
    elif randname=='standard_t': return stats.multivariate_t.pdf(x, df=args[0])
    elif randname=='triangular':
        left, mode, right = args[:3]
        loc = left
        scale = right-loc
        c = (mode-loc)/scale
        return stats.triang.pdf(x,c,loc,scale)
    elif randname=='vonmises':
        return stats.vonmises.pdf(x,args[1], args[0])
    else: raise Exception("Invalid randname distribution: "+randname+". Ensure that it is a part of numpy.random/scipy.stats")

class Timer():
    """class for model timers used in functions (e.g. for conditional faults) 
    Attributes
    ----------
    name : str
        timer name
    time : float
        internal timer clock time
    tstep : float
        time to increment at each time-step
    mode : str (standby/ticking/complete)
        the internal state of the timer
    """
    def __init__(self, name):
        """
        Initializes the Tymer

        Parameters
        ----------
        name : str
            Name for the timer
        """
        self.name=str(name)
        self.time=0.0
        self.tstep=-1.0
        self.mode='standby'
    def __repr__(self):
        return 'Timer '+self.name+': mode= '+self.mode+', time= '+str(self.time)
    def t(self):
        """ Returns the time elapsed """
        return self.time
    def inc(self, tstep=[]):
        """ Increments the time elapsed by tstep"""
        if self.time>=0.0:
            if tstep:   self.time+=tstep
            else:       self.time+=self.tstep
            self.mode='ticking'
        if self.time<=0: self.time=0.0; self.mode='complete'
    def reset(self):
        """ Resets the time to zero"""
        self.time=0.0
        self.mode='standby'
    def set_timer(self,time, tstep=-1.0, overwrite='always'):
        """ Sets timer to a given time
        
        Parameters
        ----------
        time : float
            set time to count down in the timer
        tstep : float (default -1.0)
            time to increment the timer at each time-step
        overwrite : str
            whether/how to overwrite the previous time
            'always' (default) sets the time to the given time
            'if_more' only overwrites the old time if the new time is greater
            'if_less' only overwrites the old time if the new time is less
            'never' doesn't overwrite an existing timer unless it has reached 0.0
            'increment' increments the previous time by the new time
        """
        if overwrite =='always':                        self.time=time
        elif overwrite=='if_more' and self.time<time:   self.time=time
        elif overwrite=='if_less' and self.time>time:   self.time=time
        elif overwrite=='never' and self.time==0.0:     self.time=time
        elif overwrite=='increment':                    self.time+=time
        self.tstep=tstep
        self.mode='set'
    def in_standby(self):
        """Whether the timer is in standby (time has not been set)"""
        return self.mode=='standby'
    def is_ticking(self):
        """Whether the timer is ticking (time is incrementing)"""
        return self.mode=='ticking'
    def is_complete(self):
        """Whether the timer is complete (after time is done incrementing)"""
        return self.mode=='complete'
    def is_set(self):
        """Whether the timer is set (before time increments)"""
        return self.mode=='set'

# def phases(times, names=[]):
#     """ Creates named phases from a set of times defining the edges of the intervals """
#     if not names: names = range(len(times)-1)
#     return {names[i]:[times[i], times[i+1]] for (i, _) in enumerate(times) if i < len(times)-1}
# def trunc(x, n=2.0, truncif='greater'):
#     """truncates a value to a given number (useful if behavior unchanged by increases)
    
#     Parameters
#     ----------
#     x : float/int 
#         number to truncate
#     n : float/int (optional)
#         number to truncate to if >= number
#     truncif: 'greater'/'less'
#         whether to truncate if greater or less than the given number
#     """
#     if truncif=='greater' and x>n:      y=n
#     elif  truncif=='greater' and x<n:   y=n
#     else:                               y=x
#     return y
# def union(probs):
#     """ Calculates the union of a list of probabilities [p_1, p_2, ... p_n] p = p_1 U p_2 U ... U p_n """
#     while len(probs)>1:
#         if len(probs) % 2: 
#             p, probs = probs[0], probs[1:]
#             probs[0]=probs[0]+p -probs[0]*p
#         probs = [probs[i-1]+probs[i]-probs[i-1]*probs[i] for i in range(1, len(probs), 2)]
#     return probs[0]










