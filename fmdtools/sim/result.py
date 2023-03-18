# -*- coding: utf-8 -*-
"""
Description: A module defining how simulation results (histories) structured and processed. Has classes:

- :class:`Result`:  Class for defining result dictionaries (nested dictionaries of metric(s))
- :class:`History`: Class for defining simulation histories (nested dictionaries of arrays or lists)

And functions:
- :func:`load`:             Loads a given file to a Result/History
- :func:`load_folder`:      Loads a given folder to a Result/History
"""

from collections import UserDict

import numpy as np
import copy
import sys, os


def file_check(filename, overwrite):
    """Check if files exists and whether to overwrite the file"""
    if os.path.exists(filename):
        if not overwrite: raise Exception("File already exists: "+filename)
        else:                   
            print("File already exists: "+filename+", writing anyway...")
            os.remove(filename)
    if "/" in filename:
        last_split_index = filename.rfind("/")
        foldername = filename[:last_split_index]
        if not os.path.exists(foldername): os.makedirs(foldername)
def auto_filetype(filename, filetype=""):
    """Helper function that automatically determines the filetype (pickle, csv, or json) of a given filename"""
    if not filetype:
        if '.' not in filename: raise Exception("No file extension")
        if filename[-4:]=='.pkl':       filetype="pickle"
        elif filename[-4:]=='.csv':     filetype="csv"     
        elif filename[-5:]=='.json':    filetype="json"
        else: raise Exception("Invalid File Type in: "+filename+", ensure extension is pkl, csv, or json ")
    return filetype
def create_indiv_filename(filename, indiv_id, splitchar='_'):
    """Helper file that creates an individualized name for a file given the general filename and an individual id"""
    filename_parts = filename.split(".")
    filename_parts.insert(1,'.')
    filename_parts.insert(1,splitchar+indiv_id)   
    return "".join(filename_parts)
def clean_resultdict_keys(resultdict_dirty):
    """
    Helper function for recreating results dictionary keys (tuples) from a dictionary loaded from a file (where keys are strings)
    (used in csv/json results)

    Parameters
    ----------
    resultdict_dirty : dict
        Results dictionary where keys are strings

    Returns
    -------
    resultdict : dict
        Results dictionary where keys are tuples
    """
    resultdict = {}
    for key in resultdict_dirty:
        newkey = tuple(key.replace("'","").replace("(","").replace(")","").split(", "))
        if any(['t=' in strs for strs in  newkey]):
            joinfirst = [ind for ind, strs in enumerate(newkey) if 't=' in strs][0] +1
        else:                           joinfirst=0
        if joinfirst==2:
            newkey = tuple([", ".join(newkey[:2])])+newkey[2:]
        elif joinfirst>2:
            nomscen = newkey[:joinfirst-2]
            faultscen = tuple([", ".join(newkey[joinfirst-2:joinfirst])])
            vals = newkey[joinfirst:]
            newkey = nomscen+faultscen+vals
        resultdict[newkey]=resultdict_dirty[key]
    return resultdict
def load(filename, filetype="", renest_dict=True, indiv=False):
    """
    Loads a given (endclasses or mdlhists) results dictionary from a (pickle/csv/json) file.
    e.g. a file saved using process.save_result or save_args in propagate functions.

    Parameters
    ----------
    filename : str
        Name of the file.
    filetype : str, optional
        Use to specify a filetype for the file (if not included in the filename). The default is "".
    renest_dict : bool, optional
        Whether to return . The default is True.
    indiv : bool, optional
        Whether the result is an individual file (e.g. in a folder of results for a given simulation). 
        The default is False.

    Returns
    -------
    result : Result
        Corresponding result/hist object with data loaded from the file.
    """
    import dill, json, csv, pandas
    if not os.path.exists(filename): raise Exception("File does not exist: "+filename)
    filetype = auto_filetype(filename, filetype)
    if filetype=='pickle':
        with open(filename, 'rb') as file_handle:
            resultdict = dill.load(file_handle)
        file_handle.close()
    elif filetype=='csv': # add support for nested dict mdlhist using flatten_hist?
        if indiv:   resulttab = pandas.read_csv(filename, skiprows=1)
        else:           resulttab = pandas.read_csv(filename)
        resultdict = resulttab.to_dict("list")
        resultdict = clean_resultdict_keys(resultdict)
        for key in resultdict:
            if len(resultdict[key])==1 and isinstance(resultdict[key], list):
                resultdict[key] = resultdict[key][0]
            else: resultdict[key] = np.array(resultdict[key])             
        if renest_dict: resultdict = History.fromdict(resultdict)
        if indiv: 
            scenname = [*pandas.read_csv(filename, nrows=0).columns][0]
            resultdict = {scenname: resultdict}
    elif filetype=='json':
        with open(filename, 'r', encoding='utf8') as file_handle:
            loadeddict = json.load(file_handle)
            if indiv:   
                key = [*loadeddict.keys()][0]
                loadeddict = loadeddict[key]
                loadeddict= {key+", "+innerkey:values for innerkey, values in loadeddict.items()}
                resultdict = clean_resultdict_keys(loadeddict)
            else:       resultdict = clean_resultdict_keys(loadeddict)
            
            if renest_dict: resultdict = History.fromdict(resultdict)
        file_handle.close()
    else:
        raise Exception("Invalid File Type")
    return resultdict
def load_folder(folder, filetype, renest_dict=True):
    """
    Loads endclass/mdlhist results from a given folder 
    (e.g., that have been saved from multi-scenario propagate methods with 'indiv':True)

    Parameters
    ----------
    folder : str
        Name of the folder. Must be in the current directory
    filetype : str
        Type of files in the folder ('pickle', 'csv', or 'json')
    renest_dict : bool, optional
        Whether to return result as a nested dict (as opposed to a flattenned dict). 
        The default is True.

    Returns
    -------
    resultdict : dict
        endclasses/mdlhists result dictionary reconstructed by the files in the folder.
    """
    files = os.listdir(folder)
    files_toread = []
    for file in files:
        read_filetype = auto_filetype(file)
        if read_filetype==filetype:
            files_toread.append(file)
    return files_toread
def get_dict_attr(dict_in, des_class, *attr):
    """Gets attributes *attr from a given dict dict_in of class des_class"""
    if len(attr)==1:    return dict_in[attr[0]]
    else:               return get_dict_attr(des_class(dict_in[attr[0]]), *attr[1:])
def fromdict(resultclass, inputdict):
    """Creates new history/result from given dictionary"""
    newhist = History()
    for k, val in inputdict.items():
        if isinstance(val, dict):   newhist[k]=resultclass.fromdict(val)
        else:                       newhist[k]=val
        return newhist
def check_include_errors(result, to_include):
    if type(to_include)!=str:
        for k in to_include:
            check_include_error(result,k)
    else:
        check_include_error(result,to_include)
def check_include_error(result,to_include):
    if to_include!='all' and to_include not in result:
        raise Exception("to_include key "+to_include+" not in result keys: "+str(result.keys()))

class Result(UserDict):
    """
    Result is a special type of dictionary that makes it convenient to store, access,
    and load results form a model/simulation. 
    
    As a dictionary, it supports dict-based item assignement (e.g. r['x']=10) but
    also enables convenient access via __getattr__, e.g.:
        r[x] = 10
        r.x
        > 10
    
    It also can return a flattened version of its nested sturcture via Result.flatten(), e.g.
        r = Result(y=Result(z=1))
        rf = r.flatten()
        r[('y','z')]
        > 1
    
    It also enables saving and loading to files via r.save(), r.load(), and r.load_folder()
    """
    def __repr__(self, ind=0):
        str_rep = ""
        for k, val in self.items():
            if isinstance(val, np.ndarray) or isinstance(val, list):
                if type(k)==tuple: k = str(k)
                val_rep = ind*"--"+k+": "
                if len(val_rep)>40: val_rep = val_rep[:20]
                form = '{:>'+str(40-len(val_rep))+'}'
                vv = form.format("array("+str(len(val))+")")
                str_rep = str_rep+val_rep+vv+'\n'
            elif isinstance(val, Result):
                res = val.__repr__(ind+1)
                if res:
                    val_rep = ind*"--"+k+": \n"+res
                    str_rep = str_rep+val_rep
            else: 
                val_rep = ind*"--"+k+": "
                if len(val_rep)>40: val_rep = val_rep[:20]
                form = '{:>'+str(40-len(val_rep))+'}'
                vv = form.format(str(val))
                str_rep = str_rep+val_rep+vv+'\n'
        return str_rep
    def all(self):
        return tuple(self.data.values())
    def __eq__(self, other):
        return all([all(v==other[k]) if isinstance(v, np.ndarray) else v==other[k] for k,v in self.data.items()])
    def keys(self):
        return self.data.keys()
    def items(self):
        return self.data.items()
    def values(self):
        return self.data.values()
    def __reduce__(self):
        return (type(self), (), None, None, iter(self.items()))
    def __getattr__(self, argstr):
        args = argstr.split(".")
        try:
            return get_dict_attr(self.data, self.__class__, *args)
        except:
            raise AttributeError("Not in dict: "+str(argstr))
    def __setattr__(self, key, val):
        if key == "data":
            UserDict.__setattr__(self, key, val)
        else:
            self.data[key]=val
    def fromdict(inputdict):
        return fromdict(Result, inputdict)
    def load(filename, filetype="", renest_dict=True, indiv=False):
        """Loads as Result using :func:`load'"""
        inputdict = load(filename, filetype="", renest_dict=True, indiv=False)
        return fromdict(Result, inputdict)
    def load_folder(folder, filetype, renest_dict=True):
        """Loads as History using :func:`load_folder'"""
        files_toread = load_folder(folder, filetype, renest_dict=True)
        result = Result()
        for filename in files_toread:
            result.update(Result.load_folder(folder+'/'+filename, filetype, renest_dict=renest_dict, indiv=True))
        return result
    def flatten(self, newhist=False, prevname=(), to_include='all'):
        """
        Recursively creates a flattened result of the given nested model history

        Parameters
        ----------
        prevname : tuple, optional
            Current key of the flattened history (used when called recursively). The default is ().
        to_include : str/list/dict, optional
            What attributes to include in the dict. The default is 'all'. Can be of form
            - list e.g. ['att1', 'att2', 'att3'] to include the given attributes
            - dict e.g. fxnflowvals {'flow1':['att1', 'att2'], 'fxn1':'all', 'fxn2':['comp1':all, 'comp2':['att1']]}
            - str e.g. 'att1' for attribute 1 or 'all' for all attributes
        Returns
        -------
        newhist : dict
            Flattened model history of form: {(fxnflow, ..., attname):array(att)}
        """
        if newhist is False: 
            newhist = self.__class__()
        #TODO: Add some error handling for when the attributes in "to_include" aren't actually in hist
        check_include_errors(self, to_include)
        for att, val in self.items():
            newname = prevname+tuple([att])
            if isinstance(val, Result): 
                new_to_include = get_sub_include(att, to_include)
                if new_to_include: 
                    val.flatten(newhist, newname, new_to_include)
            elif to_include=='all' or att in to_include: 
                if len(newname)==1: newhist[newname[0]] = val
                else:               newhist[newname] = val
        return newhist
    def nest(self):
        """
        Re-nests a flattened result   
        """
        newhist = self.__class__()
        key_options = set([h[0] for h in self.keys()])
        for key in key_options:
            if (key,) in self:     newhist[key] = self[(key,)]
            else:
                subhist = self.__class__(**{histkey[1:]:val for histkey, val in self.items() if key==histkey[0]})                       
                newhist[key] = subhist.nest()
        return newhist
    def get_memory(self):
        """
        Determines the memory usage of a given history and profiles by 
    
        Returns
        -------
        mem_total : int
            Total memory usage of the history (in bytes)
        mem_profile : dict
            Memory usage of each construct of the model history (in bytes)
        """
        fhist = self.flatten()
        mem_total = 0
        mem_profile = dict.fromkeys(fhist.keys())
        for k,h in fhist.items():
            if np.issubdtype(h.dtype, np.number) or np.issubdtype(h.dtype, np.flexible) or np.issubdtype(h.dtype, np.bool_):
                mem=h.nbytes
            else:
                mem=0
                for entry in h:
                    mem+=sys.getsizeof(entry)
            mem_total+=mem
            mem_profile[k]=mem
        return mem_total, mem_profile
    def save(self, filename, filetype="", overwrite=False, result_id=''):
        """
        Saves a given result variable (endclasses or mdlhists) to a file filename. 
        Files can be saved as pkl, csv, or json.
    
        Parameters
        ----------
        filename : str
            File name for the file. Can be nested in a folder if desired.
        filetype : str, optional
            Optional specifier of file type (if not included in filename). The default is "".
        overwrite : bool, optional
            Whether to overwrite existing files with this name. The default is False.
        result_id : str, optional
            For individual results saving. Places an identifier for the result in the file. The default is ''.
        """
        import dill, json, csv
        file_check(filename, overwrite)
        
        variable = self
        filetype = auto_filetype(filename, filetype)
        if filetype=='pickle':
            with open(filename, 'wb') as file_handle:
                if result_id: variable = {result_id:variable}
                dill.dump(variable, file_handle)
        elif filename[-4:]=='.csv': # add support for nested dict mdlhist using flatten_hist?
            variable = variable.flatten()
            with open(filename, 'w', newline='') as file_handle:
                writer = csv.writer(file_handle)
                if result_id: writer.writerow([result_id])
                writer.writerow(variable.keys())
                if isinstance([*variable.values()][0], np.ndarray):
                    writer.writerows(zip(*variable.values()))
                else:
                    writer.writerow([*variable.values()])
        elif filename[-5:]=='.json':
            with open(filename, 'w', encoding='utf8') as file_handle:
                variable = variable.flatten()
                new_variable = {}
                for key in variable:
                    if isinstance(variable[key], np.ndarray):
                        new_variable[str(key)] =  [var.item() for var in variable[key]]
                    else:
                        new_variable[str(key)] =  variable[key]
                if result_id: new_variable = {result_id:new_variable}
                strs = json.dumps(new_variable, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)
                file_handle.write(str(strs))
        else:
            raise Exception("Invalid File Type")
        file_handle.close()

def is_known_immutable(val):
    return type(val) in [int, float, str, tuple, bool] or isinstance(val, np.number)


def get_sub_include(att, to_include):
    """Determines what attributes of att to include based on the provided dict/str/list/set to_include"""
    if type(to_include) in [list, set, tuple, str]:
        if att in to_include:                               new_to_include = 'all'
        elif type(to_include)==str and to_include=='all':   new_to_include='all'
        else:                                               new_to_include=False
    elif type(to_include)==dict and att in to_include:      new_to_include = to_include[att]
    else:                                                   new_to_include= False
    return new_to_include

def init_hist_iter(att, val, timerange=None, track=None, dtype=None, str_size='<U20'):
    """
    Initializes the history for a given attribute att with value val. Enables
    the recursive definition of a history as a nested structure.
    
    If a timerange is given, the base fields are initializes as fixed-length numpy
    arrays corresponding to the data type of the field. Otherwise, an emty list 
    is initialized.

    Parameters
    ----------
    att : str
        Name of the attribute.
    val : dict/field
        dict to be initialized as a History or field to be initialized as a list or numpy array
    timerange : iterable, optional
        Time-range to initialize the history over. The default is None.
    track : list/str/dict, optional
        argument specifying attributes for :func:`get_sub_include'. The default is None.
    dtype : str, optional
        Datatype to initialze the array as (if given). The default is None.
    str_size : str, optional
        Data type for strings. The default is '<U20'.
    
    Returns
    ---------
    Hist : History, List, or np.array
        Initialized history structure corresponding to the attribute
    """
    sub_track = get_sub_include(att, track)
    if sub_track and hasattr(val, 'create_hist'): return val.create_hist(timerange, sub_track)
    elif sub_track and isinstance(val, dict):     return init_dicthist(val, timerange, sub_track)
    elif sub_track:
        if timerange is None:                     return [val]
        elif dtype:                               return np.empty([len(timerange)], dtype=dtype)
        elif type(val)==str:                      return np.empty([len(timerange)], dtype=str_size)
        else:
            try:                                  return np.full(len(timerange), val)
            except:                               return np.empty((len(timerange),), dtype=object)
def init_dicthist(start_dict, timerange, track="all"):
    """
    Initializes histories for dictionary attributes (if any)

    Parameters
    ----------
    start_dict : dict
        Dictionary to initialize.
    timerange : iterable
        Timerange to initalize the hist over
    track : TYPE, optional
        DESCRIPTION. The default is "all".
    Returns
    -------
    Hist : History
        Initialized history structure corresponding to the attribute

    """
    hist = History()
    for att, val in start_dict.items():
        hist[att]=init_hist_iter(att,val, timerange, track)
    return hist


class History(Result):
    """ 
    History is a special time of :class:'Result' specifically for keeping simulation
    histories (e.g., a log of states over time).
    
    It can be updated over time t using h.log(obj, t), where obj is an object with
    (nested) attributes that match the keys of the (nested) dictionary.
    """
    def fromdict(inputdict):
        return fromdict(History, inputdict)
    def load(filename, filetype="", renest_dict=True, indiv=False):
        """Loads as History using :func:`load'"""
        inputdict = load(filename, filetype=filetype, renest_dict=renest_dict, indiv=indiv)
        return fromdict(History, inputdict)
    def load_folder(folder, filetype, renest_dict=True):
        """Loads as History using :func:`load_folder'"""
        files_toread = load_folder
        hist = History()
        for filename in files_toread:
            hist.update(History.load_folder(folder+'/'+filename, filetype, renest_dict=renest_dict, indiv=True))
        return hist
    def copy(self):
        """Creates a new independent copy of the current history dict"""
        newhist =History()
        for k, v in self.items():
            if isinstance(v, History):  newhist[k]=v.copy()
            else:                       newhist[k]=np.copy(v)
        return newhist
    def init_time(self,time, timerange):
        """
        Adds time to the history. Used at the top level of the history after 
        initialization to log timesteps.

        Parameters
        ----------
        time : float
            time to initialize the initial time at.
        timerange : iterable
            Timerange to initialize the history over.
        """
        if not hasattr(self, 'time'):
            self['time'] = init_hist_iter('time', time, timerange=timerange, track='all', dtype=float)
    def log(self, obj, t_ind, time=None):
        """
        Updates the history from obj at the time t_ind

        Parameters
        ----------
        obj : Model/Function/State...
            Object to log
        t_ind : int
            Time-index of the log.
        time : float
            Real time for the history (if initialized). Used at the top level of the history.
        """
        for att, hist in self.items():
            if att=='time' and time is not None:
                val=time
            else:
                try:    val=obj[att]
                except: 
                    try: val=getattr(obj, att)
                    except: val = att in obj
            
            if type(hist)==History:             hist.log(val, t_ind)
            else:
                if not is_known_immutable(val): val=copy.deepcopy(val)
                if type(hist)==list:              hist.append(val)
                elif isinstance(hist, np.ndarray):  
                    if t_ind >= len(hist):
                        raise Exception("Time beyond range of model history--check staged execution and simulation time settings (end condition, mdl.modelparams.times)")
                    if not np.can_cast(type(val), type(hist[t_ind])):
                        raise Exception(str(att)+" changed type: "+str(type(hist[t_ind]))+" to "+str(type(val))+" at t_ind="+str(t_ind))
                    try:
                        hist[t_ind]=val
                    except Exception as e:
                        raise Exception("Value too large to represent: "+att+"="+str(val)) from e
    def cut(self, end_ind=None, start_ind=None, newcopy=False):
        """Cuts the history to a given index"""
        if newcopy: hist = self.copy()
        else:       hist = self
        for name, att in hist.items():
            if isinstance(att, History): hist[name]=hist[name].cut(end_ind, start_ind, newcopy=False)
            else:       
                if end_ind is None:     hist[name]=att[start_ind:]  
                elif start_ind is None: hist[name]=att[:end_ind+1]  
                else:                   hist[name]=att[start_ind:end_ind+1] 
                    
        return hist 
    def get_slice(self,t_ind=0):
        """
        Returns a dictionary of values from (flattenned) version of the history at t_ind
        """
        flathist = self.flatten()
        slice_dict = dict.fromkeys(flathist)
        for key, arr in flathist.items():
            slice_dict[key]=flathist[key][t_ind]
        return slice_dict