# -*- coding: utf-8 -*-
"""
Description: A module for defining time-based properties for use in blocks. Has Classes:
    
- :class:`Timer`: Class defining timers
- :class:`Time`: Class containing all time-related Block constructs (e.g., timers).
"""
from decimal import Decimal
from recordclass import dataobject
from fmdtools.analyze.result import History, init_hist_iter, get_sub_include
from .common import  get_dataobj_track, get_obj_track

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
    default_track = ('time', 'mode')
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
    def copy(self):
        cop = self.__class__(self.name)
        cop.time=self.time
        cop.mode=self.mode
        cop.dt = self.dt
        return cop
    def create_hist(self, timerange, track):
        h = History()
        track = get_obj_track(self, track, all_possible=('time', 'mode'))
        h.init_att('time', self.time, timerange=timerange, track=track, dtype=float)
        h.init_att('mode', self.mode, timerange=timerange, track=track, str_size='<U8')
        return h

class Time(dataobject):
    """
    Class for defining all time-based aspects of a Block (e.g., time, timestep, timers). 
    
    Attributes
    ----------
    time : float
        real time for the model
    dt : float
        timestep size
    t_ind : int
        index of the given time
    t_loc : float
        local time (e.g., for actions with durations)
    run_times : int
        number of times to run the behavior if running at a different timestep than global
    timers : dict
        dictionary of instantiated timers
    use_local : bool
        Whether to use the local timetep (vs global timestep)
    timernames: tuple
        Names of timers to instantiate.
    """
    time:       float=0.0
    dt:         float=1.0
    t_ind:      int=0
    t_loc:      float=0.0
    run_times:  int=1
    timers:     dict = {}
    use_local:  bool=True
    timernames = ()
    default_track = ('timers')
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.timers: self.timers = {}
        for timername in self.timernames:
            self.timers[timername]=Timer(timername)
        self.set_timestep()
    def __getattr__(self, item):
        if item in self.timers: return self.timers[item]
        else:                   return super().__getattribute__(item)
    def return_mutables(self):
        return (t.time for t in self.timers.values())
    def set_timestep(self):
        """Sets the timestep of the function given the option use_local 
        (which selects whether it uses local_timestep or global_timestep)"""
        global_tstep = Decimal(str(self.dt))
        local_tstep = Decimal(str(self.__defaults__[self.__fields__.index('dt')]))
        if self.use_local:
            dt=local_tstep
            if dt < global_tstep:
                if global_tstep%dt:
                    raise Exception("Local timestep: "+str(dt)+" doesn't line up with global timestep: "+str(global_tstep))
            else:
                if dt%global_tstep:
                    raise Exception("Local timestep: "+str(dt)+" doesn't line up with global timestep: "+str(global_tstep))
            self.run_times = int(global_tstep/dt)
        else:   
            dt=global_tstep
            self.run_times=1
        self.dt = float(dt)
        for timer in self.timers.values():
            timer.dt=-self.dt
    def reset(self):
        """Resets time to the initial state"""
        self.time=0.0
        self.t_ind=0
        self.t_loc=0.0
        for timer in self.timers.values():
            timer.reset()
    def copy(self, *args, **t_args):
        """ Copies the timer"""
        cop = self.__class__(*args, **t_args)
        for timer in self.timers:
            cop.timers[timer] = self.timers[timer].copy()
        cop.run_times=self.run_times
        cop.t_ind=self.t_ind
        cop.t_loc=self.t_loc
        cop.dt=self.dt
        return cop
    def create_hist(self, timerange, track):
        """
        Creates a History corresponding to Time

        Parameters
        ----------
        timerange : iterable, optional
            Time-range to initialize the history over. The default is None.
        track : list/str/dict, optional
            argument specifying attributes for :func:`get_sub_include'. The default is None.
                DESCRIPTION. The default is None.

        Returns
        -------
        hist : History
            History of time/timer attribues specified in track.
        """
        hist = History()
        track = get_dataobj_track(self, track)
        hist.init_att('time', self.time, timerange=timerange, track=track, dtype=float)
        if 'timers' in track:
            track_timers = get_sub_include('timers', track)
            for tname, timer in self.timers.items():
                sub_track = get_sub_include(tname, track_timers)
                hist[tname] = timer.create_hist(timerange, sub_track)
        return hist