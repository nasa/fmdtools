#!/usr/bin/env python
# coding: utf-8

# # Degradation Modelling 
# 
# This rover shows how degradation modelling can be performed to model the resilience of an engineered system over its entire lifecycle.

# In[1]:


import fmdtools.analyze as an
import fmdtools.sim.propagate as prop
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp


# In[2]:


from examples.rover.rover_degradation import DriveDegradation, PSFDegradationLong, PSFDegradationShort


# Degradation models are defined independently of the fault model, but have attributes (e.g., functions) which may correspond to it directly.
# 
# Because degradation may only occur in specific functions/flows (and may not have inter-functional dependencies), it is not necessary for the degradation model to have the same 

# In[3]:


deg_mdl = DriveDegradation()
deg_mdl


# In[4]:


deg_mdl_hum_long = PSFDegradationLong()
deg_mdl_hum_long


# In[5]:


deg_mdl_hum_short = PSFDegradationShort()
deg_mdl_hum_short


# In[6]:


from examples.rover.rover_model import Rover, plot_map
fault_mdl = Rover(p={'ground':{'linetype': 'turn'}})
graph = an.graph.FunctionArchitectureGraph(fault_mdl)
fig, ax = graph.draw()


# In[7]:


fig.savefig("func_model.pdf", format="pdf", bbox_inches = 'tight', pad_inches = 0)


# In[8]:


endresults, mdlhist = prop.nominal(fault_mdl)
fig, ax = plot_map(fault_mdl, mdlhist)


# In[9]:


fig.savefig("sine_rover_environment.pdf", format="pdf", bbox_inches = 'tight', pad_inches = 0)


# As shown, there are two degradation models here:
# - one which focusses solely on faults in the drive system, and
# - one which focusses on the human degradation of fatigue
# Below we simulate these to model to the degradation behaviors being modelled in this drive system.

# ## Drive Degradation

# In[10]:


deg_mdl = DriveDegradation()
endresults, mdlhist = prop.nominal(deg_mdl)
fig, ax = mdlhist.plot_line('s.wear', 's.corrosion', 's.friction', 's.drift', 'r.s.corrode_rate', 'r.s.wear_rate', 'r.s.yaw_load')


# The major behaviors are:
# - wear
# - corrosion
# - friction
# - drift
# 
# These behaviors result from the accumulation of the following rates over each time-step:
# - yaw_load 
# - corrode_rate
# - wear_rate
# 
# These degradation behaviors have additionally been defined to simulate stochastically if desired:

# In[11]:


deg_mdl = DriveDegradation()
endresults, mdlhist = prop.nominal(deg_mdl, run_stochastic=True)
fig, ax = mdlhist.plot_line('s.wear', 's.corrosion', 's.friction', 's.drift', 'r.s.corrode_rate', 'r.s.wear_rate', 'r.s.yaw_load')


# To get averages/percentages over a number of scenarios, we can view these behaviors over a given number of random seeds:

# In[12]:


from fmdtools.sim.sample import ParameterSample
ps = ParameterSample()
ps.add_variable_replicates([], replicates=100, seed_comb='independent')
endclasses_deg, mdlhists_deg = prop.parameter_sample(deg_mdl, ps, run_stochastic=True)


# In[13]:


fig, ax = mdlhists_deg.plot_line('s.wear', 's.corrosion', 's.friction', 's.drift',
                                 'r.s.corrode_rate', 'r.s.wear_rate', 'r.s.yaw_load',
                                 title="", xlabel='lifecycle time (months)', aggregation = 'mean_bound')


# In[14]:


fig.savefig("drive_degradations.pdf", format="pdf", bbox_inches = 'tight', pad_inches = 0)


# As shown, while wear and friction proceed monotonically, drift can go one way or another, meaning that whether the rover drifts left or right is basically up to chance. We can further look at slices of these distributions:

# In[15]:


fig, axs = mdlhists_deg.plot_metric_dist([1, 10, 20, 25], 's.wear', 's.corrosion', 's.friction', 's.drift', bins=10, alpha=0.5)


# Given the parameter information (friction and drift) that the degradation model produced, we can now simulate the model with this information over time in the nominal scenarios.

# In[16]:


from fmdtools.sim.sample import ParameterDomain, ParameterHistSample
from examples.rover.rover_model import RoverParam


# We can do this using a `ParameterHistSample` to sample the histories of the various scenarios at different times. 
# 
# First, by defining a `ParameterDomain`:

# In[17]:




# In[34]:


# fig.savefig("drive_resilience_degradation.pdf", format="pdf", bbox_inches = 'tight', pad_inches = 0)


# As shown, while there is some resilience early in the lifecycle (resulting in a small proportion of faults being recovered), this resilience goes away with degradation.

# ## Human Degradation

# We can also perform this assessment for the human error model, which is split up into two parts:
# - long term "degradation" of experience over months
# - short term "degradation" of stress and fatigue over a day

# In[35]:


psf_long = PSFDegradationLong()
endresults,  hist_psf_long = prop.nominal(psf_long)


# In[36]:


hist_psf_long.plot_line('s.experience')


# In[37]:


hist_psf_long.plot_line('s.experience')


# In[38]:


from examples.rover.rover_degradation import LongParams
pd_hl = ParameterDomain(LongParams)
pd_hl.add_variable("experience_param", var_lim=())
pd_hl


# In[39]:


pd_hl(10)


# In[40]:


ps_hl = ParameterSample(pd_hl)
xs = np.random.default_rng(seed=101).gamma(1,1.9,101)
# round so that dist is 0-10
xs = [min(x, 9.9) for x in xs]
weight = 1/len(xs)
for x in xs:
    ps_hl.add_variable_scenario(x, weight=weight)
ps_hl


# In[41]:


ec_psf_long, hist_psf_long= prop.parameter_sample(psf_long, ps_hl, run_stochastic=True)


# In[42]:


hist_psf_long.plot_line('s.experience')


# In[43]:


fig, axs = hist_psf_long.plot_metric_dist([0, 40, 50, 60, 100], 's.experience', bins=20, alpha=0.5, figsize=(8,4))


# In[44]:


# fig.savefig("experience_degradation.pdf", format="pdf", bbox_inches = 'tight', pad_inches = 0)


# Short-term degradation

# In[45]:


psf_short = PSFDegradationShort()
er, hist_short = prop.nominal(psf_short)
fig, axs = hist_short.plot_line('s.fatigue', 's.stress')


# short-term degradation (over no external params)

# In[46]:


ps_psf_short = ParameterSample()
ps_psf_short.add_variable_replicates([], replicates=25)
ps_psf_short.scenarios()


# In[47]:


ec_psf_short, hist_psf_short = prop.parameter_sample(psf_short, ps_psf_short, run_stochastic=True)
fig, axs = hist_psf_short.plot_line('s.fatigue', 's.stress', 'r.s.fatigue_param', aggregation="percentile")


# short-term degradation over long-term params

# In[48]:


from examples.rover.rover_degradation import PSFShortParams
pd_short_long = ParameterDomain(PSFShortParams)
pd_short_long.add_variable("experience")
ps_short_long = ParameterHistSample(hist_psf_long, 's.experience', paramdomain=pd_short_long)
ps_short_long.add_hist_groups(reps= 10, ts = [0, 40, 50, 60, 100])

# note - need to add a way to combine replicates (seeds) over replicates (input times/groups/)


# In[49]:


pd_short_long(10)


# In[50]:


ps_short_long.scenarios()


# In[51]:


ec, hist_short_long = prop.parameter_sample(psf_short, ps_short_long, run_stochastic=True)


# In[52]:


fig, axs = hist_short_long.plot_line('s.fatigue', 's.stress', 'r.s.fatigue_param', aggregation="percentile")


# In[53]:


# fig.savefig("stress_degradation.pdf", format="pdf", bbox_inches = 'tight', pad_inches = 0)


# Now sample in the model:

# In[54]:


from examples.rover.rover_model_human import RoverHuman, RoverHumanParam
pd_comb_mdl = ParameterDomain(RoverHumanParam)
# pd_comb_mdl.add_constant('drive_modes', {"mode_args": "degradation"})
pd_comb_mdl.add_variable("psfs.fatigue")
pd_comb_mdl.add_variables("psfs.stress")
pd_comb_mdl(1,1)


# In[55]:


ps_comb_mdl = ParameterHistSample(hist_short_long, "s.fatigue", "s.stress", paramdomain=pd_comb_mdl)
ps_comb_mdl.add_hist_groups(reps= 10, ts = [0, 1, 3, 5, 8])
ps_comb_mdl.scenarios()


# In[57]:


mdl_hum = RoverHuman(p={'ground': {'linetype': 'turn'}})
# mdl_hum = RoverHuman()


# In[67]:


ec, hist = prop.nominal(mdl_hum)
plot_map(mdl_hum, hist)
ec


# In[73]:


from fmdtools.analyze.result import Result
n_ec = Result.fromkeys(ps_comb_mdl.scen_names())


# In[77]:


n_ec['hist_0']= 1


# In[78]:


n_ec


# In[71]:


# In[68]:


ec_comb, hist_comb = prop.parameter_sample(mdl_hum, ps_comb_mdl)


# In[69]:


fig, ax = plot_map(mdl_hum, hist_comb)


# In[70]:


ec_comb


# In[ ]:
from fmdtools.analyze.tabulate import NominalEnvelope

ne = NominalEnvelope(ps_comb_mdl, ec_comb, 'at_finish',
                     'p.psfs.fatigue', 'p.psfs.stress')
ne.plot_scatter()


# In[ ]:


ne.variable_groups
hist_comb.plot_line('flows.psfs.s.attention', 'flows.motor_control.s.rpower', 'flows.motor_control.s.lpower')

