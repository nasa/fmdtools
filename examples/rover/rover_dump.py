# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 15:48:52 2023

@author: dhulse
"""
from rover_model import Rover, RoverParam
import fmdtools.sim.propagate as prop
from fmdtools.sim.approach import SampleApproach

"""
mdl = Rover(p=gen_p('turn'))
#dot = an.graph.show(mdl, gtype="fxnflowgraph", renderer='graphviz')
#p = gen_p('sine')
#mdl = Rover(p)
endresults,  mdlhist = prop.nominal(mdl)
phases, modephases = mdlhist.get_modephases()
plot_map(mdl, mdlhist)

mdl_id = Rover(valp={'drive_modes':'else'})
app_id = SampleApproach(mdl_id, faults='drive', phases={'drive':phases['avionics']['drive']})
endclasses_id, mdlhists_id = prop.approach(mdl_id, app_id, staged=True)


p = gen_p('sine',cor_d=-180, cor_f=1)
mdl_thing = Rover(p=p)
_,_, reshist = prop.one_fault(mdl,'drive','stuck_right', time=15, staged=True)
plt.figure()
f = plot_trajectories({'nominal':mdlhist}, reshist,  faultalpha=0.6)

an.plot.mdlhistvals(reshist, time=15, fxnflowvals={'drive':['friction','drift', 'transfer'], 'power':'all'})

app_opt = SampleApproach(mdl, faults='drive', phases={'drive':phases['avionics']['drive']}, defaultsamp={'samp':'evenspacing','numpts':4})

#endresults,  mdlhist = prop.one_fault(mdl, 'drive','elec_open', time=1, staged=False)
#an.plot.mdlhistvals(mdlhist, fxnflowvals={'drive':['friction','drift', 'transfer']})

#endresults,  mdlhist = prop.one_fault(mdl, 'drive','hmode_34', time=1, staged=False)
#an.plot.mdlhistvals(mdlhist, fxnflowvals={'drive':['friction','drift']})


x = [100,0,2,0,2,-2,0,0,0]
x_p = gen_p('sine', ub_f=x[0], lb_f=x[1], ub_t=x[2],lb_t=x[3], ub_d=x[4], lb_d=x[5], cor_f=x[6], cor_d=x[7], cor_t=x[8])
mdl_0 = Rover(p=x_p)


_,_, nomhist = prop.nominal(mdl_0)
phases, modephases = mdlhist.get_modephases()
app_0 = SampleApproach(mdl, faults='drive', phases={'drive':phases['avionics']['drive']}, defaultsamp={'samp':'evenspacing','numpts':4})
endclasses_0, mdlhists_0 = prop.approach(mdl_0, app_0, staged=True)
plt.figure()
f = plot_trajectories(mdlhists_0, app=app_0,  faultalpha=0.6)


x = [100,0,2,0,2,-2,1,-180,1]
x_p = gen_p('sine', ub_f=x[0], lb_f=x[1], ub_t=x[2],lb_t=x[3], ub_d=x[4], lb_d=x[5], cor_f=x[6], cor_d=x[7], cor_t=x[8])
mdl_1 = Rover(p=x_p)
endclasses_1, mdlhists_1 = prop.approach(mdl_1, app_0, staged=True)

#compare_trajectories(mdlhists_0, mdlhists_1, mdlhist1_name='fault trajectories', mdlhist2_name='comparison trajectories', faulttimes = app.times, nomhist=nomhist)


#an.graph.show(  scale=0.7)

#an.plot.mdlhistvals(mdlhist, legend=False)
#an.plot.mdlhistvals(mdlhist)

plot_map(mdl, mdlhist)

#endresults,  mdlhist = prop.one_fault(mdl, 'drive','elec_open', staged=True, time=13, gtype='typegraph')
endresults,  mdlhist_feed = prop.one_fault(mdl, 'perception', 'bad_feed', staged=True, time=7, gtype='typegraph')
plot_trajectories(mdlhist_feed, mdlhist, faultalpha=1.0)


#an.plot.mdlhistvals(mdlhist, fxnflowvals={'power':['charge','power']}, time=7, phases=phases, modephases=modephases)
#an.plot.mdlhistvals(mdlhist, fxnflowvals={'ground':['x','y', 'angle','vel', 'liney', 'ang']}, time=7, phases=phases)
#an.plot.mdlhistvals(mdlhist, fxnflowvals={'pos_signal':['x','y', 'angle','vel', 'heading']}, time=7, phases=phases)
#an.plot.mdlhistvals(mdlhist, fxnflowvals={'motor_control':['rpower','lpower']}, time=7, phases=phases)
an.plot.mdlhistvals(mdlhist, fxnflowvals={'avionics':['mode']}, time = 13, phases=phases, modephases=modephases)
an.plot.mdlhistvals(mdlhist, fxnflowvals={'perception':['mode']}, time = 13, phases=phases, modephases=modephases)
#an.plot.mdlhistvals(mdlhist, fxnflowvals={}, time = 7, phases=phases, modephases=modephases)
app = NominalApproach()
app.add_param_ranges(gen_p,'sine', 'sine', amp=(0, 10, 0.2), wavelength=(10,50,10))
app.assoc_probs('sine', amp=(stats.uniform.pdf, {'loc':0,'scale':10}), wavelength=(stats.uniform.pdf,{'loc':10, 'scale':40}))
#app.add_param_ranges(gen_p,'turn', radius=(5,40,5), start=(0, 20,5))

#labels, faultfxns, degnodes, faultlabels
#    an.graph.plot_bipgraph(classgraph, {node:node for node in classgraph.nodes},[],[],{}, pos=pos)
an.graph.show( gtype='typegraph', scale=0.7)

#endresults,  mdlhist = prop.one_fault(mdl, 'drive','elec_open', staged=True, time=13, gtype='fxnflowgraph')
endresults,  mdlhist = prop.one_fault(mdl, 'perception', 'bad_feed', staged=True, time=13, gtype='fxnflowgraph')
an.graph.show( gtype='fxnflowgraph', scale=0.7)

reshist, _, _ = an.process.hist(mdlhist)
typehist = an.process.typehist(mdl, reshist)
an.graph.results_from(mdl, reshist, [10,15,20])
an.graph.results_from(mdl, typehist, [10,15,20], gtype='typegraph') #), gtype='typegraph')
an.graph.result_from(mdl, reshist, 10, gtype='fxnflowgraph', renderer='graphviz')

#endclasses, mdlhists= prop.nominal_approach(mdl, app, pool = mp.Pool(5))

#fig = an.plot.nominal_vals_1d(app, endclasses, 'amp')
#fig = an.plot.nominal_vals_1d(app, endclasses, 'radius')

#app = NominalApproach()
#app.add_param_ranges(gen_p,'sine','sine', amp=(0, 10, 0.2), wavelength=(10,50,10), dummy=(1,10,1))

#endclasses, mdlhists= prop.nominal_approach(mdl, app, pool = mp.Pool(5))
#fig = an.plot.nominal_vals_3d(app, endclasses, 'amp', 'wavelength', 'dummy')
#app = SampleApproach(mdl, phases = phases, modephases = modephases)

#endclasses, mdlhist = prop.approach(mdl, app)

#app_joint = SampleApproach(mdl, phases = phases, modephases = modephases, jointfaults={'faults':2})

#endclasses, mdlhist = prop.approach(mdl, app_joint)


#tab = an.tabulate.phasefmea(endclasses, app_joint)
#an.plot.samplecosts(app_joint, endclasses)

#an.plot.phases(phases)

#figs = an.plot.phases(phases, modephases, mdl)
#figs = an.plot.phases(phases, modephases, mdl, singleplot=False)
"""
