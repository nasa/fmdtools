# -*- coding: utf-8 -*-
"""
pickle tests
Created on Wed Mar 24 16:19:59 2021

@author: dhulse
"""

from ex_pump import * 
import fmdtools.faultsim.propagate as propagate
import fmdtools.resultdisp as rd

IE = ImportEE([{'current':1.0, 'voltage':1.0}])
mdl = Pump()


#mdl.flows["EE_1"]._attributes

check_pickleability(IE)

check_pickleability(mdl)
check_model_pickleability(mdl)

pickle.dump( IE, open( "fxn_save.p", "wb" ) )
IE_loaded = pickle.load( open( "fxn_save.p", "rb" ) )

pickle.dump( mdl, open( "mdl_save.p", "wb" ) )
mdl_loaded = pickle.load( open( "mdl_save.p", "rb" ) )


## Compare simulation results
endresults, resgraph, mdlhist=propagate.nominal(mdl, track=True)
endresults_loaded, resgraph_loaded, mdlhist_loaded=propagate.nominal(mdl_loaded, track=True)
#plot graph
rd.graph.show(resgraph)
rd.graph.show(resgraph_loaded)
#plot the flows over time
rd.plot.mdlhistvals(mdlhist, 'Nominal')
rd.plot.mdlhistvals(mdlhist_loaded, 'Nominal')
# both plots give equivalent outputs!!