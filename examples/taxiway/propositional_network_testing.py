# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 14:34:59 2023

@author: dhulse
"""
from model import taxiway_model
from fmdtools.analyze import graph
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
from recordclass import dataobject, asdict

def determine_true(flow, mapping):
    a=1
def get_degraded_fields(mdlhists, to_include = 'flows'):
    from fmdtools import resultdisp as rd
    import numpy as np
    faulty = rd.process.flatten_hist(mdlhists['faulty'], to_include=to_include)
    nominals = rd.process.flatten_hist(mdlhists['nominal'], to_include=to_include)
    degraded = dict.fromkeys(nominals)
    for att, vals in nominals.items():
        degraded[att]=nominals[att]==faulty[att]
    perc_degraded = (len(degraded)-np.sum(np.array(list(degraded.values())), axis = 0))/len(degraded)
    return degraded, perc_degraded    

def get_faulty_fxns(mdlhists):
    from fmdtools import resultdisp as rd
    import numpy as np
    fxnhist = mdlhists['faulty']['functions']
    faulty = dict.fromkeys(fxnhist)
    for fxn in faulty:
        faulty[fxn]=np.sum(np.array([*fxnhist[fxn]['faults'].values()]), axis=0)
    perc_faulty = np.sum(np.array([i>0 for i in faulty.values()]), axis=0)/len(faulty)
    return faulty, perc_faulty

def get_off_true_ground(mdlhists, scenname='faulty'):
    flowhist = mdlhists[scenname]['flows']['ground']
    perc_flowhist = flowhist['atc']
    true_flowhist = {f:v for f,v in flowhist.items() if f in perc_flowhist}
    true_fields, perc_true = get_degraded_fields({"nominal":true_flowhist, "faulty":perc_flowhist}, to_include="alla")
    return true_fields, perc_true

def get_cycled(mdlhists, scenname='faulty', assets = ['ma1', 'ma2', 'ma3', 'ua1', 'ua2', 'ua3', 'h1', 'h2']):
    fxnhist = mdlhists[scenname]['functions']
    cycled = dict.fromkeys(assets)
    for asset in assets:
        cycled[asset] = fxnhist[asset]['cycled']
    perc_cycled = np.sum([*cycled.values()], axis=0)/len(cycled)
    return cycled, perc_cycled

#TODO: this isn't working. Are the closest locations getting updated in the mdlhist?
def get_off_true_location(mdlhists, scenname='faulty', assets = ['ma1', 'ma2', 'ma3', 'ua1', 'ua2', 'ua3', 'h1', 'h2']):
    flowhist = mdlhists.get(scenname).flows.location
    off_true_locations = dict.fromkeys(assets)
    for asset in assets:
        loc = flowhist.get(asset).s
        
        closest_dist=100
        off_true_locations[asset]=[]
        for i, stage in enumerate(flowhist.get(asset).s.stage):
            if stage!='taxi': 
                off_true_locations[asset].append(False)
            else:
                for other_asset in assets:
                    if other_asset!=asset:
                        asset_loc = flowhist.get(other_asset).s
                        dist = np.sqrt((asset_loc['x'][i]-loc['x'][i])**2+(asset_loc['y'][i]-loc['y'][i])**2)
                        if dist<closest_dist:
                            closest_dist = dist
                            closest_asset=other_asset
                if closest_dist<10:
                    perc_closest = flowhist.get(asset).closest.s.x[i], flowhist.get(asset).closest.s.y[i]
                    true_closest = flowhist.get(closest_asset).s.x[i], flowhist.get(closest_asset).s.y[i]
                    off_true_locations.get(asset).append(not(perc_closest==true_closest))
                else: off_true_locations.get(asset).append(True)
    perc_off_true = np.sum([*off_true_locations.values()], axis=0)/len(off_true_locations)
    return off_true_locations, perc_off_true

if __name__ == "__main__":
    
    mdl = taxiway_model()
    
    from fmdtools.sim import propagate as prop
    from fmdtools.analyze.graph import ModelGraph, ModelTypeGraph
    from fmdtools.analyze.graph import MultiFlowGraph, CommsFlowGraph
    
    endresults, mdlhist = prop.one_fault(mdl, "ma3", "lost_sight",
                                    desired_result={93: {"graph.flows.location":(MultiFlowGraph, {'include_glob':False})},
                                                    110:{"graph.flows.location":(MultiFlowGraph, {'include_glob':False})}, 
                                                    20:{"graph": ModelGraph},
                                                    120:{"graph": ModelGraph, "endclass":()}})
    mg = ModelGraph(mdl)
    mg.set_pos(**nx.spring_layout(mg.g))
    mg.set_edge_labels(title='')
    mg.draw(figsize=(10,10))
    
    
    tg = ModelTypeGraph(mdl)
    tg.draw()
    
    #mg.set_node_styles(name={('ua1', 'ua2'):{'node_color':'green'}})
    mg.add_node_groups(highlight=('ua1', 'ua2'), highlight2=("h1", "h2"))
    mg.set_node_styles(group={'highlight':{'node_color':'green'}, 'highlight2': {'node_color':'red'}},
                       label={})
    mg.draw(figsize=(10,10))
    d = mg.draw_graphviz()
    
    gg=ModelGraph(mdl)
    gg.set_exec_order(mdl)
    gg.draw()
    
    #graph.show(mdl.graph, pos=pos, scale=1.0, figsize=(10,10), highlight=[["ua1"],["ua2"],[*mdl.fxns['ua1'].flows.keys()]])
    
    
    # Requests graph view
    #g1=mdl.Requests.create_multigraph(include_glob=False)
    #draw_graph(g1, nodelabels="last", node_styles=node_styles, 
    #           edge_styles=edge_styles)
    
    cg = CommsFlowGraph(mdl.flows['requests'], include_glob=False, ports_only=True)
    cg.set_edge_styles(label={'sends':{'arrows':True}})
    cg.set_pos(**nx.spring_layout(cg.g))
    cg.draw()
    
    # Location Perception - arrows between true and closest should be grey to show link
    #g=mdl.flows['location'].create_multigraph(include_glob=True)  
    #g=mdl.flows['location'].create_multigraph(include_glob=False)
    
    gv2 = MultiFlowGraph(mdl.flows['location'], include_glob=False)
    gv2.set_edge_styles(label={'sends':{'arrows':True}})
    gv2.draw(nodelabels="last")
    
    # Ground Perception
    
    gv3 = MultiFlowGraph(mdl.flows['ground'],
                         include_glob=True, include_states=True,
                             send_connections={"asset_area":"asset_area", 
                                               "area_allocation":"area_allocation",
                                               "asset_assignment":"asset_assignment"})

    gv3.draw(nodelabels='last')
    d=gv3.draw_graphviz()
    
    gv4= endresults.t93p0.graph.flows.location
    gv4.set_pos(**nx.spring_layout(nx.Graph(endresults.t93p0.location)))
    gv4.set_node_styles(degraded={'node_color':'yellow'})
    gv4.draw(figsize=(6,5), withlegend=False, title="t=93")
    d=gv4.draw_graphviz()
    
    endresults, mdlhist = prop.one_fault(mdl, "atc", "single_wrong_command", time=10, 
                                         desired_result={11:{"graph.flows.requests":(CommsFlowGraph, {'include_glob':False, "ports_only":True})}, 
                                                         20:{"graph":ModelGraph}})
    off, perc_off = get_off_true_location(mdlhist)
    
    gv5 = endresults.t11p0.graph.flows.requests
    gv5.set_edge_styles(label={'sends':{'arrows':True}})
    #gv5.set_edge_labels(title='')
    gv5.set_node_styles(degraded={'node_color':'yellow'})
    gv5.set_node_labels(label='last', subtext='status')
    gv5.draw(nodelabels='last')
    
    d=gv5.draw_graphviz()
    
    
    