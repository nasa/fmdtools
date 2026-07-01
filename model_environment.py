from fmdtools.define.object.coords import Coords, CoordsParam
from fmdtools.define.environment import Environment
from fmdtools.define.block.function import Function
from fmdtools.define.container.state import State
import numpy as np

class BeachMapParam(CoordsParam):
    x_size: int = 20
    y_size: int = 10
    blocksize: float = 500 #500 meters per tile
    base_locations: tuple = ((2240.0,0.0),(3200.0,0.0))
    rescue_locations: tuple = ((10.0,50.0),)
    num_distress: int = 1
    time_to_reach: float = 10.0

class BeachMap(Coords):
    container_p = BeachMapParam
    state_survival_rate = (float, 1)
    state_person_to_rescue = (bool, False)
    point_person_location =(0,0)
    state_with_buoy = (bool, False)

    feature_beach = (bool, False)
    feature_water = (bool, False)
    feature_base = (bool, False)
    feature_rescue_location = (bool, False)
    
    def dynamic_behavior(self):
        self.p.time_to_reach -= 1

    def check_vicinity(self, point):
        return(abs(point[0] - self.person_location[0]), abs(point[1] - self.person_location[1]))

    def init_properties(self, *args, **kwargs):
        self.set_pts(self.p.base_locations, "base", True)
        self.set_pts(self.p.rescue_locations, "rescue_location", True)
        b = self.p.blocksize
        self.set_range("beach", True, ymin=0, ymax=b)
        self.set_range("water", True, ymin=b*2)

        water_pts=[]
        for pt in self.pts:
            if self.get(*pt, "water"):
                water_pts.append(pt)
        
    def get_leading_edge(self):
        distress_pts = [*self.find_all_prop("state_person_to_rescue", True, np.equal)]
        leading_edge = []
        for pt in distress_pts:
            neighbors = self.get_neighbors(*pt, direction='direct')
            any_to_spread = any([
                not (self.get(*p, "state_person_to_rescue") or self.get(*p, "cleared"))
                for p in neighbors
            ])
            if any_to_spread:
                leading_edge.append(pt)
        return leading_edge
    
    def find_closest_edge(self, *pt):
        edge_pts = self.get_leading_edge()
        if edge_pts:
            dists = np.sqrt(np.sum((np.array([*pt]) - edge_pts)**2, 1))
            return edge_pts[np.argmin(dists)]
        else:
            return []

    def set_time_to_reach(self, tstep=1.0):
        for pt in self.find_all_prop("state_person_to_rescue"):
            neighbors = self.get_neighbors(*pt, direction="direct")
            for npt in neighbors:
                if not self.get(*npt, "cleared") and not self.get(*npt, "person_in_distress"):
                    time_to_reach = self.get(*npt, "time_to_reach")
                    if np.isnan(time_to_reach):
                        self.set(*npt, "time_to_reach", self.get_reach_time(*npt))
                    else:
                        self.set(*npt, "time_to_reach", time_to_reach - tstep)

    def set_person_in_distress(self):
        for pt in self.find_all_prop("time_to_reach", value=0.0,
                                     comparator=np.less_equal):
            self.set(*pt, "state_person_to_rescue", True)
            self.set(*pt, "time_to_reach", np.nan)

    
class BeachEnvironment(Environment):
    coords_c = BeachMap

sim_properties = {
    'beach':              {'color': 'sandybrown'},
    'water':         {'color': 'steelblue'},
    'person_to_rescue': {'color': 'red',    'as_bool': True, 'alpha': 0.5},
    'base':               {'color': 'black'},
    'rescue_location': {'color':'pink'},
}
