from fmdtools.define.object.coords import Coords, CoordsParam
from fmdtools.define.environment import Environment
from fmdtools.define.block.function import Function
from fmdtools.define.container.state import State
import numpy as np

class BeachMapParam(CoordsParam):
    """goodness gracious!"""
    x_size: int = 20 # width of grid
    y_size: int = 5 # length of grid
    blocksize: float = 100 #100 meters per tile
    base_locations: tuple = ((400.0,0.0),(1600.0,0.0)) # points of where to launch drones from
    rescue_locations: tuple = ((1000.0,300.0),) # points where people need rescue
    num_distress: int = 1
    time_to_reach: float = 10.0
    vicinity_radius: float = 200.0


class BeachBehaviorState(State):
        time_to_reach: int = 30 
        person_drowning: bool = False
        rescue: bool = False
        
class BeachMap(Coords):
    container_p = BeachMapParam
    container_s = BeachBehaviorState
    state_survival_rate = (float, 1)
    state_person_to_rescue = (bool, False)
    point_person_location =(0,0)
    state_with_buoy = (bool, False)
    state_rescued = (bool, False)
    state_distress_timer = (float, 30.0)

    feature_beach = (bool, False)
    feature_water = (bool, False)
    feature_base = (bool, False)
    feature_rescue_location = (bool, False)

    def init_properties(self, *args, **kwargs):
        self.set_pts(self.p.base_locations, "base", True)
        self.set_pts(self.p.rescue_locations, "person_to_rescue", False)
        b = self.p.blocksize
        self.set_range("beach", True, ymin=0, ymax=b)
        self.set_range("water", True, ymin=b*2)

        victim_pts = self.p.rescue_locations[:self.p.num_distress]
        if victim_pts:
            self.person_location = victim_pts[0]

    def check_vicinity(self, point, radius=None):
        if radius is None: 
            radius = self.p.vicinity_radius
        # i will edit this in a sec couks ethsi was a big change 
        active = []
        for pt in self.p.rescue_locations:
            if self.get(pt[0],pt[1], "person_to_rescue"):
                active.append(pt)
        if not active:
            return (False, None)
        
        pts = np.array(active)
        dists = np.hypot(pts[:,0]-point[0],pts[:,1]-point[1])
        i = int(np.argmin(dists))
        return (dists[i]<=radius, tuple(pts[i]))
    
class BeachEnvironment(Environment):
    coords_c = BeachMap

class BeachBehavior(Function):
    container_s = BeachBehaviorState
    flow_environment = BeachEnvironment

    def dynamic_behavior(self):
        env = self.environment.c
        for pt in env.p.rescue_locations:
            timer = env.get(pt[0], pt[1], "distress_timer")
            if timer > 0:
                env.set(pt[0], pt[1], "distress_timer", timer - self.t.dt)
            elif not env.get(pt[0], pt[1], "person_to_rescue"):
                env.set_pts([pt], "person_to_rescue", True)


sim_properties = {
    'beach':              {'color': 'sandybrown'},
    'water':         {'color': 'steelblue'},
    'person_to_rescue': {'color': 'red'},
    'base':               {'color': 'black'},
    'with_buoy': {'color':'yellow'},
    'rescued': {'color':'green'},
    #,'rescue_location': {'color':'pink'},
}
