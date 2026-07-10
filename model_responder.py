from fmdtools.define.container.mode import Mode
from fmdtools.define.block.function import Function
from fmdtools.define.block.action import Action
from fmdtools.define.architecture.action import ActionArchitecture
from fmdtools.define.architecture.function import FunctionArchitecture
from fmdtools.define.container.state import State
from fmdtools.define.flow.base import Flow
from fmdtools.define.container.parameter import Parameter
from fmdtools.define.object.geom import Geom
from fmdtools.define.architecture.geom import GeomArchitecture
from model_environment import BeachEnvironment
import numpy as np
from shapely.geometry import Point, Polygon
from fmdtools.define.object.geom import GeomPoly, PolyParam, ExGeomState, GeomParameter

import numpy as np

def make_peripheral_shell(cx, cy, r=150.0):
    angles = np.linspace(0, np.pi, 50)
    curved = [(cx + r*np.cos(a), cy + r*np.sin(a)) for a in angles]
    flat = [(cx + r, cy), (cx - r, cy)]
    return tuple(curved + flat)

def make_paracentral_shell(cx, cy, scan_angle=np.pi/2, r=300.0, half_cone=np.radians(4)):
    angles_para = np.linspace(scan_angle - half_cone, scan_angle + half_cone, 20)
    curved_para = [(cx + r*np.cos(a), cy + r*np.sin(a)) for a in angles_para]
    return tuple([(cx, cy)] + curved_para + [(cx, cy)])

# default static versions (for ga/check_vision_zones)
cx, cy = 400.0, 0.0
shell_pts = make_peripheral_shell(cx, cy)
shell_para = make_paracentral_shell(cx, cy)

class SemicircleParam(GeomParameter):
    coordinates: tuple = (shell_pts, ())

class ParacentralParam(GeomParameter):
    coordinates: tuple = (shell_para, ())

class ParacentralZone(GeomPoly):
    container_p = ParacentralParam
    container_s = ExGeomState

class PeripheralZone(GeomPoly):
    container_p = SemicircleParam
    container_s = ExGeomState

class ResponderGeomArchitecture(GeomArchitecture):
    def init_architecture(self, **kwargs):
        self.add_geom('peripheral', PeripheralZone)
        self.add_geom('paracentral', ParacentralZone)

ga = ResponderGeomArchitecture()

def check_vision_zones(ga, point):
    return {name: geom.at(point) for name, geom in ga.geoms.items()}

print(check_vision_zones(ga, (400, 10)))

class ResponderState(State):
    location: tuple = (400,0)
    on_rescue: bool = False
    path_index: int = 0
    path: list = []
    x: float = 400
    y: float = 0.0
    time_rescued: float = -1.0
    scan_angle: float = np.pi / 2  # current scan direction, starts pointing up
    scan_direction: float = 1.0    # 1 = sweeping right, -1 = sweeping left

class ResponderMode(Mode):
    opermodes = ("standby", "rescue")
    mode: str = "standby"


class Responder(Function):
    container_s = ResponderState
    container_m = ResponderMode
    flow_environment = BeachEnvironment
    arch_ga = ResponderGeomArchitecture

    def init_block(self, **kwargs):
        self.s.x = 400.0
        self.s.y = 0.0
        self.s.location = (400.0, 0.0)


    def create_rescue_path(self, speed):
        victim_loc = getattr(self.flow_environment.c, "person_location", self.environment.c.p.rescue_locations[0])
        dx = victim_loc[0] - self.s.location[0]
        dy = victim_loc[1] - self.s.location[1]
        dist = np.sqrt(dx**2 + dy**2)

        steps = int(dist / speed)    # number of 1-second steps at "speed" m/s

        if dist > 1e-9:
            ux = dx / dist
            uy = dy / dist
        else:
            ux, uy = 0.0, 0.0 

        path = []
        for i in range(steps):
            point = (self.s.location[0] + ux * speed * i,
                    self.s.location[1] + uy * speed * i)
            path.append(point)

        path.append(victim_loc)   # end at victim

        return path #return list of points of path

    def set_rescue_goal(self):
        """Determine path to follow to rescue person."""
        self.m.set_mode("rescue")
        self.s.path = self.create_rescue_path(4)


    def follow_rescue_path(self):
        if self.s.path_index < len(self.s.path):
            self.s.location = self.s.path[self.s.path_index]
            self.s.x = self.s.location[0]
            self.s.y = self.s.location[1]
            self.s.path_index += 1
        else:
            if self.s.time_rescued < 0:
                self.s.time_rescued = self.t.time

            victim = self.s.location
            self.environment.c.set_pts([victim], "person_to_rescue", False)
            self.m.set_mode("standby")  # arrived, go back to standby
            self.environment.c.set_pts([victim],"with_buoy",False)
            self.environment.c.set_pts([victim],"rescued",True)
        

    def scan(self):
        scan_speed = np.radians(5)
        half_cone = np.radians(4)
        left_limit = np.radians(10)
        right_limit = np.radians(170)
        
        self.s.scan_angle += scan_speed * self.s.scan_direction
        
        if self.s.scan_angle >= right_limit:
            self.s.scan_direction = -1.0
        elif self.s.scan_angle <= left_limit:
            self.s.scan_direction = 1.0
        
        cx, cy = self.s.x, self.s.y
        r_para = 350.0
        angles_para = np.linspace(self.s.scan_angle - half_cone,
                                self.s.scan_angle + half_cone, 20)
        curved_para = [(cx + r_para*np.cos(a), cy + r_para*np.sin(a)) 
                    for a in angles_para]
        shell_para = tuple([(cx, cy)] + curved_para + [(cx, cy)])

        from shapely.geometry import Point, Polygon
        cone = Polygon(shell_para)

        # use same logic as drone's check_vicinity
        in_range, pt = self.environment.c.check_vicinity((cx, cy), radius=r_para)
        print(f"in_range: {in_range}, pt: {pt}")
        if in_range:
            if pt is not None:
                p = Point(float(pt[0]), float(pt[1]))
                print(f"cone contains: {cone.contains(p)}, covers: {cone.covers(p)}, pt: {pt}, cone bounds: {cone.bounds}")
                if cone.contains(p) or cone.covers(p):
                    self.environment.c.s.rescue = True

    def dynamic_behavior(self):
        if self.m.in_mode("standby"):
            self.scan()  # always scanning
            if self.environment.c.s.rescue:
                self.set_rescue_goal()
        if self.m.in_mode("rescue"):
            self.follow_rescue_path()



def plot_combined_response_from(time, history={}, mdl=None, fig=None, ax=None,
                                 legend_kwargs={}, title='', **kwargs):
    from fmdtools.analyze.common import prep_animation_title, clear_prev_figure
    from matplotlib.patches import Polygon as MplPolygon
    from model_environment import sim_properties

    kw = prep_animation_title(time, title=title)
    title = kw['title']
    if fig:
        kw = clear_prev_figure(fig=fig, ax=ax)
        fig = kw.get('fig', None)
        ax = kw.get('ax', None)

    fig, ax = mdl.flows['environment'].c.show_from(
        time, history.flows.environment.c,
        properties=sim_properties, legend_kwargs=legend_kwargs,
        fig=fig, ax=ax
    )

    cx = history.fxns.responder.s.x[time]
    cy = history.fxns.responder.s.y[time]
    scan_angle = history.fxns.responder.s.scan_angle[time]
    mode = history.fxns.responder.m.mode[time]
    base_x, base_y = 400.0, 0.0
    at_base = np.sqrt((cx - base_x)**2 + (cy - base_y)**2) < 50.0

    if mode == 'standby' and at_base:
        peri_patch = MplPolygon(list(make_peripheral_shell(cx, cy)), closed=True,
                                 fill=False, edgecolor='blue', linewidth=1.5, label='peripheral')
        cone_patch = MplPolygon(list(make_paracentral_shell(cx, cy, scan_angle)), closed=True,
                                 fill=False, edgecolor='yellow', linewidth=1.5, label='paracentral')
        ax.add_patch(peri_patch)
        ax.add_patch(cone_patch)

    nhist = history.cut(time, newcopy=True)
    nhist.plot_trajectory('fxns.beach_aircraft.s.x', 'fxns.beach_aircraft.s.y',
                          fig=fig, ax=ax, color='red')
    nhist.plot_trajectory('fxns.responder.s.x', 'fxns.responder.s.y',
                          fig=fig, ax=ax, color='green')
    return fig, ax


responder = Responder()
print(responder.environment.c.find_all_prop("person_to_rescue"))
#print(responder.create_rescue_path(4))