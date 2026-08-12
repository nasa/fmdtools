"""
Lifeguard (responder) model for the water-rescue simulation.

The responder stands at their post scanning the water with a sweeping
"paracentral" vision cone plus a peripheral semicircle. Once a rescue alert
is raised (by their own scan or the drone), they swim a straight-line path
to the victim and complete the rescue.

Also contains the combined plotting helper used to animate the simulation.
"""
from fmdtools.define.container.mode import Mode
from fmdtools.define.block.function import Function
from fmdtools.define.container.state import State
from fmdtools.define.architecture.geom import GeomArchitecture
from model_environment import BeachEnvironment
import numpy as np
from fmdtools.define.object.geom import GeomPoly, ExGeomState, GeomParameter


# --- Vision zone geometry helpers ---

def make_peripheral_shell(cx, cy, r=150.0):
    """Build a semicircle polygon (radius r) centered at (cx, cy).

    Represents the responder's peripheral vision facing the water.
    """
    angles = np.linspace(0, np.pi, 50)
    curved = [(cx + r*np.cos(a), cy + r*np.sin(a)) for a in angles]
    flat = [(cx + r, cy), (cx - r, cy)]  # close the shape along the shoreline
    return tuple(curved + flat)


def make_paracentral_shell(cx, cy, scan_angle=np.pi/2, r=300.0, half_cone=np.radians(4)):
    """Build a narrow vision cone polygon pointing in `scan_angle` direction.

    Represents the responder's focused (paracentral) vision, which sweeps
    back and forth across the water.
    """
    angles_para = np.linspace(scan_angle - half_cone, scan_angle + half_cone, 20)
    curved_para = [(cx + r*np.cos(a), cy + r*np.sin(a)) for a in angles_para]
    return tuple([(cx, cy)] + curved_para + [(cx, cy)])


# default static versions (for ga/check_vision_zones) at the responder's post
cx, cy = 400.0, 0.0
shell_pts = make_peripheral_shell(cx, cy)
shell_para = make_paracentral_shell(cx, cy)


class SemicircleParam(GeomParameter):
    """Geometry parameter for the peripheral (semicircle) vision zone."""
    coordinates: tuple = (shell_pts, ())


class ParacentralParam(GeomParameter):
    """Geometry parameter for the paracentral (cone) vision zone."""
    coordinates: tuple = (shell_para, ())


class ParacentralZone(GeomPoly):
    """Polygon geometry for the focused vision cone."""
    container_p = ParacentralParam
    container_s = ExGeomState


class PeripheralZone(GeomPoly):
    """Polygon geometry for the peripheral vision semicircle."""
    container_p = SemicircleParam
    container_s = ExGeomState


class ResponderGeomArchitecture(GeomArchitecture):
    """Bundles the responder's two vision zones into one geometry architecture."""
    def init_architecture(self, **kwargs):
        self.add_geom('peripheral', PeripheralZone)
        self.add_geom('paracentral', ParacentralZone)


ga = ResponderGeomArchitecture()


def check_vision_zones(ga, point):
    """Return {zone_name: bool} indicating which vision zones contain `point`."""
    return {name: geom.at(point) for name, geom in ga.geoms.items()}


print(check_vision_zones(ga, (400, 10)))


class ResponderState(State):
    """State of the responder: position, rescue path progress, and scan sweep."""
    location: tuple = (400, 0)     # current (x, y) position
    on_rescue: bool = False        # whether currently performing a rescue
    path_index: int = 0            # progress along the rescue path
    path: list = []                # list of waypoints to the victim
    x: float = 400                 # x position (kept in sync with location)
    y: float = 0.0                 # y position (kept in sync with location)
    time_rescued: float = -1.0     # sim time the rescue completed (-1 = not yet)
    scan_angle: float = np.pi / 2  # current scan direction, starts pointing up
    scan_direction: float = 1.0    # 1 = sweeping right, -1 = sweeping left


class ResponderMode(Mode):
    """Responder modes: standing by (scanning) or performing a rescue."""
    opermodes = ("standby", "rescue")
    mode: str = "standby"


class Responder(Function):
    """Lifeguard that scans the water and swims out to rescue victims."""
    container_s = ResponderState
    container_m = ResponderMode
    flow_environment = BeachEnvironment
    arch_ga = ResponderGeomArchitecture

    def init_block(self, **kwargs):
        """Place the responder at their post on the beach."""
        self.s.x = 400.0
        self.s.y = 0.0
        self.s.location = (400.0, 0.0)

    def create_rescue_path(self, speed):
        """Build a straight-line path from the responder to the victim.

        Parameters
        ----------
        speed : float
            Swim speed in m/s; one waypoint is generated per 1-second step.

        Returns
        -------
        list of (x, y) points ending exactly at the victim's location.
        """
        victim_loc = getattr(self.flow_environment.c, "person_location",
                             self.environment.c.p.rescue_locations[0])
        dx = victim_loc[0] - self.s.location[0]
        dy = victim_loc[1] - self.s.location[1]
        dist = np.sqrt(dx**2 + dy**2)

        steps = int(dist / speed)    # number of 1-second steps at "speed" m/s

        # unit vector toward the victim (guard against zero distance)
        if dist > 1e-9:
            ux = dx / dist
            uy = dy / dist
        else:
            ux, uy = 0.0, 0.0

        # evenly spaced waypoints along the line
        path = []
        for i in range(steps):
            point = (self.s.location[0] + ux * speed * i,
                     self.s.location[1] + uy * speed * i)
            path.append(point)

        path.append(victim_loc)   # end at victim

        return path  # list of points of path

    def set_rescue_goal(self):
        """Determine path to follow to rescue person."""
        self.m.set_mode("rescue")
        self.s.path = self.create_rescue_path(4)

    def follow_rescue_path(self):
        """Advance one waypoint along the rescue path; finish the rescue on arrival."""
        if self.s.path_index < len(self.s.path):
            # still swimming: move to the next waypoint
            self.s.location = self.s.path[self.s.path_index]
            self.s.x = self.s.location[0]
            self.s.y = self.s.location[1]
            self.s.path_index += 1
        else:
            # arrived at the victim: record the rescue and reset states
            if self.s.time_rescued < 0:
                self.s.time_rescued = self.t.time

            victim = self.s.location
            self.environment.c.set_pts([victim], "person_to_rescue", False)
            self.m.set_mode("standby")  # arrived, go back to standby
            self.environment.c.set_pts([victim], "with_buoy", False)
            self.environment.c.set_pts([victim], "rescued", True)

    def scan(self):
        """Sweep the focused vision cone across the water looking for victims.

        The cone oscillates between left/right angular limits. If a distressed
        swimmer falls inside the cone, the shared rescue alert is raised.
        """
        scan_speed = np.radians(5)     # how far the cone rotates per timestep
        half_cone = np.radians(4)      # half-width of the vision cone
        left_limit = np.radians(10)    # sweep boundary (toward +x)
        right_limit = np.radians(170)  # sweep boundary (toward -x)

        # advance the sweep, reversing direction at the limits
        self.s.scan_angle += scan_speed * self.s.scan_direction

        if self.s.scan_angle >= right_limit:
            self.s.scan_direction = -1.0
        elif self.s.scan_angle <= left_limit:
            self.s.scan_direction = 1.0

        # rebuild the cone polygon at the current position and angle
        cx, cy = self.s.x, self.s.y
        r_para = 350.0
        angles_para = np.linspace(self.s.scan_angle - half_cone,
                                  self.s.scan_angle + half_cone, 20)
        curved_para = [(cx + r_para*np.cos(a), cy + r_para*np.sin(a))
                       for a in angles_para]
        shell_para = tuple([(cx, cy)] + curved_para + [(cx, cy)])

        from shapely.geometry import Point, Polygon
        cone = Polygon(shell_para)

        # use same logic as drone's check_vicinity to find the nearest victim
        in_range, pt = self.environment.c.check_vicinity((cx, cy), radius=r_para)
        print(f"in_range: {in_range}, pt: {pt}")
        if in_range:
            if pt is not None:
                p = Point(float(pt[0]), float(pt[1]))
                print(f"cone contains: {cone.contains(p)}, covers: {cone.covers(p)}, "
                      f"pt: {pt}, cone bounds: {cone.bounds}")
                # only trigger the alert if the victim is inside the vision cone
                if cone.contains(p) or cone.covers(p):
                    self.environment.c.s.rescue = True

    def dynamic_behavior(self):
        """Per-timestep behavior: scan while on standby, swim when rescuing."""
        if self.m.in_mode("standby"):
            self.scan()  # always scanning
            if self.environment.c.s.rescue:
                self.set_rescue_goal()
        if self.m.in_mode("rescue"):
            self.follow_rescue_path()


def plot_combined_response_from(time, history={}, mdl=None, fig=None, ax=None,
                                legend_kwargs={}, title='', **kwargs):
    """Plot a single frame of the simulation at time `time`.

    Draws the beach map, the responder's vision zones (when on standby at
    base), and the trajectories of the drone (red) and responder (green).
    Intended for use with fmdtools animation utilities.
    """
    from fmdtools.analyze.common import prep_animation_title, clear_prev_figure
    from matplotlib.patches import Polygon as MplPolygon
    from model_environment import sim_properties

    kw = prep_animation_title(time, title=title)
    title = kw['title']
    if fig:
        # clear the previous frame when animating
        kw = clear_prev_figure(fig=fig, ax=ax)
        fig = kw.get('fig', None)
        ax = kw.get('ax', None)

    # base layer: the beach/water map with victim status colors
    fig, ax = mdl.flows['environment'].c.show_from(
        time, history.flows.environment.c,
        properties=sim_properties, legend_kwargs=legend_kwargs,
        fig=fig, ax=ax
    )

    # responder position/scan state at this frame
    cx = history.fxns.responder.s.x[time]
    cy = history.fxns.responder.s.y[time]
    scan_angle = history.fxns.responder.s.scan_angle[time]
    mode = history.fxns.responder.m.mode[time]
    base_x, base_y = 400.0, 0.0
    at_base = np.sqrt((cx - base_x)**2 + (cy - base_y)**2) < 50.0

    # only draw the vision zones when the responder is scanning from base
    if mode == 'standby' and at_base:
        peri_patch = MplPolygon(list(make_peripheral_shell(cx, cy)), closed=True,
                                fill=False, edgecolor='blue', linewidth=1.5,
                                label='peripheral')
        cone_patch = MplPolygon(list(make_paracentral_shell(cx, cy, scan_angle)),
                                closed=True, fill=False, edgecolor='yellow',
                                linewidth=1.5, label='paracentral')
        ax.add_patch(peri_patch)
        ax.add_patch(cone_patch)

    # overlay movement trajectories up to this frame
    nhist = history.cut(time, newcopy=True)
    nhist.plot_trajectory('fxns.beach_aircraft.s.x', 'fxns.beach_aircraft.s.y',
                          fig=fig, ax=ax, color='red')
    nhist.plot_trajectory('fxns.responder.s.x', 'fxns.responder.s.y',
                          fig=fig, ax=ax, color='green')
    return fig, ax


responder = Responder()
print(responder.environment.c.find_all_prop("person_to_rescue"))
#print(responder.create_rescue_path(4))
