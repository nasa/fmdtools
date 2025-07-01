import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import networkx as nx

class Environment:
    X_SIZE = 12
    Y_SIZE = 12
    BLOCK  = 10.0

    def __init__(self, *, seed: int = 5):
        rng = np.random.default_rng(seed)
        self.features = {
            "occupied":   self._random_mask(rng, 0.40),
            "disallowed": self._random_mask(rng, 0.40),
            "restricted": self._random_mask(rng, 0.0),
        }
        x0, y0 = 0, 30
        x1, y1 = 20, 110
        i0 = int(y0 // self.BLOCK)
        j0 = int(x0 // self.BLOCK)
        i1 = min(int(y1 // self.BLOCK), self.Y_SIZE - 1)
        j1 = min(int(x1 // self.BLOCK), self.X_SIZE - 1)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                self.features["restricted"][i, j] = True
        x0, y0 = 30, 110
        x1, y1 = 110, 110
        i0 = int(y0 // self.BLOCK)
        j0 = int(x0 // self.BLOCK)
        i1 = int(y1 // self.BLOCK)
        j1 = int(x1 // self.BLOCK)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                self.features["restricted"][i, j] = True
        x0, y0 = 60, 0
        x1, y1 = 110, 70
        i0 = int(y0 // self.BLOCK)
        j0 = int(x0 // self.BLOCK)
        i1 = int(y1 // self.BLOCK)
        j1 = int(x1 // self.BLOCK)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                self.features["restricted"][i, j] = True
        self._clear_feature_at((0, 0))
        self._clear_feature_at((10.0, 10.0))
        self._clear_feature_at((0, 10.0))
        self._clear_feature_at((10.0, 0))
        self._clear_feature_at((100.0, 100.0))
        self._clear_feature_at((90.0, 100.0))
        self._clear_feature_at((100.0, 90.0))
        self._clear_feature_at((90.0, 90.0))

    def _random_mask(self, rng, p_true):
        return rng.random((self.X_SIZE, self.Y_SIZE)) < p_true

    def _index(self, x, y):
        return int(x // self.BLOCK), int(y // self.BLOCK)

    def _clear_feature_at(self, pt):
        i, j = self._index(*pt)
        for layer in self.features.values():
            layer[i, j] = False

class DroneFlightGrid:
    x_size          = 96
    y_size          = 96
    blocksize       = 1.25
    fuel_rate       = 2
    max_distance    = 6
    disallowed_cost = 10
    occupied_cost   = 20
    restricted_cost = 1000
    point_start     = (10.0, 10.0)
    point_end       = (100.0, 100.0)

    def __init__(self, env: Environment):
        self.env = env
        self.grid = np.array([[(j*self.blocksize, i*self.blocksize)
                               for j in range(self.x_size)]
                              for i in range(self.y_size)])
        self.grid_costs   = np.zeros((self.y_size, self.x_size))
        self.fuel_costs   = np.empty((self.y_size, self.x_size), object)
        self.total_costs  = np.empty((self.y_size, self.x_size), object)
        self.edge_weights = np.empty((self.y_size, self.x_size), object)
        for i in range(self.y_size):
            for j in range(self.x_size):
                self.fuel_costs[i, j]   = {}
                self.total_costs[i, j]  = {}
                self.edge_weights[i, j] = {}

    def to_index(self, x, y):
        j = int(x // self.blocksize)
        i = int(y // self.blocksize)
        j = min(max(j, 0), self.x_size - 1)
        i = min(max(i, 0), self.y_size - 1)
        return j, i

    def recursive_neighbor_gen(self, j, i, dist_remaining, visited=None):
        if visited is None:
            visited = set()
        if (j, i) in visited or dist_remaining < 0:
            return set()
        visited.add((j, i))
        neigh = {(j, i)}
        cx = j*self.blocksize + self.blocksize/2
        cy = i*self.blocksize + self.blocksize/2
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                nxp = cx + dx*self.blocksize
                nyp = cy + dy*self.blocksize
                if 0 <= nxp < self.x_size*self.blocksize and 0 <= nyp < self.y_size*self.blocksize:
                    jj, ii = self.to_index(nxp, nyp)
                    neigh |= self.recursive_neighbor_gen(jj, ii, dist_remaining-1, visited)
        return neigh

    def get_grid_costs(self):
        env = self.env
        for i in range(self.y_size):
            for j in range(self.x_size):
                neighbors = self.recursive_neighbor_gen(j, i, 1) - {(j, i)}
                occ_vals = []
                dis_vals = []
                res_vals = []
                for (j2, i2) in neighbors:
                    wx2 = j2*self.blocksize + self.blocksize/2
                    wy2 = i2*self.blocksize + self.blocksize/2
                    ei = min(int(wx2 // env.BLOCK), env.X_SIZE-1)
                    ej = min(int(wy2 // env.BLOCK), env.Y_SIZE-1)
                    occ_vals.append(env.features["occupied"][ei, ej])
                    dis_vals.append(env.features["disallowed"][ei, ej])
                    res_vals.append(env.features["restricted"][ei, ej])
                if occ_vals:
                    avg_occ = sum(occ_vals)/len(occ_vals)
                    avg_dis = sum(dis_vals)/len(dis_vals)
                    avg_res = sum(res_vals)/len(res_vals)
                else:
                    ei,jj = self.to_index(*self.grid[j, i])
                    avg_occ = env.features["occupied"][ei,jj]
                    avg_dis = env.features["disallowed"][ei,jj]
                    avg_res = env.features["restricted"][ei,jj]
                env_cost = (self.occupied_cost*avg_occ +
                            self.disallowed_cost*avg_dis +
                            self.restricted_cost*avg_res)
                self.grid_costs[i, j] = env_cost

    def get_edge_weights(self):
        self.get_grid_costs()
        for i in range(self.y_size):
            for j in range(self.x_size):
                cx = j*self.blocksize + self.blocksize/2
                cy = i*self.blocksize + self.blocksize/2
                for (j2, i2) in self.recursive_neighbor_gen(j, i, self.max_distance):
                    nxp = j2*self.blocksize + self.blocksize/2
                    nyp = i2*self.blocksize + self.blocksize/2
                    dist = math.hypot(cx-nxp, cy-nyp)
                    fuel = self.fuel_rate * dist
                    total = fuel + self.grid_costs[i2, j2] * dist
                    self.fuel_costs[i, j][(j2, i2)]   = fuel
                    self.total_costs[i, j][(j2, i2)]  = total
                    self.edge_weights[i, j][(j2, i2)] = total

    def nx_graph_gen(self):
        self.get_edge_weights()
        G = nx.DiGraph()
        for i in range(self.y_size):
            for j in range(self.x_size):
                v = (j, i)
                for u, w in self.edge_weights[i, j].items():
                    G.add_edge(v, u, weight=w)
        return G

    def a_star(self, start_xy, goal_xy):
        start = self.to_index(*start_xy)
        goal  = self.to_index(*goal_xy)
        G = self.nx_graph_gen()
        heur = lambda a, b: math.hypot(a[0]-b[0], a[1]-b[1])
        path_ij = nx.astar_path(G, start, goal, heuristic=heur, weight="weight")
        return [tuple(self.grid[j, i]) for (j, i) in path_ij]

    def a_star_worldcoords(self):
        return self.a_star(self.point_start, self.point_end)


def run_demo():
    env  = Environment(seed=5)
    grid = DroneFlightGrid(env)
    path = grid.a_star_worldcoords()
    for p in path:
        print(np.array(p))

    fig, ax = plt.subplots(figsize=(6, 6))
    cmap_hex = {"occupied":   "#d62728",
                "disallowed": "#1f77b4",
                "restricted": "#7f7f7f"}
    for layer, hexcol in cmap_hex.items():
        ax.imshow(
            env.features[layer],
            cmap=ListedColormap(["none", hexcol]),
            origin="lower",
            extent=(0, env.X_SIZE*env.BLOCK, 0, env.Y_SIZE*env.BLOCK),
            alpha=0.5
        )
    suitable = ~(env.features["occupied"] |
                 env.features["disallowed"] |
                 env.features["restricted"])
    ax.imshow(
        suitable.astype(int),
        cmap=ListedColormap(["none", "lightgreen"]),
        origin="lower",
        extent=(0, env.X_SIZE*env.BLOCK, 0, env.Y_SIZE*env.BLOCK),
        alpha=0.5
    )
    xs, ys = zip(*path)
    ax.plot(xs, ys, "-o", color="black", lw=1, ms=2)
    ax.set_aspect("equal")
    ax.set_xlim(0, env.X_SIZE*env.BLOCK)
    ax.set_ylim(0, env.Y_SIZE*env.BLOCK)
    ax.set_xticks(np.arange(0, grid.x_size*grid.blocksize+1, 25))
    ax.set_yticks(np.arange(0, grid.y_size*grid.blocksize+1, 25))
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticklabels())
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_demo()
