# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 12:23:04 2026

@author: dhulse
"""
import json, re

from fmdtools.define.object.geom import GeomJSON, GeomJSONParameter
from fmdtools.define.architecture.geom import GeomArchitecture
from fmdtools.define.base import load_json
from fmdtools.define.environment import Environment

from pyproj import Transformer

from shapely import from_geojson
from shapely.ops import transform

bay_area_transformer = Transformer.from_crs("EPSG:4326", "EPSG:26943", always_xy=True)

js = load_json('moffett_overpass_aeroway_data.geojson', load_dict=False)


class LocalJSONParameter(GeomJSONParameter):

    def create_shape(self):
        return transform(bay_area_transformer.transform, from_geojson(self.geojson))

class LocalGeomJSON(GeomJSON):

    container_p = LocalJSONParameter


class RunwayParameter(LocalJSONParameter):
    buffer_edge: tuple = (100.0*0.3048, (('cap_style', 'flat'), ))


class RunwayGeom(GeomJSON):
    container_p = RunwayParameter


class TaxiwayParameter(LocalJSONParameter):
    buffer_edge: tuple = (37.5*0.3048, (('cap_style', 'flat'), ))


class TaxiwayGeom(GeomJSON):
    container_p = TaxiwayParameter



def get_name(jsondict):
    if 'name' in jsondict['properties']:
        name = jsondict['properties']['name']
    elif 'ref' in jsondict['properties']:
        name = jsondict['properties']['ref']
    else:
        name = jsondict['id']
    name = re.sub(r'[-\.\s]+', '_', name)
    return name.lower()



class MoffettField(GeomArchitecture):
    container_p = GeomJSONParameter

    def init_architecture(self, **kwargs):
        all_feats = json.loads(self.p.geojson)
        for feat in all_feats['features']:
            p={'geojson': json.dumps(feat)}
            if feat['properties']['aeroway'] == 'runway':
                self.add_geom("runway_"+get_name(feat), RunwayGeom, p=p)
            elif feat['properties']['aeroway'] == 'taxiway':
                self.add_geom("taxiway_"+get_name(feat), TaxiwayGeom, p=p)
            elif feat['properties']['aeroway'] == 'taxilane':
                self.add_geom("taxilane_"+get_name(feat), LocalGeomJSON, p=p)
            elif feat['properties']['aeroway'] == 'apron':
                self.add_geom("apron_"+get_name(feat), LocalGeomJSON, p=p)
            elif feat['properties']['aeroway'] == 'helipad':
                self.add_geom("helipad_"+get_name(feat), LocalGeomJSON, p=p)
            elif 'building' in feat['properties']:
                self.add_geom("building_"+get_name(feat), LocalGeomJSON, p=p)
            elif feat['properties']['aeroway'] == 'holding_position':
                self.add_geom("hp_"+get_name(feat), LocalGeomJSON, p=p)
            elif 'name' in feat['properties']:
                self.add_geom(get_name(feat), LocalGeomJSON, p=p)

    def show_infrastructure(self, overlay_labels=True, title = "Moffett Field", **kwargs):
        taxiways = {name: {'shapes': {'shape': {'label': 'taxiway', 'color': 'red'}, 'edge': {'fill': True, 'color': 'gray', 'label': ''}}}
                    for name in self.geoms if name.startswith('taxiway_')}
        runways = {name: {'shapes': {'shape': {'label': 'runway', 'color': 'blue', 'overlay_label': name}, 'edge': {'fill': True, 'color': 'gray', 'label': ''}}}
                   for name in self.geoms if name.startswith('runway_')}
        aprons = {name: {'shapes': {'shape': {'label': 'apron', 'color': 'pink', 'fill': True}}}
                   for name in self.geoms if name.startswith('apron_')}
        helipads = {name: {'shapes': {'shape': {'label': 'helipad', 'color': 'purple', 'fill': True, 'overlay_label': 'helipad'}}}
                   for name in self.geoms if name.startswith('helipad_')}
        buildings = {name: {'shapes': {'shape': {'label': 'building', 'color': 'darkorange', 'fill': True, 'overlay_label': name}}}
                   for name in self.geoms if name.startswith('building_')}
        hps = {name: {'shapes': {'shape': {'label': 'holding_position', 'color': 'black'}}}
                   for name in self.geoms if name.startswith('hp_')}
        taxilanes = {name: {'shapes': {'shape': {'label': 'taxilane', 'color': 'red', 'linestyle': '--'}}}
                   for name in self.geoms if name.startswith('taxilane_')}
        geoms={**taxiways, **runways, **aprons, **helipads, **buildings, **hps, **taxilanes}
        if not overlay_labels:
            for geom in geoms:
                geoms[geom]['shapes']['shape'].pop("overlay_label", "")
        return self.show(geoms=geoms, xlabel="meters", ylabel="meters", title=title, **kwargs)

    def show_as_one_color(self, title = "Moffett Field", **kwargs):
        geoms = {name: {'shapes': {'shape': {'label': 'airport', "color": "gray"}}} for name in self.geoms}
        return self.show(geoms=geoms, xlabel="meters", ylabel="meters", title=title, **kwargs)
# from feature collection


class AirportEnvironment(Environment):
    arch_ga = MoffettField
    default_ga = {'p': 'moffett_overpass_aeroway_data.geojson'}



if __name__ == "__main__":
    r = GeomJSON(p='moffett_overpass_aeroway_data.geojson')
    # fig, ax = r.show(shapes={'shape': {'color': 'red', 'label': 'airport'}})


    all_feats = json.loads(r.p.geojson)
    all_feats['features'][11]

    wg = RunwayGeom(p={"geojson": json.dumps(all_feats['features'][11])})
    wg.show()

    mf = MoffettField(p = 'moffett_overpass_aeroway_data.geojson')
    
    # mf.show(figsize=(10,10))
    
    # mf.show(figsize=(10,10), z=True)
    mf.show_infrastructure(figsize=(10,10))
    mf.show_as_one_color(legend=dict(bbox_to_anchor = (0.5,1)))

    ae = AirportEnvironment()
