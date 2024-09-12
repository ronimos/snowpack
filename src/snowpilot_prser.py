# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 13:35:55 2024

@author: Ron Simenhois
"""

import os
from glob import glob
import numpy as np
import pandas as pd
import argparse
from scipy.spatial import KDTree 
from geopy.distance import geodesic
import xml.etree.ElementTree as ET
from tqdm import tqdm
from collections import defaultdict
from param_config import HAND_HARD_2_NUMERIC, SNOW_PIT_PATH

WRF_LOCATIONS = os.path.abspath(os.path.join(os.getcwd(), '../data', 'WRF_locations', 'SnopwPackLocations.csv'))


class SnowPilotParser:
    
    def __init__(self,
                 file_name):
        
        self.file_name = file_name
        self.common_tag = '{http://caaml.org/Schemas/SnowProfileIACS/v6.0.3}'
        self.test_results = None
        self.layers = None
        self.HS = None
        self.location = None
        self.datetime = None
        self.get_HS = self._get_HS
        self.get_pit_loaction = self._get_pit_loaction
        self.get_pit_datetime = self._get_pit_datetime
        
    
    
    def _get_pit_loaction(self):
        """
        This function fetch the pit lat, lon, elevation, aspect and slope angle
        from its caamal file and add it to the SnowPilotParser objet instance

        Returns
        -------
        None.

        """
        POS_LAT_LONG = '{http://www.opengis.net/gml}pos'
        locations_params_tags = [self.common_tag + 'ElevationPosition', 
                                 self.common_tag + 'AspectPosition', 
                                 self.common_tag + 'SlopeAnglePosition']
        name_front_trim = len(self.common_tag)
        name_back_trim = -len('Position')

        location = {}

        root = ET.parse(self.file_name).getroot()
        try: 
            loc = next(root.iter(POS_LAT_LONG), None).text
            location['lat'], location['long'] = map(float, loc.split(' '))
        except AttributeError:
            self.location = None
            return # No lat, lon loation for this pit
            
        position_params = [t for t in root.iter() if t.tag in locations_params_tags]
        for tp in position_params:
            location[tp.tag[name_front_trim: name_back_trim]] = tp.find(self.common_tag + 'position').text
    
        self.location = location
        
        
    @classmethod
    def get_pit_loaction(cls, 
                         file_name):
        """
        This function is a class methous that uses _get_pit_loaction to fetch 
        the pit's lat, lon, elevation, aspect and slope angle its caamal file.


        Parameters
        ----------
        cls : TYPE
            DESCRIPTION.
        file_name : str
            path to the pit cammal xml file.

        Returns
        -------
        dect
            a dictionary with the pit's lat, lon, elevation, aspect and 
            slope angle.

        """
        
        pit = SnowPilotParser(file_name)
        pit._get_pit_loaction()
        
        return pit.location


    def _get_HS(self):
        """
        This fuction gets the HS for a sigle pit from its caamal.xml file
    
        Parameters
        ----------
        SnowPilotParser object
    
        Returns
        -------
        None.
    
        """
        
        hs_tag = self.common_tag + 'profileDepth'
        
        root = ET.parse(self.file_name).getroot()
        try:
            hs = next(root.iter(hs_tag), None).text
            self.HS = float(hs)
        except AttributeError:
            self.HS = None
            
            
    @classmethod
    def get_HS(cls, 
               file_name):
        """
        This calss methoud (static) function uses _get_HS to fetch the pit's
        depth from the pit's caamal xml file

        Parameters
        ----------
        cls : SnowPilotParser
            SnowPilotParser class instance.
        file_name : str
            The path to the pit's caamal xml file.

        Returns
        -------
        float
            the snowpit's depth.

        """
        
        pit = SnowPilotParser(file_name)
        pit._get_HS()
        
        return pit.HS
        
    def get_pit_layers(self):
        """
        This fuction fetch the pit's layers and their parameters from the pit's
        caamal xml file. It then add thise layer as a Pandas DataFrame to the
        SnowPilotParser object instance.

        Returns
        -------
        None.

        """
        
        
        root = ET.parse(self.file_name).getroot()
        direction = next(root.iter(self.common_tag + 'SnowProfileMeasurements')).attrib['dir']
        profile = next(root.iter(self.common_tag + 'stratProfile'))
        layers = profile.iter(self.common_tag + 'Layer')
        hs = float(next(root.iter(self.common_tag + 'profileDepth'), None).text)
        heights, hardness, grain_type = [], [], []
        
        for layer in layers:
            if direction == 'top down':
                height = hs - float(layer.find(self.common_tag + 'depthTop').text)
                
            else:
                height = float(layer.find(self.common_tag + 'depthTop').text)
            heights.append(height)
            try:
                hrd = layer.find(self.common_tag + 'hardness').text
                hrd = HAND_HARD_2_NUMERIC.get(hrd, np.nan)
            except AttributeError:
                # Maybe split these layers into 2 layers?
                try:
                    hrd_t = layer.find(self.common_tag + 'hardnessTop').text
                    hrd_t = HAND_HARD_2_NUMERIC.get(hrd_t, np.nan)
                    hrd_b = layer.find(self.common_tag + 'hardnessBottom').text
                    hrd_b = HAND_HARD_2_NUMERIC.get(hrd_b, np.nan)
                    hrd = (hrd_b + hrd_t) / 2
                except AttributeError:   
                    hrd = np.nan
            hardness.append(hrd)
            try:
                gp = layer.find(self.common_tag + 'grainFormPrimary').text[:2]
            except AttributeError:
                gp = ''
            grain_type.append(gp)
            
        pit_params = pd.DataFrame(dict(heights=heights, 
                                       hardness=hardness, 
                                       grain_type=grain_type))
            
        self.layers = pit_params


    def get_test_rerults(self):
        """
        This fuction fetch the pit's test results from its caamal xml file and 
        add them as a dictionaty to the SnowPilotParser object instance.
        The test results dictionary contain a dictionary for each test with 
        depth, Fx character, test score, test results (e.g. ECTP), and Cr 
        column length for PST. 

        Returns
        -------
        None.

        """
        
        def try_float(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                return x
        
        test_types = {'CT' : 'ComprTest',
                      'ECT': 'ExtColumnTest',
                      'PST': 'PropSawTest'
                      }
        
        results = {}
        root = ET.parse(self.file_name).getroot()
        tests = next(root.iter(self.common_tag + 'stbTests'))
        for _type, tag in test_types.items():
            for i, test in enumerate(tests.iter(self.common_tag + tag)):
                try:
                    results[f'{_type} {i+1}'] = \
                        {e.tag[len(self.common_tag):]: try_float(e.text) for e in test[0].iter() if e.text.strip() != ''}
                except AttributeError:
                    pass
        self.test_results = results
        

    def get_loc_from_ECT(self):
        """
        This function retrieves the most unstable ECT result in the pit and 
        returns these results. If there is an ECTP result, it returns the 
        results with the lowest number of taps. Otherwise, it returns the 
        ECTN result with the lowest number of taps.
        If there were only ECTX or no ECT at all, it returns results with ECTX.

        Returns
        -------
        loc : dict
            A dictionary with the test with lowest number of taps for ECTP 
            and ECTN result and the falure depth.

        """
        
        if self.test_results is None:
            self.get_test_rerults()
        if self.test_results == {}:
            return {'test': 'ECTX', 'depth': -1}

        ects = {'ECTP': 'ECTX', 'ECTN': 'ECTX'}
        for t, v in self.test_results.items():
            if t.startswith('ECT'):
                if v['testScore'] < ects[v['testScore'][:4]]:
                    result = v['testScore'][:4]
                    ects[result] = v['testScore']
                    ects[f'{result}_depth'] = float(v['depthTop'])
        if ects['ECTP'] != 'ECTX':
            loc = {'test': ects['ECTP'], 'depth': ects['ECTP_depth']}
        else:
            loc = {'test': ects['ECTN'], 'depth': ects['ECTN_depth']}
        
        return loc


    def get_loc_from_PST(self):
        """
        This function looks at all the SPT in a pit, returns the PST with “End” 
        results, and returns the test result with the smallest CutLen to 
        ColumnWidth ratio. If there is not PST test in the pit or no PST End, 
        it returns None.

        Returns
        -------
        scary_psts : dict
            a test result dictionary with fracture type,
                                          cut length,
                                          columns width,
                                          depth of the tested layer.
        """
        
        if self.test_results is None:
            self.get_test_rerults()

        scary_psts = {'fracturePropagation': None,
                      'cutLength': 101, 
                      'columnLength': 100}
        for t, v in self.test_results.items():
            if t.startswith('PST'):
                if v['fracturePropagation'] == 'End':
                    r_c = float(v['cutLength'])
                    c_l = float(v['columnLength'])
                    if r_c / c_l < scary_psts['cutLength'] / scary_psts['columnLength']:
                        scary_psts = v.copy()                        
        if scary_psts['fracturePropagation'] is None:
            return None
        return scary_psts
  
        
    def _get_pit_datetime(self):
        """
        This function fetch the pit's date and time from its caamal xml file 
        an add it to the SnowPilotParser object instance as 

        Returns
        -------
        None.

        """
        
        root = ET.parse(self.file_name).getroot()
        dt = next(root.iter(self.common_tag + 'timePosition')).text
        self.datetime = pd.to_datetime(dt)
        
        
    @classmethod
    def get_pit_datetime(cls, 
                         file_name):
        """
        This class methoud (static) uses _get_pit_datetime to fetch
        the dat and time of a pit from it's caamal xml file

        Parameters
        ----------
        cls : _get_pit_datetime class
            DESCRIPTION.
        file_name : str
            The pit's caamal xml file path.

        Returns
        -------
        Pandas Timestamp
            The pit's date and time.

        """
        
        prof = SnowPilotParser(file_name)
        prof._get_pit_datetime()
        
        return prof.datetime
       


def get_WRF_cell_id(pit, wrf_locations):
    """
    This function find the coresponding WRF grid cell location. It update the 
    pits dictionary with the WRF cell id, and the pit's distance to the cell's 
    center and different in elevation.

    Parameters
    ----------
    pit : dict
        dictionary with snowpit parameteres.
    wrf_locations : Pandas Dataframe
        dataframe containing the WRF cells location information.

    Returns
    -------
    None.

    """
    
    ft_2_meter = 3.28084
    
    loc = [pit['lat'], pit['lon']]
    _ , wtf_cell_id = KDTree(wrf_locations[['lat', 'lon']].values).query(loc)
    wrf_greed_loc = wrf_locations.iloc[wtf_cell_id, :][['lat', 'lon']].to_list()
    dist = geodesic(loc, wrf_greed_loc).meters
    pit['dist_to_cell'] = dist
    try:
        pit['d_elev'] = abs(float(pit['Elevation']) - wrf_locations.iloc[wtf_cell_id, :]['elev'])/ft_2_meter
    except KeyError:
        pit['d_elev'] = 99999
    pit['wrf id'] = wrf_locations.iloc[wtf_cell_id, list(wrf_locations.columns).index('id')]
    pit['zone'] = wrf_locations.iloc[wtf_cell_id, list(wrf_locations.columns).index('zone')]
    
    
def get_map_pro_to_snowPilot_file_paths(df, season='2023'):
    """
    This fuction fetch and map the coresponding SNOWPCK pro file for every 
    SnowPilot pit.

    Parameters
    ----------
    df : Pandas Dataframe
        Dataframe containing the pits geodata and HS.
    season : str
            The first calander year of the winter season

    Returns
    -------
    df : Pandas Dataframe
        The pits Dataframe with additional column contaning that corespoding 
        pro file path for each pit raw.

    """
    
    season = season[:4]
    aspect_id = {'N': 1, 'NE': 1, 'E': 2, 'SE': 3, 'S': 3, 'SW': 3, 'W': 4, 'NW': 1}
    
    df.loc[:, 'wrf id'] = df.loc[:, 'wrf id'].map(lambda x: f'{x:0>6}')
    df.loc[:, 'zone'] = df.loc[:, 'zone'].apply(lambda x: f'{x:0>3}')#astype(str).str.strip().map(lambda x: f'{x:0>3}')
    
    pro_paths = []
    for i, r in df.iterrows():
        zone = r['zone']
        wrf_id = r['wrf id']
        try:
            asp = aspect_id[r['Aspect']]
            path = f'/ssd/snowpack/output.{season}/zone{zone}/{wrf_id}/{wrf_id}{asp}_res.pro'
            pro_paths.append(path)
        except KeyError:
            pro_paths.append(None)
    df.loc[:, 'pro_path'] = pro_paths
    
    return df


if __name__ == "__main__":
  
    
    parser = argparse.ArgumentParser(description='Winter season')
    parser.add_argument('-s', '--season', type=str, default='2023-2024', help='winter season, e.g. 2023-2024')
    
    args = parser.parse_args()

    season = args.season
    SNOW_PIT_PATH = os.path.join(SNOW_PIT_PATH, f'{season}')

    snowpilot_pits_path = os.path.join(SNOW_PIT_PATH, 'snowpits-*-caaml.xml')
    pits_list = glob(snowpilot_pits_path)  
        
    pits_list = [pit for pit in pits_list if pit['HS'] is not None]
    
    wrf_locations = pd.read_csv(WRF_LOCATIONS,
                                usecols=['id', 'zone', 'lat', 'lon', 'elev'])
    for pit in tqdm(pits_list):
        get_WRF_cell_id(pit, wrf_locations)
        
    MAX_DIST_TO_CEL = 1000 #in meteres
    MAX_ELEV_DIFF = 150 #in feet
            
    df = pd.DataFrame.from_records(pits_list)
    df = df.query(f"d_elev < {MAX_ELEV_DIFF} and dist_to_cell < {MAX_DIST_TO_CEL}")
    
    df = get_map_pro_to_snowPilot_file_paths(df, season)
    df.to_csv(os.path.join(WRF_LOCATIONS, f'{season}_pilot_to_pro_map.csv'),index=False)
