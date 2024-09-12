# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 09:35:25 2024

@author: Ron Simenhois
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 16:22:16 2024

@author: Ron Simenhois

This script is a modification of https://gitlabext.wsl.ch/richter/read_profiles/-/tree/main
It reads SNOWPACK model output pro files into a dictionary. It also recalculates
the critical cut length (rc) on flat terrain according to Richter et al.,
2019 (https://doi.org/10.5194/tc-13-3353-2019) since the original implementation
of Gaume et al., 2017 (https://doi.org/10.5194/tc-11-217-2017) resulted in
negative values for slope simulations.
"""

import os
import numpy as np
import pandas as pd
import datetime
import json
from param_config import GRAIN_TYPE_CODE, HAND_HARNESS, WRF_LOCATIONS

# numeric values to grain type name acording to
# https://snowpack.slf.ch/doc-release/html/snowpackio.html
# convertion dictionary


class ProParser():
    
    def __init__(self, file_name):
        
        self.profile = None
        self.file_name = file_name
        self.get_singel_profile = self.get_singel_profile
        self.get_pro_SH = self._get_pro_SH
        

    def read_profile(self,
                     timestamp=None,
                     is3d=False,
                     remove_soil=False,
                     only_header=False,
                     **kwargs):
        """
        This function reads a pro file into a dictionary. The dictionary has two
        keys: the info key with general information and the data key with a profile
        for each run timestamp. Each profile is a dictionary with the profile’s
        parameters as keys and a numpy array with the values for each parameter
        for each layer.
    
        Parameters
        ----------
        timestamp : TYPE: a list of timestamps to asign to the profiles or None, optional
            DESCRIPTION. The default is None. In this case, the script reads the timestamps
                         from the pro file
        is3d : TYPE Bolean, optional, is this an Alp 3D file or a single loaction pro file
            DESCRIPTION. The default is False.
        remove_soil : TYPE, optional
            DESCRIPTION. The default is False.
        only_header : TYPE Bolean, optional to return only header or the whole season snowpack.
            DESCRIPTION. The default is False.
    
        Returns
        -------
        None 
        """
        def try_float(x):
            try:
                return float(x)
            except ValueError:
                return x
    
        location_params = ['Altitude',
                          'Latitude',
                          'Longitude',
                          'SlopeAngle',
                          'SlopeAzi',
                          'StationName']
    
        try:
            with open(self.file_name) as fp:
                content = fp.readlines()
        except FileNotFoundError:
            self.profile = None
            return
        with open('pro_params/param_codes.json') as fp:
            param_codes = json.load(fp)
    
        prof={'info':{}, 'data':{}}
        is_data=False
    
        for line in content:
            if (param := line.strip().split('='))[0] in location_params:
                prof['info'][param[0]] = try_float(param[1].strip())
            elif line.startswith('[DATA]'):
                if only_header: break
                is_data=True
    
            elif line.startswith('0500') and is_data:
                ts = datetime.datetime.strptime( line.strip().split(',')[1],
                                                '%d.%m.%Y %H:%M:%S' )
                prof['data'][ts]={}
            elif line.startswith('0501') and is_data:
                height = np.array(line.strip().split(','))[2:].astype(float)
                if len(height) == 1 and height.item() == 0: continue
                else: prof['data'][ts]['height'] = height
    
            elif is_data:
                if code := param_codes.get(line.strip().split(',')[0], False):
                    prof['data'][ts][code]= np.array([try_float(x) for x in line.strip().split(',') ])[2:]
    
        if is3d:
            self.getCoordinates(self.profile)
        if remove_soil:
            for profile in prof['data'].values():
                try:
                    i_gr = np.where((profile['height']==0))[0].item()
                    gr_offset = {'height': i_gr + 1}
                    for key in profile.keys():
                        profile[key] = profile[key][gr_offset.get(key, i_gr)]
                except (ValueError, KeyError):
                    continue
    
        for ts in prof['data'].keys():
            for var in prof['data'][ts].keys():
                data = prof['data'][ts][var]
                try: prof['data'][ts][var] = np.where((data==-999),np.nan,data)
                except: pass
    
        if timestamp:
            return prof['data'][timestamp]
    
        ts = sorted(prof['data'].keys())
        for dt in ts:
            try:
                pro=prof['data'][dt]
                rc=self.rc_flat(pro)
                prof['data'][dt]['critical_cut_length'] = rc
    
                pendepth = self.comp_pendepth(pro, prof['info']['SlopeAngle'])
                prof['data'][dt]['penetration_depth'] = pendepth
            except:
                continue
    
        self.profile = prof

    def getCoordinates(self,
                       yllcorner=186500,
                       xllcorner=779000,
                       gridsize=1,nx=600,
                       ny=600,
                       dem=None):
        
        
        return self.profile['info']

    """
    Add improved and validated critical cut length for flat field according to Richter et al., 2019
    (https://doi.org/10.5194/tc-13-3353-2019)
    Original formulation of Gaume et al., 2017
    (https://doi.org/10.5194/tc-11-217-2017) resulted in negative values for slope simulations
    In Mayer et al., 2022 (https://doi.org/10.5194/tc-16-4593-2022) flat rc values from slope simulation were ranked highly for computing probality of being unstable
    Also add penetrations depth
    """
    def rc_flat(self, prof):
        """
        This function calculate the critical cut length (rc) on flat terain for each layer
        according to Richter et al., 2019 (https://doi.org/10.5194/tc-13-3353-2019)
        The original formulation of Gaume et al., 2017 (https://doi.org/10.5194/tc-11-217-2017)
        resulted in negative values for slope simulations.
    
        Parameters
        ----------
        pro : dic
            A dictionary containing a single snow profile. The dic's keys are the
            profile parameters and variables are np arrays for the parameter values
            for each layer.
    
        Returns
        -------
        None
    
        """
        rho_ice = 917. #kg m-3
        gs_0 = 0.00125 #m
        # Get the thickness of each layer
        thick = np.diff(np.concatenate((np.array([0]), prof['height'])))
        rho = prof['density']
        gs = prof['grain_size']*0.001 #[m]
        # Get accumulated load on each layer
        rho_sl = np.append(np.flip(np.cumsum(rho[::-1]*thick[::-1])/np.cumsum(thick[::-1]))[1:len(rho)],np.nan)
        tau_p = prof['shear_strength']*1000. #[Pa]
        eprime = 5.07e9*(rho_sl/rho_ice)**5.13 / (1-0.2**2) #Eprime = E' = E/(1-nu**2) ; poisson ratio nu=0.2
        dsl_over_sigman = 1. / (9.81 * rho_sl) #D_sl/sigma_n = D_sl / (rho_sl*9.81*D_sl) = 1/(9.81*rho_sl)
        a = 4.6e-9
        b = -2.
        rc_flat = np.sqrt(a*( rho/rho_ice * gs/gs_0 )**b)*np.sqrt(2*tau_p*eprime*dsl_over_sigman)
        
        return rc_flat
        

    def comp_pendepth(self, prof, slopeangle):
        """
        This function compute the load penertation depth of a skier
    
        Parameters
        ----------
        pro : dic
            A dictionary containing a single snow profile. The dic's keys are the
            profile parameters and variables are np arrays for the parameter values
            for each layer.
        slopeangle : int
            slope angle in degrees.
    
        Returns
        -------
        float, the depth of a skier load
        """
        top_crust = 0
        thick_crust = 0
        rho_Pk = 0
        dz_Pk = 1.e-12
        crust = False
        e_crust = -999
    
        layer_top = prof['height']
        ee = len(layer_top)-1
        thick = np.diff(np.concatenate((np.array([0]), layer_top)))
        rho = prof['density']
        HS = layer_top[-1]
        graintype = prof['grain_type']
        min_thick_crust = 3 #cm
    
        while (ee >= 0) & ((HS-layer_top[ee])<30):
    
            rho_Pk = rho_Pk + rho[ee]*thick[ee]
            dz_Pk = dz_Pk + thick[ee]
    
            if crust == False:
            ##Test for strong mf-crusts MFcr.
            ## Look for the first (from top) with thickness perp to slope > 3cm
                if (graintype[ee] == 772) & (rho[ee] >500.): ## very high density threshold, but implemented as this in SP
                    if e_crust == -999:
                       e_crust = ee
                       top_crust = layer_top[ee]
                       thick_crust = thick_crust + thick[ee]
                    elif (e_crust - ee) <2:
                       thick_crust = thick_crust + thick[ee]
                       e_crust = ee
                elif e_crust > 0:
                   if thick_crust*np.cos(np.deg2rad(slopeangle)) > min_thick_crust:
                       crust = True
                   else:
                       e_crust = -999
                       top_crust = 0
                       thick_crust = 0
    
            ee = ee-1
    
        rho_Pk = rho_Pk/dz_Pk        #average density of the upper 30 cm slab
        return np.min([0.8*43.3/rho_Pk*100, (HS-top_crust)]) #NOTE  Pre-factor 0.8 introduced May 2006 by S. Bellaire , Pk = 34.6/rho_30


    def _get_singel_profile(self,
                            date_time):
        
        def _len(v):
            
            try:
                return len(v)
            except TypeError:
                return -1
            
        if self.profile is None:
            return None
        date_time = pd.to_datetime(date_time)
        data = self.profile['data']
        ts = sorted(data.keys())  
        dt = ts[np.abs(pd.to_datetime(ts) - date_time).argmin()]
        if pd.to_datetime(dt)-date_time <= datetime.timedelta(hours=24):
            profile = data[dt]
            # remove surface grains 
            n_layers = len(profile['height']) 
            profile['grain_type'] = profile['grain_type'][:n_layers]
            profile = {k:v for k,v in profile.items() if _len(v)==n_layers}
            
            return profile
        # Esle
        return None
        

    @classmethod
    def get_singel_profile(cls,  
                           file_name,
                           date_time):
        
        pro = ProParser(file_name)
        pro.read_profile()
        
        return pro._get_singel_profile(date_time)
    

    def _get_pro_SH(self, 
                    date_time):
        if self.profile is None:
            return None, None
        date_time = pd.to_datetime(date_time)
        ts = sorted(self.profile['data'].keys())  
        dt = ts[np.abs(pd.to_datetime(ts) - date_time).argmin()]
        try:
            if pd.to_datetime(dt)-date_time <= datetime.timedelta(hours=24):
                hs = self.profile['data'][dt]['height'][-1]
            else:
                hs = 0.01
        except KeyError: #the pit is empty / melted
            hs = np.nan 
        return hs, dt
    
    
    @classmethod
    def get_pro_SH(cls, 
                   file_name,
                   date_time):
        
        pro = ProParser(file_name)
        pro.read_profile()
        
        return pro._get_pro_SH(date_time)
    
    
    def get_sk38(self):
        
        date_time = []
        sk38 = []
        height = []
        
        for dt, prof in self.profile['data'].items():
            date_time.append(dt)
            try:
                idx = np.argmin(prof['sk38'])
                sk38.append(min(prof['sk38'])) 
                height.append(prof['height'][idx])
            except KeyError:
                sk38.append(None) 
                height.append(None)
                
        sk38_vals = pd.DataFrame({'datetime': date_time,
                                  'sk38': sk38,
                                  'height': height}).set_index('datetime')
        
        return sk38_vals
                
            
    def get_ssi(self):
        
        date_time = []
        ssi = []
        height = []
        
        for dt, prof in self.profile['data'].items():
            date_time.append(dt)
            try:
                idx = np.argmin(prof['ssi'])
                ssi.append(min(prof['ssi'])) 
                height.append(prof['height'][idx])
            except KeyError:
                ssi.append(None) 
                height.append(None)
                
        ssi_vals = pd.DataFrame({'datetime': date_time,
                                 'ssi': ssi,
                                 'height': height}).set_index('datetime')
        
        return ssi_vals
            
    def get_rc(self):
        
        date_time = []
        rc = []
        height = []
        
        for dt, prof in self.profile['data'].items():
            date_time.append(dt)
            try:
                rcs = prof['critical_cut_length']
                rcs[np.isnan(rcs)] = rcs[~np.isnan(rcs)].max() + 1
                idx = np.argmin(rcs)
                rc.append(min(prof['critical_cut_length'])) 
                height.append(prof['height'][idx])
            except (KeyError, ValueError):
                rc.append(None) 
                height.append(None)
                
        rc_vals = pd.DataFrame({'datetime': date_time,
                                 'critical_cut_length': rc,
                                 'height': height}).set_index('datetime')
        
        return rc_vals
            
        
    def get_relative_threshold_sum(self):
        """
        Returns the Relative Threshold Sum approach  (RTA) weak layer. 
        This is according to Monti, Fabiano, and Jürg Schweizer, "A relative difference 
        approach to detect potential weak layers within a snow profile", 2013, Proceedings ISSW.
        
        Parameters
        ----------
        self : ProParcer object

        Returns
        -------
        false if error, a lise with RTA value for each layer otherwise.

        """
        def norm(a):
            try:
                if (a_range:= a.max() - a.min()) > 0:
                    return (a - a.min()) / a_range
                else:
                    return np.zeros_like(a)
            except ValueError:
                return np.zeros(1)
            
        if self.profile is None:
            self.read_profile()
            
        profile = self.profile['data']
        for dt, prof in profile.items():
            size = len(prof.get('height', []))
            if size == 0:
                continue
            hs = prof['height'][-1]
            height = hs - prof['height']
            height_norm = norm(height)
            grain_size = prof['grain_size']
            grain_size_norn = norm(grain_size)
            d_grain_size = np.diff(grain_size)
            d_grain_size_norm = norm(d_grain_size)
            hard = np.abs(prof['hand_hardness'])
            hard_norm = 1 - norm(hard)
            d_hard = np.diff(hard)
            d_hard_norm = norm(d_hard)
            # grain types receive a score depending on their primary and secondary forms
            primary = prof['grain_type'] // 100
            secondary = prof['grain_type'] / 10 // 100
            primary_is_persistent = np.where((primary==4) | (primary==5) | (primary==6) | (primary==9), 1, 0)
            secondary_is_persistent = np.where((secondary==4) | (secondary==5) | (secondary==6) | (secondary==9), 1, 0)
            grain_type = (primary_is_persistent + secondary_is_persistent) / 2 
            grain_type_norm = norm(grain_type[:-1])
                
            rta = height_norm + grain_size_norn + d_grain_size_norm + hard_norm + d_hard_norm + grain_type_norm
            
            self.profile['data'][dt]['rta'] = rta
            
    
    def get_weakest_persistent_layer(self):
                
        if self.profile is None:
            self.read_profile()
 
        date_time = []
        pw = []
        height = []

        for dt, prof in self.profile['data'].items():
            date_time.append(dt)    
            size = len(prof.get('height', []))
            if size == 0:
                height.append(None)
                pw.append(None)
                continue
            grain_types = prof['grain_type'][:-1]
            primary = grain_types // 100
            secondary = grain_types / 10 // 100
            primary_is_persistent = np.where((primary==4) | (primary==5) | (primary==6) | (primary==9), 1, 0)
            secondary_is_persistent = np.where((secondary==4) | (secondary==5) | (secondary==6) | (secondary==9), 1, 0)
            grain_type = (primary_is_persistent + secondary_is_persistent) / 2 
            hard = np.exp(5 - np.abs(prof['hand_hardness']))
            loc = np.multiply(grain_type, hard)  
            
            self.profile['data'][dt]['presistant_weakeness'] = loc
            
            height.append(prof['height'][np.argmax(loc)])
            pw.append(max(loc))
            
        return(pd.DataFrame({'datetime': date_time,
                             'weakest_presistant_layer': pw,
                             'height': height}).set_index('datetime'))
                    

if __name__ == '__main__':
    
    from glob import glob
    from tqdm import tqdm
    import argparse
    
    parser = argparse.ArgumentParser(description='Winter season')
    parser.add_argument('-s', '--season', type=str, default='2023-2024', 
                        help='winter season, e.g. 2022-2023, 2023-2024')
    
    args = parser.parse_args()
    season = args.season

    import datetime
    st_id='081687'
        
    aspect_paths = {'none' :'',
                    'north': 1,
                    'east' : 2,
                    'south': 3,
                    'west' : 4}
    
    
    aspect = 'none'
    name_mask = f'../data/snowpack/output.2023/**/**{st_id}{aspect_paths[aspect]}_res.pro'
    compare_paths = glob(name_mask, recursive=True)
    snpk = ProParser(compare_paths[1])    
    snpk.read_profile()
    print("done!")
    
    
    
    # df = pd.read_csv(os.path.abspath(os.path.join(os.getcwd(), '../data', 'WRF_locations', 
    #                                               f'{season}_pilot_to_pro_map.csv')),
    #                  parse_dates=['PRO_DateTime'], 
    #                  date_parser=lambda t: pd.to_datetime(t)
    #                  )
    # pro_hs, pro_dt = [], []
    
    # for _, r in tqdm(df.iterrows()):
    #     break
    #     hs, dt = ProParser.get_pro_SH(r['pro_path'], r['Time'])
    #     pro_hs.append(hs)
    #     pro_dt.append(dt)

    # df['PRO HS'] = pro_hs  
    # df['PRO_DateTime'] = pro_dt
    
    # df.to_csv(os.path.join(WRF_LOCATIONS, 'pilot_vs_pro.csv'))
    

