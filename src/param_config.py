# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:56:19 2024

@author: Avalanche
"""
import os
import numpy as np
import pandas as pd

HAND_HARD_2_NUMERIC = {'F'  : 1,
                       'F+' : 1.5,
                       '4F-': 1.5,
                       '4F' : 2,
                       '4F+': 2.5,
                       '1F-': 2.5,
                       '1F' : 3,
                       '1F+': 3.5,
                       'P-' : 3.5,
                       'P'  : 4,
                       'P+' : 4.5,
                       'K-' : 4.5,
                       'K'  : 5,
                       'K+' : 5.5,
                       'I'  : 6}

HAND_HARNESS =  {v:k for k,v in HAND_HARD_2_NUMERIC.items()}

# SNOWPILOT conversion:
GRAIN_TYPE_CODE = {1: 'Precipitation particules (PP)',
                   2: 'Decomposing fragmented PP (DF)',
                   3: 'Rounded grains (RG)',
                   4: 'Faceted crystals (FC)',
                   5: 'Depth hoar (DH)',
                   6: 'Surface hoar (SH)',
                   7: 'Melt forms (MF)',
                   8: 'Ice formations (IF)',
                   9: 'Rounding faceted particules (FCxr)'
                   }


GRAIN_TYPE_CODE_S = {1: 'PP',
                     2: 'DF',
                     3: 'RG',
                     4: 'FC',
                     5: 'DH',
                     6: 'SH',
                     7: 'MF',
                     8: 'IF',
                     9: 'FCxr'
                     }

#{k: v[v.find('(')+1:-1] for k, v in GRAIN_TYPE_CODE.items()}

# SnowPilot conversion:
GRAIN_TYPE_COLORS_BY_ID = {1: 'lime',
                           2: 'darkgreen',
                           3: 'pink',
                           4: 'lightblue',
                           5: 'blue',
                           6: 'magenta',
                           7: 'crimson',
                           8: 'crimson',
                           9: 'skyblue'
                           }

GRAIN_TYPE_INDEX_ID = {''    : 0,
                       'PP'  : 1,
                       'DF'  : 2,
                       'RG'  : 3,
                       'FC'  : 4,
                       'DH'  : 5,
                       'SH'  : 6,
                       'MF'  : 7,
                       'FCxr': 8,
                       'MFcr': 9,
                       }

GRAIN_TYPE_COLORS_BY_NAME = {'PP': 'lime',
                             'DF': 'darkgreen',
                             'RG': 'pink',
                             'FC': 'lightblue',
                             'DH': 'blue',
                             'SH': 'magenta',
                             'MF': 'crimson',
                             'FCxr': 'crimson',
                             'MFcr': 'crimson',
                             ''    : 'whitesmoke'
                             }

GRAIN_TYPE_COLORS_BY_NAME = {'PP': 'lime',
                             'DF': 'darkgreen',
                             'RG': 'pink',
                             'FC': 'lightblue',
                             'DH': 'blue',
                             'SH': 'magenta',
                             'MF': 'crimson',
                             'FCxr': 'crimson',
                             'MFcr': 'crimson',
                             ''    : 'whitesmoke'
                             }


GRAIN_TYPE_NAME_TO_COLOR = {'PP'  : 'rbg(0, 255, 0)',
                            'DF'  : 'rbg(34, 139, 34)',
                            'RG'  : 'rbg(255, 182, 193)',
                            'FC'  : 'rbg(173, 216, 230)',
                            'DH'  : 'rbg(0, 0, 255)',
                            'SH'  : 'rbg(255, 0, 255)',
                            'MF'  : 'rbg(255, 0, 0)',
                            'IF'  : 'rbg(255, 0, 0)',
                            'FCxr': 'rbg(0, 255, 255)',
                            ''    : 'rgb(200,200,200)'}



grain_type_similaty_table = np.array(
                            [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                             [0.5, 1. , 0.8, 0.5, 0.2, 0. , 0. , 0. , 0.2, 0. ],
                             [0.5, 0.8, 1. , 0.8, 0.4, 0. , 0. , 0. , 0.4, 0. ],
                             [0.5, 0.5, 0.8, 1. , 0.4, 0.1, 0. , 0. , 0.5, 0. ],
                             [0.5, 0.2, 0.4, 0.4, 1. , 0.5, 0.3, 0. , 0.6, 0. ],
                             [0.5, 0. , 0. , 0.1, 0.5, 1. , 0.9, 0. , 0.4, 0. ],
                             [0.5, 0. , 0. , 0. , 0.3, 0.9, 1. , 0. , 0. , 0. ],
                             [0.5, 0. , 0. , 0. , 0. , 0. , 0. , 1. , 0. , 0.2],
                             [0.5, 0.2, 0.4, 0.5, 0.6, 0.4, 0. , 0. , 1. , 0. ],
                             [0.5, 0. , 0. , 0. , 0. , 0. , 0. , 0.2, 0. , 1. ]])


grain_type_similaty_table = pd.DataFrame(grain_type_similaty_table, 
                                         columns=GRAIN_TYPE_INDEX_ID.keys(), 
                                         index=GRAIN_TYPE_INDEX_ID.keys())

test_grading_score = {'ECTP': 1,
                      'ECTN': 2}

DATA_PATH = os.path.abspath(os.path.join(os.getcwd(), '../data'))
SNOW_PIT_PATH = os.path.join(DATA_PATH, 'SnowPilotData')
WRF_LOCATIONS = os.path.join(DATA_PATH, 'WRF_locations')
RESULTS_PATH = os.path.abspath(os.path.join(os.getcwd(), '../results'))

