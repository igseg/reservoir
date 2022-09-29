import pandas as pd
import numpy  as np
from os import listdir
from os.path import isfile, join
import re
import statistics
import matplotlib.pyplot as plt
import seaborn as sns


class dam_data:
    def __init__(self, cap_max = 0, cap_min = 0, dam_height = 0, area_skm = 0, river = 0, near_city = 0, dam_name = None, main_use = None):
        self.cap_max    = cap_max
        self.cap_min    = cap_min
        self.dam_height = dam_height
        self.area_skm   = area_skm
        self.river      = river
        self.near_city  = near_city
        self.name       = dam_name
        self.main_use   = main_use
        
    def __str__(self):
           return f'id is: {self.id}\n name is: {self.name}\n main use is: {self.main_use}\n maximum capacity is: {self.cap_max}\n minimum capacity is: {self.cap_min}\n height of the dam is: {self.dam_height}\n area is: {self.area_skm}\n river is: {self.river}\n nearest city is: {self.near_city}'
        
def add_year_month_database(data):

    for idx in data.index:
        date = data.loc[idx,'date']
        date = date.split('-')

        if np.isnan(data.inflow[idx]):
            continue

        data.loc[idx,'yearmonth'] = date[0] + date[1]
        
# Get every filename of the dataseries path and separates its  name by '_' and '.' leaving the DAM_ID as the element 1.


def get_time_series_ids():

    mypath = 'time_series_all/'

    file_names = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f[-3:]=='csv']
    file_names.remove('.csv')
    aux = file_names.copy()
    file_names = [f.split('_') for f in file_names]
    for idx in range(len(file_names)):
        items = [k.split('.') for  k in file_names[idx]]
        result = []
        for item in items:
            for ite in item:
                result.append(ite)
        file_names[idx] = result
    
    file_ids = [eval(f[1]) for f in file_names]
    return file_ids

# Go from dam_id to dam_name

def dam_id_to_name(dam_id):
    reservoir_data = pd.read_csv('attributes/reservoir_attributes.csv')

    idx_reservoir = np.where(reservoir_data['DAM_ID'] == dam_id)[0][0]
    dam_name      = reservoir_data.loc[idx_reservoir, 'DAM_NAME']
    
    return dam_name

# Go from dam_name to its use

def dam_name_to_data(dam_name):

    dams_data = pd.read_csv('GRanD_dams_v1_1.csv', delimiter=';')

    try: idx_dam = np.where(dams_data['DAM_NAME'] == dam_name)[0][0]
    except:
        return None
    
    main_use = dams_data.loc[idx_dam, 'MAIN_USE']
    
    cap_max    = float(dams_data.loc[idx_dam, 'CAP_MAX'].replace(',', '.'))
    cap_min    = float(dams_data.loc[idx_dam, 'CAP_MIN'].replace(',', '.'))
    dam_height = float(dams_data.loc[idx_dam, 'DAM_HGT_M'])
    area_skm   = float(dams_data.loc[idx_dam, 'AREA_SKM'].replace(',', '.'))
    river      = dams_data.loc[idx_dam, 'RIVER']
    near_city  = dams_data.loc[idx_dam, 'NEAR_CITY']

    
    return dam_data(cap_max=cap_max, cap_min=cap_min, dam_height=dam_height, area_skm = area_skm
                   , river = river, near_city= near_city, dam_name= dam_name, main_use=main_use)


def from_id_to_dam_data(dam_id):
    
    name = dam_id_to_name(dam_id)
    
    dam  = dam_name_to_data(name)
    
    if dam == None:
        return None
    
    dam.id = dam_id
    return dam

def from_id_to_time_series(dam_id):
    
    return 'time_series_all/ResOpsUS_'+str(dam_id)+'.csv'

def id_to_series_data(dam_id):

    path = from_id_to_time_series(dam_id)

    ts = pd.read_csv(path)

    inflows  = ts.inflow.values
    outflows = ts.outflow.values

    isnan_inflow  = ~np.isnan(inflows)
    isnan_outflow = ~np.isnan(outflows)

    isnan = isnan_outflow * isnan_inflow

    inflows  = inflows[isnan]
    outflows = outflows[isnan]

    correlation = np.corrcoef(inflows,outflows)[1,0]

    nan_data = np.sum(ts.isna())
    num_rows = ts.shape[0]
    
    return correlation, nan_data, num_rows    

def month_num_to_short(month):
    if month == '01':
        month = 'Jan-'
    elif month == '02':
        month = 'Feb-'
    elif month == '03':
        month = 'Mar-'
    elif month == '04':
        month = 'Apr-'
    elif month == '05':
        month = 'May-'
    elif month == '06':
        month = 'Jun-'
    elif month == '07':
        month = 'Jul-'
    elif month == '08':
        month = 'Aug-'
    elif month == '09':
        month = 'Sep-'
    elif month == '10':
        month = 'Oct-'
    elif month == '11':
        month = 'Nov-'
    elif month == '12':
        month = 'Dec-'
        
    return month

def yearmonth_year(yearmonth):
    return str(yearmonth)[:4]


def yearmonth_month(yearmonth):
    return str(yearmonth)[4:]


def from_index_to_dates(yearmonths):
    times = []
    
    for date in yearmonths:
        year  = yearmonth_year( date)
        month = yearmonth_month(date)
        
        month = month_num_to_short(month)
        date  = month+year
        
        times.append(date)
        
    return times