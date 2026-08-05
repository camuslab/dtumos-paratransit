import sys
sys.path.append("..")
###
from module.point_generator import point_generator_with_OSM
import pandas as pd 
import numpy as np 
import copy 
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

#########################
# Convert time-standard #
#########################
# - This time-standard of simulation is based on minutes except for YMD(Year-Month-date)
# - HOUR:MINUTE:SECOND -> Minute

def convert_time_standard(operation_record):
    operation_record['ride_time'] = pd.to_datetime(operation_record['ride_time'])

    YMD = list(set(operation_record['ride_time'].dt.strftime('%Y%m%d')))

    target_YMD = min([datetime.strptime(i,'%Y%m%d') for i in YMD])   

    operation_record['ride_time'] = operation_record['ride_time'] - target_YMD
    operation_record['ride_time'] = operation_record['ride_time']/pd.Timedelta(minutes=1)
    operation_record['ride_time'] = np.floor(operation_record['ride_time']).astype('int')
    
    return operation_record, target_YMD


#####################
# Extract Passenger #
#####################
# - Since wheelchair use per passenger is unknown, passenger types are assigned by ratio
# - columns : ['ID', 'ride_time', 'ride_geometry', 'alight_geometry', 'dispatch_time', 'type']
def extract_passenger(operation_record, simulation_inf):
    
    # no change in passenger count
    if not('passenger_increase_ratio' in simulation_inf.keys()):
        # extract passenger data from taxi_operation_record
        passenger = operation_record[['ride_time', 'ride_lat', 'ride_lon', 'alight_lat', 'alight_lon']]

        # generate passenger IDs
        passenger = passenger.reset_index(drop=False)
        passenger = passenger.rename(columns={'index': 'ID'})

        # generate passenger dispatch_time
        passenger['dispatch_time'] = 0 # dispatch_time is the elapsed time until a taxi is secured!

        # assign passenger type (0: no wheelchair, 1: wheelchair)
        type_list = np.random.choice(2 ,size = len(passenger), p=[0.23, 0.77]) ## 0 (no wheelchair): 23%, 1 (wheelchair): 77%
        passenger["type"] = type_list
        
        return passenger
    # passenger count (increase / decrease)
    else:
        passenger = operation_record[['ride_time', 'ride_lat', 'ride_lon', 'alight_lat', 'alight_lon']]
        
        passenger_increase_ratio = simulation_inf['passenger_increase_ratio']
        
        if  passenger_increase_ratio <= 1:        
            passenger = passenger.sample(frac=passenger_increase_ratio).sort_values('ride_time').reset_index(drop=True).reset_index()
            passenger = passenger.rename(columns={'index': 'ID'})
            
            type_list = np.random.choice(2 ,size = len(passenger), p=[0.23, 0.77]) # 0 (no wheelchair): 23%, 1 (wheelchair): 77%
            passenger["type"] = type_list
        else:
            add_passenger = passenger.sample(frac=passenger_increase_ratio-1).copy()
            add_passenger = add_passenger.reset_index(drop=True)
            
            # generate points based on place
            point_generator = point_generator_with_OSM()
            add_passenger_point = point_generator.point_generator_about_placeName(place=simulation_inf['target_region'], count=len(add_passenger) * 2)
            
            ride_point = add_passenger_point[:len(add_passenger)].reset_index(drop=True)
            alight_point = add_passenger_point[len(add_passenger):].reset_index(drop=True)
            
            add_passenger[['ride_lat', 'ride_lon']] = ride_point[['lat', 'lon']] 
            add_passenger[['alight_lat', 'alight_lon']] = alight_point[['lat', 'lon']]           
            
            passenger = pd.concat([passenger, add_passenger]).sort_values('ride_time').reset_index(drop=True).reset_index()
            passenger = passenger.rename(columns={'index': 'ID'})
            
            type_list = np.random.choice(2 ,size = len(passenger), p=[0.23, 0.77]) # 0 (no wheelchair): 23%, 1 (wheelchair): 77%
            passenger["type"] = type_list
        
        # generate passenger dispatch_time
        passenger['dispatch_time'] = 0 # dispatch_time is the elapsed time until a taxi is secured!
        
        '''Revise from here'''
        # reorder passenger columns
        passenger = passenger[['ID', 'ride_time', 'ride_lat', 'ride_lon', 'alight_lat', 'alight_lon', 'dispatch_time', 'type']]
    
        return passenger


################
# Extract Taxi #
################
# -'Create taxi_schedule from the minimum ride time and maximum alight time per vehicle_id.
# - Day-shift drivers (starting before 17:00) work 9 hours
# - Night-shift drivers (starting after 17:00) work 12 hours
def extract_taxi(operation_record, simulation_inf):
    
    if not('taxi_schedule' in simulation_inf.keys()):
        
        taxi_schedule_dict = dict()

        for id, row in operation_record.groupby('vehicle_id'):
            taxi_schedule_dict[id] = [row['cartype'].iloc[0], row['ride_time'].min(), row['ride_time'].max()]

        taxi_schedule = pd.DataFrame(taxi_schedule_dict).T.reset_index()
        taxi_schedule.columns = ['vehicle_id', 'cartype', 'work_start', 'work_end']

        taxi_schedule['temporary_stopTime'] = 0 

        ## create the taxi operation schedule
        bins = [i*60 for i in range(6,31)]
        labels = [i for i in range(6,30)]

        work_startTime = pd.cut(taxi_schedule['work_start'], bins=bins, labels=labels, right=False)
        taxi_schedule['work_start'] = work_startTime.tolist()

        ## Day- and night-shift drivers have different working hours, assigned as below
        # - Group A (day shift): starts before 17:00, works 9 hours: 06:00-17:00
        # - Group B (night shift): starts after 17:00, works 12 hours: remaining hours
        A_group_timeTable = list(range(6,17))

        A_taxi_schedule = taxi_schedule.loc[(taxi_schedule['work_start'].isin(A_group_timeTable))]
        B_taxi_schedule = taxi_schedule.loc[~(taxi_schedule['work_start'].isin(A_group_timeTable))]

        A_taxi_schedule['work_end'] = A_taxi_schedule['work_start'] + 9
        B_taxi_schedule['work_end'] = B_taxi_schedule['work_start'] + 12

        taxi_schedule = pd.concat([A_taxi_schedule, B_taxi_schedule]).reset_index(drop=True)

        ## Since simulation time runs from 6h to 30h, adjust vehicles operating past 30h
        # - e.g., a vehicle working 17h-31h => changed to work 0h-1h and 17h-30h
        taxi_inMorning = taxi_schedule.loc[(taxi_schedule['work_end'] <= 30)]
        taxi_inNight = taxi_schedule.loc[(taxi_schedule['work_end'] > 30)]

        over_time = taxi_inNight['work_end'] - 30
        taxi_inNight['work_end'] = 30

        taxi_inNight_copy = copy.deepcopy(taxi_inNight)
        taxi_inNight_copy['work_start'] = 0 
        taxi_inNight_copy['work_end'] = over_time

        taxi_inNight = pd.concat([taxi_inNight, taxi_inNight_copy])

        taxi_schedule = pd.concat([taxi_inMorning, taxi_inNight]).sort_values('work_start').reset_index(drop=True)

        taxi_schedule['work_start'] = taxi_schedule['work_start'] * 60
        taxi_schedule['work_end'] = taxi_schedule['work_end'] * 60


        point_generator = point_generator_with_OSM()
        taxi_point = point_generator.point_generator_about_placeName(place=simulation_inf['target_region'], count=len(taxi_schedule))
        taxi_schedule['lat'] = taxi_point['lat']
        taxi_schedule['lon'] = taxi_point['lon']
        
        return taxi_schedule
    else:
        ## use user-generated simulation data
        taxi_schedule = simulation_inf['taxi_schedule']
        
        taxi_schedule['temporary_stopTime'] = 0

        point_generator = point_generator_with_OSM()
        taxi_point = point_generator.point_generator_about_placeName(place=simulation_inf['target_region'], count=len(taxi_schedule))
        taxi_schedule['lat'] = taxi_point['lat']
        taxi_schedule['lon'] = taxi_point['lon']        
        
        ## Since simulation time runs from 6h to 30h, adjust vehicles operating past 30h
        # - e.g., a vehicle working 17h-31h => changed to work 0h-1h and 17h-30h
        taxi_inMorning = taxi_schedule.loc[(taxi_schedule['work_end'] <= 30)]
        taxi_inNight = taxi_schedule.loc[(taxi_schedule['work_end'] > 30)]

        over_time = taxi_inNight['work_end'] - 30
        taxi_inNight['work_end'] = 30

        taxi_inNight_copy = copy.deepcopy(taxi_inNight)
        taxi_inNight_copy['work_start'] = 0 
        taxi_inNight_copy['work_end'] = over_time

        taxi_inNight = pd.concat([taxi_inNight, taxi_inNight_copy])

        taxi_schedule = pd.concat([taxi_inMorning, taxi_inNight]).sort_values('work_start').reset_index(drop=True)

        taxi_schedule['work_start'] = taxi_schedule['work_start'] * 60
        taxi_schedule['work_end'] = taxi_schedule['work_end'] * 60
    
        return taxi_schedule


#####################
# Main data extract #
#####################
def extract_main(operation_record, simulation_inf):
    
    operation_record, YMD = convert_time_standard(operation_record)

    passenger = extract_passenger(operation_record, simulation_inf)

    taxi = extract_taxi(operation_record, simulation_inf) 
    
    return passenger, taxi, YMD