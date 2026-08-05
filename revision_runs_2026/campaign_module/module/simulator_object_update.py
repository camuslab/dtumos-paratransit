from .simulator_helper import save_json_data
import pandas as pd
import numpy as np 


###########################
# Passenger status update #
###########################
# - 1. extract active (requested) data from passenger
# - 2. switch passenger to failed status once the configured fail_time is reached, otherwise dispatch_time +1
def update_passenger(requested_passenger, fail_passenger, passenger, simul_configs, time):
    
    fail_time = simul_configs['fail_time']
    save_path = simul_configs['save_path']
    
    current_requested_passenger = passenger.loc[(passenger['ride_time'] == time)]
    passenger = passenger.loc[(passenger['ride_time'] != time)]
    passenger = passenger.reset_index(drop=True)
    
    if len(requested_passenger) > 0:
        
        requested_passenger['dispatch_time'] = requested_passenger['dispatch_time'] + 1
        
        current_fail_passenger = requested_passenger.loc[(requested_passenger['dispatch_time'] >= fail_time)]
        fail_passenger= pd.concat([fail_passenger, current_fail_passenger])
        fail_passenger = fail_passenger.reset_index(drop=True)
        
        if len(current_fail_passenger) > 0:             
            
            requested_passenger = requested_passenger.loc[(requested_passenger['dispatch_time'] < fail_time)]
            
            current_fail_passenger = [{'passenger_id':row['ID'], 'status':0,
                                       'location': [row['ride_lon'], row['ride_lat']], 
                                       'timestamp':[row['ride_time'], row['ride_time'] + row['dispatch_time']]}\
                                           for _, row in current_fail_passenger.iterrows()]            
            
            save_json_data(current_fail_passenger, save_path, file_name='passenger_marker')
            del current_fail_passenger
    
    
    requested_passenger = pd.concat([requested_passenger, current_requested_passenger])
    requested_passenger = requested_passenger.reset_index(drop=True)
    
    return requested_passenger, fail_passenger, passenger

#########################
# Vehicle status update #
#########################
# active_vehicle, empty_vehicle columns : ['vehicle_id', 'work_end', 'temporary_stopTime', 'geometry', 'P_ID', 'P_ride_geometry', 'P_alight_geometry', 'P_disembark_time']
# - 1. check vehicles starting work
# - 2. check passenger drop-offs for vehicles in service
# - 3. check vehicles ending work
def update_vehicle(active_vehicle, empty_vehicle, vehicle, simul_configs, time):
    
    save_path = simul_configs['save_path']
    
    # check work start
    current_start_vehicle = vehicle.loc[(vehicle['work_start'] == time)]
    
    if len(current_start_vehicle) > 0:
        
        if 'cartype' in current_start_vehicle.columns:
            current_start_vehicle = current_start_vehicle[['vehicle_id', 'cartype', 'work_end', 'temporary_stopTime', 'lat', 'lon']]
        else:
            current_start_vehicle = current_start_vehicle[['vehicle_id', 'work_end', 'temporary_stopTime', 'lat', 'lon']]
        
        current_start_vehicle['temporary_stopTime'] = time
        current_start_vehicle['P_ID'] = np.nan
        current_start_vehicle['P_ride_lat'] = np.nan
        current_start_vehicle['P_ride_lon'] = np.nan
        current_start_vehicle['P_alight_lat'] = np.nan
        current_start_vehicle['P_alight_lon'] = np.nan
        current_start_vehicle['P_request_time'] = np.nan
        current_start_vehicle['P_disembark_time'] = np.nan
        
        empty_vehicle = pd.concat([empty_vehicle, current_start_vehicle])
        empty_vehicle = empty_vehicle.reset_index(drop=True)
        
        vehicle = vehicle.loc[(vehicle['work_start'] != time)]
        vehicle = vehicle.reset_index(drop=True)
        
    # check passenger drop-off
    if len(active_vehicle) > 0:
        
        current_empty_vehicle = active_vehicle.loc[(active_vehicle['P_disembark_time'] <= time)].copy()
        
        if len(current_empty_vehicle) > 0:
            # update vehicles that just dropped off passengers
            current_empty_vehicle['lat'] = current_empty_vehicle['P_alight_lat']
            current_empty_vehicle['lon'] = current_empty_vehicle['P_alight_lon']
            current_empty_vehicle['temporary_stopTime'] = current_empty_vehicle['P_disembark_time']
            
            current_empty_vehicle['P_ID'] = np.nan
            current_empty_vehicle['P_ride_lat'] = np.nan
            current_empty_vehicle['P_ride_lon'] = np.nan
            current_empty_vehicle['P_alight_lat'] = np.nan
            current_empty_vehicle['P_alight_lon'] = np.nan
            current_empty_vehicle['P_disembark_time'] = np.nan
        
            empty_vehicle = pd.concat([empty_vehicle, current_empty_vehicle])
            empty_vehicle = empty_vehicle.reset_index(drop=True)
            
            # update vehicles still in service
            active_vehicle = active_vehicle.loc[(active_vehicle['P_disembark_time'] > time)]
            active_vehicle = active_vehicle.reset_index(drop=True)
            
    # check work end
    if len(empty_vehicle) > 0: 
        
        end_vehicle = empty_vehicle.loc[(empty_vehicle['work_end'] < time+5)] 
        end_vehicle = end_vehicle.loc[(end_vehicle['temporary_stopTime'] != time)]        
    
        empty_vehicle = empty_vehicle.loc[(empty_vehicle['work_end'] >= time+5)]   
        empty_vehicle = empty_vehicle.reset_index(drop=True)
        
        if len(end_vehicle) > 0:
            
            # when temporary_stopTime is NaN, the vehicle is off duty but ended work right away during relocation, so no vehicle marker is needed
            if 'cartype' in current_start_vehicle.columns:
                end_vehicle = [{'vehicle_id':row['vehicle_id'], 'cartype':row['cartype'],
                                'location': [row['lon'], row['lat']], 
                                'timestamp':[row['temporary_stopTime'], time]}\
                                    for _, row in end_vehicle.iterrows() if ~(np.isnan(row['temporary_stopTime']))]            
            else:
                end_vehicle = [{'vehicle_id':row['vehicle_id'],
                                'location': [row['lon'], row['lat']], 
                                'timestamp':[row['temporary_stopTime'], time]}\
                                    for _, row in end_vehicle.iterrows() if ~(np.isnan(row['temporary_stopTime']))]  
            
            save_json_data(end_vehicle, save_path, file_name='vehicle_marker')
            del end_vehicle
    
    return active_vehicle, empty_vehicle, vehicle