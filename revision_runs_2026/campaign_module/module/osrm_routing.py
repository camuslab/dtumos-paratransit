# library
from .utils import calculate_straight_distance
import time as _rt_time
import numpy as np
import itertools
import requests
import os as _os
import polyline
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import warnings 
warnings.filterwarnings('ignore')

#############
# OSRM base #
#############
def get_res(point):

   status = 'defined'

   global _keepalive_session
   try:
      session = _keepalive_session
   except NameError:
      session = requests.Session()
      retry = Retry(connect=10, backoff_factor=1)
      adapter = HTTPAdapter(max_retries=retry)
      session.mount('http://', adapter)
      session.mount('https://', adapter)
      _keepalive_session = session

   overview = '?overview=full'
   loc = f"{point[1]},{point[0]};{point[3]},{point[2]}" # lon, lat, lon, lat
   url = "http://127.0.0.1:" + _os.environ.get("OSRM_PORT", "5001") + "/route/v1/driving/"
   # url = 'http://router.project-osrm.org/route/v1/driving' # OSRM docker 없을때 사용
   r = session.get(url + loc + overview, timeout=60) 
   
   if r.status_code!= 200:
      
      # 400 = NoRoute/NoSegment (결정론적, 미세구간 스냅 문제) → fallback 사용
      # 그 외 = 일시적 오류 → 호출측에서 재시도
      status = 'noroute' if r.status_code == 400 else 'undefined'
      
      # distance    
      distance = calculate_straight_distance(point[0], point[1], point[2], point[3]) * 1000
      
      # route
      route = [[point[1], point[0]], [point[3], point[2]]]

      # duration & timestamp
      speed_km = 10#km
      speed = (speed_km * 1000/60)      
      duration = distance/speed
      
      timestamp = [0, duration]
         
      result = {'route': route, 'timestamp': timestamp, 'duration': duration, 'distance' : distance}
   
      return result, status
   
   res = r.json()   
   
   return res, status

##############################
# Extract duration, distance #
##############################
def extract_duration_distance(res):
   
   duration = res['routes'][0]['duration']/60
   distance = res['routes'][0]['distance']
   
   return duration, distance

#################
# Extract route #
#################
def extract_route(res):
    
    route = polyline.decode(res['routes'][0]['geometry'])
    route = list(map(lambda data: [data[1],data[0]] ,route))
    
    return route

#####################
# Extract timestamp #
#####################
def extract_timestamp(route, duration):
    
    rt = np.array(route)
    rt = np.hstack([rt[:-1,:], rt[1:,:]])

    per = calculate_straight_distance(rt[:,1], rt[:,0], rt[:,3], rt[:,2])
    per = per / np.sum(per)

    timestamp = per * duration
    timestamp = np.hstack([np.array([0]),timestamp])
    timestamp = list(itertools.accumulate(timestamp)) 
    
    return timestamp
 
########
# MAIN #
########
# - input : O_point, D_point (shapely.geometry.Point type)
# - output : trip, timestamp, duration, distance
_osrm_prof = {"n": 0, "sec": 0.0}
import atexit as _atexit
def _dump_prof():
    if _os.environ.get("OSRM_PROF") and _osrm_prof["n"]:
        with open(_os.environ["OSRM_PROF"], "a") as f:
            f.write(f"{_osrm_prof['n']},{_osrm_prof['sec']:.2f}\n")
_atexit.register(_dump_prof)

def osrm_routing_machine(OD_coords):
   # O_lat, O_lon, D_lat, D_lon = OD_coords
   # 일시적 비-200/연결오류는 재시도 (성공 결과는 기존과 동일 → 비의미적 하드닝)
   _pt0 = _rt_time.perf_counter()
   for _attempt in range(6):
      try:
         osrm_base, status = get_res(OD_coords)
      except Exception:
         status = 'undefined'
      if status == 'defined':
         break
      if status == 'noroute':
         import sys as _sys
         print(f"NOROUTE fallback: {OD_coords}", file=_sys.stderr)
         return osrm_base  # get_res가 만든 직선 보간 결과 (기존 엔진의 fallback 의도 복원)
      _rt_time.sleep(0.3 * (_attempt + 1))
   else:
      raise RuntimeError(f"OSRM unreachable after retries: {OD_coords}")

   _osrm_prof["n"] += 1
   _osrm_prof["sec"] += _rt_time.perf_counter() - _pt0
   if status == 'defined':
      duration, distance = extract_duration_distance(osrm_base)
      route = extract_route(osrm_base)
      timestamp = extract_timestamp(route, duration)
      
      result = {'route': route, 'timestamp': timestamp, 'duration': duration, 'distance' : distance}
      
      if np.isnan(result['timestamp'][-1]):
         result['timestamp'][-1] = 0.01
         result['duration'] = 0.01
         
      return result
   else: 
      return None

