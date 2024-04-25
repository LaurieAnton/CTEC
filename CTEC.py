# Computation
import math
import numpy as np
import pandas as pd
from scipy.stats import norm
# Datetime handling
from collections import Counter
from datetime import datetime, timedelta
# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# For reproducability
np.random.seed(42)
verbose = True

# IMPORTANT: PROPERTIES ARE UPDATED IN THE ORDER THEY ARE CALLED.
#            IF ONE PROPERTY DEPENDS ON ANOTHER, MAKE SURE THE
#            INDEPENDENT PROPERTY IS CALLED FIRST. IT IS BEST TO
#            CALL (AND UPDATE) PROPERTIES ONCE PER CALCULATION,
#            AT THE TOP OF A FUNCTION, IN THE CORRECT ORDER.

# avg_vehicle_length, avg_vehicle_buffer, freeflow_speed
class trafficBehaviour:
    """
    Idea for later: In freeflow, allow multiple trafficBehaviours. In congestion, do not.
    """
    def __init__(self, 
                 avg_vehicle_length: float = 4.5,
                 avg_vehicle_buffer: float = 3.6,
                 freeflow_speed: float = 90):
        '''
        A class that carries the vehicle behaviour of the traffic. In this implementation, only one class of traffic behaviour is permitted, while multiple classes are allowed for describing electical energy propagation, as the coupling only goes one way.
        
        avg_vehicle_length: float (m) -> The average vehicle length in the traffic.

        avg_vehicle_buffer: float (m) -> The average buffer (vehicle seperation when stopped on the road).

        freeflow_speed: float (km/h) -> The speed at which the traffic will flow if not impeded.

        eff_vehicle_length: property (m) -> The sum of the average vehicle and buffer lengths.

        jam_density: property (vehicle/km) -> The maximum number of vehicles that can fit on a road given the effectve vehicle length.

        time_gap: property (h) -> The time between the departure of the one vehicle and the arrival of another at a test location.

        traffic_wavespeed: property (km/h) -> The speed at which information travels backwards when faced with a traffic jam.

        critical_density: property (vehicles/km) -> The density at which traffic starts to slow down due to the density on the road. Also the point of maximum traffic flow.
        '''
        self.avg_vehicle_length = avg_vehicle_length
        self.avg_vehicle_buffer = avg_vehicle_buffer
        self.freeflow_speed = freeflow_speed

        # Used in calculating the time gap of the vehicleType
        # Assumed shaped of timeGap(v): ~ v (linear)
        v1 =  50; T1 = 1.2/(60**2)   # 1.2 s at  50 km/h
        v2 = 120; T2 = 1.4/(60**2)   # 1.4 s at 120 km/h
        global m, b 
        m = (T2-T1)/(v2-v1)         # Slope
        b = T1-m*v1                 # Intercept

    @property
    def eff_vehicle_length(self): # m
        return self.avg_vehicle_length + self.avg_vehicle_buffer
    
    @property
    def jam_density(self): # vehicle/km
        return 1000/self.eff_vehicle_length 
    
    @property
    def time_gap(self): # h
        # Used in calculating the wave speed, traffic_wavespeed = eff_vehicle_length/time_gap
        # Based on the following video:
        # https://www.youtube.com/watch?v=VwQplBZBQJs&list=LL&index=5&t=1214s
        # Assumes linear relationship between two points, (v1,T1) and (v2,T2).
        return m*self.freeflow_speed+b   # Linear output

    @property
    def traffic_wavespeed(self): # km/h
        return (self.eff_vehicle_length/1000)/self.time_gap
    
    @property
    def critical_density(self): # vehicle/km
        return self.traffic_wavespeed*self.jam_density/(self.freeflow_speed+self.traffic_wavespeed) 
    
    # battery_capacity, discharge_rate_freeflow, discharge_rate_jam, vehicleType_ID

class vehicleType:
    # Used to default count the number of instances of this class.
    instance_count = 0
    def __init__(self, 
                 battery_capacity: float = None,
                 discharge_rate_freeflow: float = 0, 
                 discharge_rate_jam: float = 0,
                 vehicleType_ID: str = None):
        '''
        A class that carries the vehicle powertrain type, used in the multiclass nature of this simulator. For each instance of a vehicleType and stationType, an initial-boundary road density solution space is created. Defaults to an internal combustion engine (ICE) type with zero numeric values.
        
        battery_capacity: float (kWh) -> The average battery size in kWh.

        discharge_rate_freeflow: float (kW) -> NEGATIVE. The average discharge rate when the vehicle is moving at the freeflow speed of underlying traffic behaviour. 

        discharge_rate_jam: float (kW) -> NEGATIVE. The average discharge rate when the car is stopped in traffic.

        vehicleType_ID: str (UL) -> Used to identify each instance of vehicleType.
        '''
        self.battery_capacity = battery_capacity
        self.discharge_rate_freeflow = discharge_rate_freeflow
        self.discharge_rate_jam = discharge_rate_jam

        # If the user does not set an ID, assign it a default value.
        self.__class__.instance_count += 1
        if vehicleType_ID == None:
            self.vehicleType_ID = f'vehicleType {self.__class__.instance_count:02}'
        else:
            self.vehicleType_ID = vehicleType_ID

# off_ramp_location, on_ramp_location, vehicle_capacity, max_charging_rate, vehicleType_ID, battery_capacity, stationType_ID
class stationType:
    # Used to default count the number of instances of this class. Format XX
    instance_count = 0
    def __init__(self, 
                 vehicleType: vehicleType, # One vehicleType per stationType
                 off_ramp_location: float, 
                 on_ramp_location: float, 
                 max_charging_rate: float, # Flat rate until 80% SoC, then tapers to 0%
                 vehicle_capacity: int = math.inf,
                 max_ramp_flow: float = math.inf,
                 stationType_ID: str = None):
        '''
        A class that carries the charging station type, used in the multiclass nature of this simulator. For each instance of a vehicleType and stationType, an initial-boundary road density solution space is created.

        vehicleType: vehicleType (class) -> The vehicleType instance that the station services.
        
        off_ramp_location: float (km) -> The distance from the start of the road to the off-ramp station entrance.

        on_ramp_location: float (km) -> The distance from the start of the road to the station on-ramp to the road.

        vehicle_capacity: float (vehicles) -> The maximum number of vehicles the station can service at any time. Note that each station instance services one vehicleType.

        max_charging_rate: float (kW) -> The maximum (nameplate) charging rate.

        max_ramp_flow: float (vehicles/h) -> The maximum flow rate at which vehicles can exit the station. Optional.

        stationType_ID: str (UL) -> Used to identify each instance of stationType. Must be a unique ID per stationType instance.
        '''
        # Copy attributes from vehicleType
        self.vehicleType_ID = vehicleType.vehicleType_ID
        self.battery_capacity = vehicleType.battery_capacity
        self.discharge_rate_freeflow = vehicleType.discharge_rate_freeflow
        self.discharge_rate_jam = vehicleType.discharge_rate_jam
        # Assign stationType attributes
        self.off_ramp_location = off_ramp_location
        self.on_ramp_location = on_ramp_location
        self.vehicle_capacity = vehicle_capacity
        self.max_ramp_flow = max_ramp_flow
        self.max_charging_rate = max_charging_rate

        if vehicleType.battery_capacity == None:
            raise TypeError('Only stations that service electric vehicles are modeled. Ensure your vehicleType has a non-zero plug-in battery capacity.')

        # If the user does not set an ID, assign it a default value. Format XX
        self.__class__.instance_count += 1
        if stationType_ID == None:
            self.stationType_ID = f'stationType {self.__class__.instance_count:02}'
        else:
            self.stationType_ID = stationType_ID

# roadStart, roadStep, roadEnd, vehicleType, stationList, stationSoCGlobalStep, densityField, SoCField
class roadScenario: 
    def __init__(self,
                 trafficBehaviour: trafficBehaviour, 
                 road_start: float = 0,
                 road_end: float = 100):
        '''
        The main scenario class that loads the trafficBehaviour, all vehicleTypes, stationTypes, and input data. For each instance of a vehicleType and stationType, an initial-boundary road density solution space is created when loaded. Temporal standardization is necessary prior to loading into the simulator.

        trafficBehaviour: trafficBehaviour (class) -> Defines how the traffic flows.
        
        road_start: float (km) -> The starting km mark.

        road_end: float (km) -> The final km mark.

        station_global_SoC_resolution: float (kWh) -> The SoC-step resolution for all charging stations.
        '''
        self.trafficBehaviour = trafficBehaviour
        self.road_start = road_start
        self.road_end = road_end

        # Initialize a dictionary to store the current state of each vehicleType
        self.vehicleType_dict = {}

        # Initialize a dictionary to store the current state of each stationType
        self.stationType_dict = {}

    def load_vehicleType(self, 
                         vehicleType: vehicleType,
                         boundary_injection_path: str = None,
                         boundary_df_input: pd.DataFrame = None,
                         assigned_by_stationType: bool = False): 
        '''
        Add a vehicleType to the road scenario. Each vehicle type is given the option to customize an initial density profile (field) on the road, and a boundary condition for vehicle injections at timestep, dt.

        vehicleType: vehicleType (class) -> A vehicleType instance.

        boundary_injection_path: str (UL) -> The file path to the boundary condition csv file for vehicleType injections at timestep, dt. Expected format is a CSV file with the first row being datetime values, the second row being vehicles/h, and the third row being average SoC, kWh/h.  Defaults to an empty profile. A header row is also expected, with column names: 'Timestamp', 'vehicles/h', 'kWh/h'.
        '''

        # Load the vehicleType in the dictionary
        self.vehicleType_dict[vehicleType.vehicleType_ID] = {'object':vehicleType}

        if not assigned_by_stationType:
            # Needed for a later check to ensure no boundary inputs are used for stationType assigned instances
            self.vehicleType_dict[vehicleType.vehicleType_ID]['assigned_by'] = 'user'

            if boundary_df_input is not None:
                boundary_injection_df = boundary_df_input
            else:
                try:
                    # Read the CSV file, ensuring headers are used as column names
                    boundary_injection_df = pd.read_csv(boundary_injection_path)
                except:
                    if boundary_injection_path == None:
                        ValueError(f'No boundary injection CVS file or boundary_df_input specified for {vehicleType.vehicleType_ID}. Please load one.')
                    else:
                        ValueError(f'Boundary injection CVS file could not be loaded for {vehicleType.vehicleType_ID}. Make sure the CSV file is properly formatted.') 
            # Format such that all quantities are in hours or per hour
            boundary_injection_df, time_bin = format_boundary_df(boundary_injection_df)
            
            self.vehicleType_dict[vehicleType.vehicleType_ID]['boundary_df'] = boundary_injection_df
            self.vehicleType_dict[vehicleType.vehicleType_ID]['time_bin'] = time_bin
            
            if verbose: print(f'Boundary injection CVS file successfully loaded for {vehicleType.vehicleType_ID}. The time bin is {time_bin*60:.2f} min(s).')
        else: self.vehicleType_dict[vehicleType.vehicleType_ID]['assigned_by'] = 'stationType'

        # Datum 1: At the critical density, the discharge rate will be the freeflow discharge rate
        sigma = self.trafficBehaviour.critical_density
        Df = vehicleType.discharge_rate_freeflow
        # Datum 1: At the maximum density, the discharge rate will be the jam discharge rate
        P = self.trafficBehaviour.jam_density
        Dj = vehicleType.discharge_rate_jam

        # Calculate discharge curve coefficients per vehicleType
        Dscale = sigma*P*(Dj-Df)/(sigma-P)  # Vertical scale
        Dshift = Df-Dscale/sigma            # Vertical shift
        self.vehicleType_dict[vehicleType.vehicleType_ID]['discharge_tuple'] = (Dscale, Dshift) # Coefficients

    def load_stationType(self,
                         stationType: stationType):
        '''
        Add a stationType to the road scenario. Each station type is given the option to customize an initial density profile (field) in it's SoC field. Intake (off ramp) and injection (on ramp) conditions are initialized to zero in the first timestep.

        stationType: stationType (class) -> A stationType instance
        '''

        self.stationType_dict[stationType.stationType_ID] = {'object': stationType}
    
    def initialize_with_loaded_data(self):

        # Determine the simulation duration, and minimum time_bin based on input data
        self.simulation_duration = 0
        self.time_bin = np.inf
        self.current_timestamp = datetime(year=2200, month=1, day=1)
        self.end_timestamp = datetime(year=1800, month=1, day=1)
        for vehicleType_ID in self.vehicleType_dict.keys():
            boundary_df = self.vehicleType_dict[vehicleType_ID]['boundary_df']
            start_time = min(boundary_df[boundary_df.columns[0]])
            end_time = max(boundary_df[boundary_df.columns[0]])
            
            # Calculate the duration for the current vehicle type in hours
            current_duration_hours = (end_time - start_time).total_seconds() / 3600
            
            # Get the maximum duration
            self.simulation_duration = max(self.simulation_duration, current_duration_hours)

            # Get the minimum timebin
            self.time_bin = min(self.time_bin, self.vehicleType_dict[vehicleType_ID]['time_bin'])

            # Get the earliest timestep
            self.current_timestamp = min(self.current_timestamp, boundary_df[boundary_df.columns[0]].min())
            # Get the latest timestep
            self.end_timestamp = max(self.end_timestamp, boundary_df[boundary_df.columns[0]].max())

        # Calculating the minimum road resolution 
        Vf = self.trafficBehaviour.freeflow_speed
        W  = self.trafficBehaviour.traffic_wavespeed

        # Calculate the minimum resolutions according to numerical stability limit
        dx_min_R = self.time_bin*Vf
        dx_min_RE = self.time_bin*(Vf+W)

        ''' Code block that enforces minimum resolution and adjusts road length. No visible impact on diffusion.
        # Create the road
        original_road_length = self.road_end - self.road_start # Initial road length
        min_resolution = max(dx_min_R, dx_min_RE) # Highest allowable resolution
        initial_num_cells = original_road_length / min_resolution # Number of cells (float)
        adjusted_num_cells = round(initial_num_cells) # Number of cells (int)
        if adjusted_num_cells <= 0:
            raise ValueError('The road is too short to be divided according to the minimum resolution.')
        adjusted_road_length = adjusted_num_cells * min_resolution # Adjust the road length to fit an integer number of cells
        # Update the class attributes
        self.road_cells = adjusted_num_cells
        self.road_resolution = min_resolution
        self.road_end = self.road_start + adjusted_road_length  # Adjust the road_end based on the new road length
        
        print(f"Based on the highest allowable resolution, the road was modified from {original_road_length} to {adjusted_road_length} for discretization into {adjusted_num_cells} cells.")
        '''
        
        ''' Code block that decreases resolution to maintain road length. '''
        # Calculate the number of road cells
        road_length = self.road_end - self.road_start
        base_num_cells = road_length / max(dx_min_R, dx_min_RE) # Determine the base number of cells


        ideal_num_cells = int(base_num_cells) # Determine the ideal number of cells by rounding down the base number (to ensure each cell is at least the min resolution)
        if ideal_num_cells == base_num_cells: # If the base number of cells was already an integer, it means each cell is exactly the minimum resolution.
            ideal_num_cells -= 1 # To ensure each cell is larger, we need one less cell than the base calculation suggests.
        if ideal_num_cells <= 0: # Adjust if the ideal number of cells turns out to be zero (in case of very short roads)
            ValueError('The road is too short to be divided according to the minimum resolution.')
        self.road_cells = ideal_num_cells
        
        # Calculate the actual size of each cell
        self.road_resolution = road_length / ideal_num_cells
        if verbose: print(f'Minimum road resolution used. There are {self.road_cells} road segments, each of length {self.road_resolution:.2f} km.')

        # First initialize each stationType to initialize SoC arrays and define it's station_vehicleType
        for stationType_ID in self.stationType_dict.keys():

            # Create boolean arrays for road placement
            off_ramp_array = np.zeros(self.road_cells)
            on_ramp_array = np.zeros(self.road_cells)

            # Calculate the indices for the off-ramp and on-ramp locations
            off_ramp_index = int((self.stationType_dict[stationType_ID]['object'].off_ramp_location - self.road_start) / self.road_resolution)
            on_ramp_index = int((self.stationType_dict[stationType_ID]['object'].on_ramp_location - self.road_start) / self.road_resolution)

            # Check to ensure the off-ramp location index is always at least one less than the on-ramp index in the other array
            if off_ramp_index == on_ramp_index:
                on_ramp_index += 1
                if verbose: print('The on and off ramps have the same cell index. Incrementing the on ramp index to enter the next cell.')
            elif off_ramp_index > on_ramp_index:
                raise ValueError('The off-ramp location must be before the on-ramp location.')
            
            # Set the values at these indices to 1
            off_ramp_array[off_ramp_index] = 1
            on_ramp_array[on_ramp_index] = 1

            # Store each array per stationType instance
            self.stationType_dict[stationType_ID]['off_ramp_index_array'] = off_ramp_array
            self.stationType_dict[stationType_ID]['on_ramp_index_array'] = on_ramp_array

            # Now that we calculated the minimum time bin, calculate the max ramp capacity (for either ramp)
            self.stationType_dict[stationType_ID]['max_ramp_capacity'] = self.stationType_dict[stationType_ID]['object'].max_ramp_flow * self.time_bin

            # Get the max charging rate
            C_max = self.stationType_dict[stationType_ID]['object'].max_charging_rate
            # Calculate the mininum SoC resolution
            dSoC_min = self.time_bin*C_max

            # Calculate the number of SoC cells
            SoC_length = self.stationType_dict[stationType_ID]['object'].battery_capacity
            base_num_cells = SoC_length / dSoC_min # Determine the base number of cells
            ideal_num_cells = int(base_num_cells) # Determine the ideal number of cells by rounding down the base number (to ensure each cell is at least the min resolution)
            if ideal_num_cells == base_num_cells: # If the base number of cells was already an integer, it means each cell is exactly the minimum resolution.
                ideal_num_cells -= 1 # To ensure each cell is larger, we need one less cell than the base calculation suggests.
            if ideal_num_cells <= 0: # Adjust if the ideal number of cells turns out to be zero (in case of very short roads)
                ValueError('The minimum resolution covers the entire battery. Either lower the charging rate or use finer temporal data.')
            self.stationType_dict[stationType_ID]['SoC_cells'] = ideal_num_cells
            
            # Calculate the actual size of each cell
            self.stationType_dict[stationType_ID]['SoC_cell_resolution'] = SoC_length / ideal_num_cells

            # Initialize the count per SoC array
            '''stest = np.zeros(ideal_num_cells)
            quarter = ideal_num_cells // 4  # Start of the second quarter
            half = ideal_num_cells // 2  # End of the second quarter
            stest[half:half+quarter] = 0'''
            self.stationType_dict[stationType_ID]['SoC_vehicles'] = np.zeros(ideal_num_cells) #stest
            self.stationType_dict[stationType_ID]['mu_in_j'] = np.zeros(ideal_num_cells)
            self.stationType_dict[stationType_ID]['mu_out_j'] = np.zeros(ideal_num_cells)

            # Define a second vehicleType for each stationType to represent the vehicles re-entering the road
            station_vehicleType = vehicleType(battery_capacity=self.stationType_dict[stationType_ID]['object'].battery_capacity,
                                              discharge_rate_freeflow=self.stationType_dict[stationType_ID]['object'].discharge_rate_freeflow,
                                              discharge_rate_jam=self.stationType_dict[stationType_ID]['object'].discharge_rate_jam,
                                              vehicleType_ID=self.stationType_dict[stationType_ID]['object'].vehicleType_ID+
                                                            '_from_'+self.stationType_dict[stationType_ID]['object'].stationType_ID)

            # Load it into the road instance
            self.load_vehicleType(station_vehicleType, assigned_by_stationType=True)

        # Once all stationType instances have been initialized, calculate the upper-bound on energy required to reach the next station 
        off_ramp_indices = [] # Initialize a list to store off ramp indices along with their stationType_ID

        # First loop: Populate off_ramp_indices
        for stationType_ID in self.stationType_dict.keys():
            off_ramp_location = self.stationType_dict[stationType_ID]['object'].off_ramp_location
            off_ramp_index = int((off_ramp_location - self.road_start) / self.road_resolution)
            vehicleType_ID = self.stationType_dict[stationType_ID]['object'].vehicleType_ID
            off_ramp_indices.append((stationType_ID, off_ramp_index, vehicleType_ID))

        # Sort off_ramp_indices by vehicleType_ID and then by off_ramp_index
        off_ramp_indices.sort(key=lambda x: (x[2], x[1]))

        # Iterate through sorted off_ramp_indices to calculate distances
        for i, (stationType_ID, off_ramp_index, vehicleType_ID) in enumerate(off_ramp_indices):
            # Determine if the next off-ramp is of the same vehicleType
            if i < len(off_ramp_indices) - 1 and off_ramp_indices[i + 1][2] == vehicleType_ID:
                # Distance to the next off ramp of the same vehicleType
                next_off_ramp_index = off_ramp_indices[i + 1][1]
                distance = (next_off_ramp_index - off_ramp_index) * self.road_resolution
            else:
                # For the last off ramp of this vehicleType, calculate the distance to the end of the road
                distance = self.road_end - (off_ramp_index * self.road_resolution + self.road_start)

            # Calculate SoC_min for the current off ramp
            Df = self.vehicleType_dict[vehicleType_ID]['object'].discharge_rate_freeflow
            Vf = self.trafficBehaviour.freeflow_speed
            SoC_min = distance * abs(Df) / Vf

            # Update the stationType_dict directly
            self.stationType_dict[stationType_ID]['min_SoC_to_next_station'] = SoC_min

        # Initialize road arrays to zero for each vehicleType instance
        for vehicleType_ID in self.vehicleType_dict.keys():
            self.vehicleType_dict[vehicleType_ID]['road_density'] = np.zeros(self.road_cells)
            self.vehicleType_dict[vehicleType_ID]['road_SoC'] = np.zeros(self.road_cells)
            self.vehicleType_dict[vehicleType_ID]['road_energy_density'] = np.zeros(self.road_cells)
            self.vehicleType_dict[vehicleType_ID]['on_ramp_flow'] = np.zeros(self.road_cells)
            self.vehicleType_dict[vehicleType_ID]['off_ramp_ratio'] = np.zeros(self.road_cells)
            self.vehicleType_dict[vehicleType_ID]['on_ramp_mean_SoC'] = np.zeros(self.road_cells)
            self.vehicleType_dict[vehicleType_ID]['TEMP_off_ramp_flow'] = np.zeros(self.road_cells)

            # Initialize appropriate SoC arrays for flows
            for stationType_ID in self.stationType_dict.keys():
                if (self.stationType_dict[stationType_ID]['object'].vehicleType_ID in vehicleType_ID and 
                    self.vehicleType_dict[vehicleType_ID]['object'].battery_capacity is not None):
                    self.vehicleType_dict[vehicleType_ID][stationType_ID]= {'class_mu_in_j':np.zeros(self.stationType_dict[stationType_ID]['SoC_cells'])}

# roadScenario, simulation
class simulator:
    """
    General simulator progression:
    -> Ramp flows calculated
    -> Traffic: Per class ramp rates
    -> Traffic: Per class updates
    -> Traffic: Aggregate update
    -> Station: SoC updates
    """
    def __init__(self, 
                 roadScenario: roadScenario,
                 simulation:str = 'traffic'): # T, TE, CTEC
        self.road = roadScenario
        self.simulation = simulation

        self.road_cell_left_markers = np.arange(self.road.road_start, 
                                                self.road.road_end, 
                                                self.road.road_resolution)
        
        # Calculate aggregate density
        agg_rho_i = np.zeros(self.road.road_cells) # Initialize
        # Sum over all loaded vehicleTypes
        for vehicleType_ID in self.road.vehicleType_dict.keys():
            agg_rho_i += self.road.vehicleType_dict[vehicleType_ID]['road_density']

        # Round to 3 decimal places, i.e. to the meter
        self.road_cell_left_markers = np.around(self.road_cell_left_markers, decimals=3)

        # Initialize the results dataframes
        self.results_dict = {'aggregate_density': self.result_dataframe_row(agg_rho_i),
                             'vehicleType_results':{},
                             'stationType_results':{}}

        for vehicleType_ID in self.road.vehicleType_dict.keys():
            self.results_dict['vehicleType_results'][vehicleType_ID] = {
                'road_density': self.result_dataframe_row(self.road.vehicleType_dict[vehicleType_ID]['road_density']),
                'road_SoC': self.result_dataframe_row(self.road.vehicleType_dict[vehicleType_ID]['road_SoC']),
                'road_energy_density': self.result_dataframe_row(self.road.vehicleType_dict[vehicleType_ID]['road_energy_density'])}
        
        for stationType_ID in self.road.stationType_dict.keys():
            self.results_dict['stationType_results'][stationType_ID] = {
                'SoC_vehicles': self.result_dataframe_row(self.road.stationType_dict[stationType_ID]['SoC_vehicles'], stationType_ID)}
    
    def result_dataframe_row(self, array, stationType_ID=None):
        if stationType_ID == None:
            return pd.DataFrame([array], 
                                columns = self.road_cell_left_markers,
                                index = [self.road.current_timestamp])
        else:
            station_cell_left_markers = np.arange(0, 
                                                  self.road.stationType_dict[stationType_ID]['object'].battery_capacity, 
                                                  self.road.stationType_dict[stationType_ID]['SoC_cell_resolution'])
            station_cell_left_markers = np.around(station_cell_left_markers, decimals=3)
            return pd.DataFrame([array], 
                                columns = station_cell_left_markers,
                                index = [self.road.current_timestamp])

    def nextStep(self):
        # Update the timestep
        self.road.current_timestamp = self.road.current_timestamp + timedelta(hours=self.road.time_bin)

        # Implement both off and on ramp vehicle and energy exchange
        # Clear the previously saved ramp related quantities
        for vehicleType_ID in self.road.vehicleType_dict.keys():
            self.road.vehicleType_dict[vehicleType_ID]['off_ramp_ratio'] = np.zeros(self.road.road_cells)
            self.road.vehicleType_dict[vehicleType_ID]['on_ramp_flow'] = np.zeros(self.road.road_cells)
            self.road.vehicleType_dict[vehicleType_ID]['on_ramp_mean_SoC'] = np.zeros(self.road.road_cells)

        # Determine the number of vehicles that can enter the station
        for stationType_ID in self.road.stationType_dict.keys():
            # Pre-calculations
            SoC_max = self.road.stationType_dict[stationType_ID]['object'].battery_capacity
            SoC_cells = self.road.stationType_dict[stationType_ID]['SoC_cells']
            bins = np.linspace(0, SoC_max, SoC_cells + 1) # Bins from 0 to battery_capacity
            bin_centers = (bins[:-1] + bins[1:]) / 2
            bin_width = bins[1] - bins[0]
            vehicles_at_station = np.sum(self.road.stationType_dict[stationType_ID]['SoC_vehicles'])

            for vehicleType_ID in self.road.vehicleType_dict.keys():
                # Check if the vehicleType can enter the station
                if (self.road.stationType_dict[stationType_ID]['object'].vehicleType_ID in vehicleType_ID and 
                    self.road.vehicleType_dict[vehicleType_ID]['object'].battery_capacity is not None):

                    # Calculate how many vehicles in vehicleType will enter
                    station_capacity = min(self.road.stationType_dict[stationType_ID]['max_ramp_capacity'], 
                                           self.road.stationType_dict[stationType_ID]['object'].vehicle_capacity-vehicles_at_station)
                    candidate_vehicle_count = np.sum(np.multiply(self.road.vehicleType_dict[vehicleType_ID]['road_density']*self.road.road_resolution,
                                                            self.road.stationType_dict[stationType_ID]['off_ramp_index_array']))
                    candidate_vehicle_mean_SoC = np.sum(np.multiply(self.road.vehicleType_dict[vehicleType_ID]['road_SoC'],
                                                            self.road.stationType_dict[stationType_ID]['off_ramp_index_array']))

                    permissible_vehicle_count = min(candidate_vehicle_count, station_capacity)
                    
                    # Create a normal distribution
                    standard_deviation=10
                    initial_SoC_profile = norm.pdf(bin_centers, candidate_vehicle_mean_SoC, standard_deviation)
                    # Multiply such that the area represents the vehicle_count
                    initial_sum = np.sum(initial_SoC_profile)
                    initial_SoC_profile *= permissible_vehicle_count / initial_sum
                    #print(initial_SoC_profile)
                    
                    k=0.2
                    mu_c = self.road.stationType_dict[stationType_ID]['min_SoC_to_next_station']*1.2
                    road_exiting_probs = 1 / (1 + np.exp(k * (bin_centers - mu_c)))
                    exiting_SoC_profile = initial_SoC_profile * road_exiting_probs  # Expected density of removed vehicles
                    final_SoC_profile = initial_SoC_profile - exiting_SoC_profile  # Remaining density after removal

                    # Use the exiting profile to directly get station incoming SoC flows
                    class_mu_in_j = exiting_SoC_profile/self.road.time_bin
                    self.road.vehicleType_dict[vehicleType_ID][stationType_ID]['class_mu_in_j'] = class_mu_in_j
                    
                    # Calculate the vehicle counts remaining on the road, after the split
                    final_vehicle_count = np.sum(exiting_SoC_profile)
                    # Calculate the splitting ratio for each vehicleType entering this stationType, class_beta_i, and store it
                    if permissible_vehicle_count > 0:
                        splitting_ratio = final_vehicle_count/permissible_vehicle_count
                        if splitting_ratio > 1:
                            # No negative densities!
                            splitting_ratio = 1
                    else:
                        splitting_ratio = 0

                    class_beta_i = splitting_ratio * self.road.stationType_dict[stationType_ID]['off_ramp_index_array']
                    self.road.vehicleType_dict[vehicleType_ID]['off_ramp_ratio'] += class_beta_i

                    # Calculate the mean value of the SoC profile of the cars still on the road
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    if np.sum(final_SoC_profile) > 0:
                        final_SoC_mean = np.sum(bin_centers * final_SoC_profile) / np.sum(final_SoC_profile)
                    else:
                        final_SoC_mean = 0
                    # Find the index of the non-zero value in the off_ramp_index_array
                    non_zero_indices = np.nonzero(self.road.stationType_dict[stationType_ID]['off_ramp_index_array'])
                    if len(non_zero_indices[0]) > 0:  # Check if there is at least one non-zero value
                        non_zero_index = non_zero_indices[0][0]  # Get the first (and only) non-zero index
                        # Modify the road_density and SoC arrays
                        self.road.vehicleType_dict[vehicleType_ID]['road_SoC'][non_zero_index] = final_SoC_mean # Should increase

            # Calculate the maximum number of vehicles that can enter the road
            road_on_capacity = max(0, self.road.trafficBehaviour.jam_density - 
                                      np.sum(np.multiply(self.road.vehicleType_dict[vehicleType_ID]['road_density'],
                                                         self.road.stationType_dict[stationType_ID]['on_ramp_index_array'])))
            max_removals = min(road_on_capacity, self.road.stationType_dict[stationType_ID]['max_ramp_capacity'])

            # region - Calculate station output flows, mu_out_j
            # Get all charging coefficients
            c_max = self.road.stationType_dict[stationType_ID]['object'].max_charging_rate
            full_length = self.road.stationType_dict[stationType_ID]['SoC_cells'] # Get the length of the SoC array
            taper_start_index = int(full_length * 0.8)
            # Create the first segment with c_max
            first_segment = np.full(taper_start_index, c_max)
            # Create the second segment that tapers down linearly from c_max to 0
            taper_length = full_length - taper_start_index
            second_segment = np.linspace(c_max, 0, taper_length)
            # Concatenate the two segments
            c_j = np.concatenate((first_segment, second_segment))
            c_jm1 = np.roll(c_j, 1); c_jm1[0] = 0 # Roll to the left
            c_jp1 = np.roll(c_j, -1); c_jp1[-1] = 0 # Roll to the right
            c_jm1_max = np.maximum(0, c_jm1)
            c_jp1_min = np.minimum(0, c_jp1)
            c_j_abs = np.abs(c_j)

            # Get all vehicle counts per SoC
            eta_j = self.road.stationType_dict[stationType_ID]['SoC_vehicles']
            eta_jm1 = np.roll(eta_j, 1); eta_jm1[0] = 0 # Roll to the left
            eta_jp1 = np.roll(eta_j, -1); eta_jp1[-1] = 0 # Roll to the right
            L_s = self.road.stationType_dict[stationType_ID]['SoC_cell_resolution']

            # Calculate the center of each bin and the removal probabilities
            prob_scale = min(1, self.road.time_bin*10) # Scaling the probability to account for delay-to-car time set at 10 minutes
            station_exiting_probs = prob_scale / (1 + np.exp(-k * (bin_centers - SoC_max * 0.9)))
            
            initial_removed_vehicles = eta_j * station_exiting_probs  # Expected density of removed vehicles
            # Apply the bounds to removal condition, Eqn 21 in multiclass paper 
            upper_bound_on_removals = np.multiply((1/self.road.time_bin - c_j_abs/L_s), eta_j) + np.multiply(c_jm1_max/L_s, eta_jm1) + np.multiply(c_jp1_min/L_s, eta_jp1)
            adjusted_removed_vehicles = np.minimum(initial_removed_vehicles, upper_bound_on_removals*self.road.time_bin)
            adjusted_removed_vehicles[adjusted_removed_vehicles<0]=0

            # Check if the total removals exceed the maximum allowed by the road and ramp limits
            total_removals = adjusted_removed_vehicles.sum()
            if total_removals > max_removals:
                # Calculate a scaling factor to ensure total removals do not exceed max_removals
                scaling_factor = max_removals / total_removals
                # Apply the scaling factor uniformly across all bins
                vehicles_done_charging = adjusted_removed_vehicles * scaling_factor
            else:
                vehicles_done_charging = adjusted_removed_vehicles

            # Calculate exit flows and mean SoC
            mu_out_j = vehicles_done_charging/self.road.time_bin
            self.road.stationType_dict[stationType_ID]['mu_out_j'] = mu_out_j
            # endregion

            if sum(mu_out_j)>0:
                # Calculate the total SoC contribution from each bin
                exiting_mean_eps = np.sum(np.multiply(vehicles_done_charging, bin_centers))/np.sum(vehicles_done_charging)
            else:
                exiting_mean_eps = 0

            # Calculate class_r_on_i, save it to the on-ramp class (separate from the off_ramp class)
            class_r_on_i = np.sum(mu_out_j) * self.road.stationType_dict[stationType_ID]['on_ramp_index_array']
            class_eps_on_i = exiting_mean_eps * self.road.stationType_dict[stationType_ID]['on_ramp_index_array']
            on_vehicleType_ID = self.road.stationType_dict[stationType_ID]['object'].vehicleType_ID+'_from_'+self.road.stationType_dict[stationType_ID]['object'].stationType_ID
            self.road.vehicleType_dict[on_vehicleType_ID]['on_ramp_flow'] += class_r_on_i
            self.road.vehicleType_dict[on_vehicleType_ID]['on_ramp_mean_SoC'] += class_eps_on_i

        # Define local copies
        T = self.road.time_bin
        L = self.road.road_resolution
        Vf = self.road.trafficBehaviour.freeflow_speed
        P = self.road.trafficBehaviour.jam_density
        W = self.road.trafficBehaviour.traffic_wavespeed
        sigma = self.road.trafficBehaviour.critical_density

        # Calculate aggregate density
        agg_rho_i = np.zeros(self.road.road_cells) # Initialize
        # Sum over all loaded vehicleTypes
        for vehicleType_ID in self.road.vehicleType_dict.keys():
            agg_rho_i += self.road.vehicleType_dict[vehicleType_ID]['road_density']
        agg_rho_i_min = np.minimum(agg_rho_i, sigma)
        agg_rho_ip1_max = np.roll(np.maximum(agg_rho_i, sigma), -1) # Value to the right
        agg_rho_ip1_max[-1] = 0 # No density at the exit end of the road

        # Calculate aggregate road to station splitting ratio arrays
        agg_beta_i = np.zeros(self.road.road_cells)
        for vehicleType_ID in self.road.vehicleType_dict.keys():
            class_beta_i = self.road.vehicleType_dict[vehicleType_ID]['off_ramp_ratio']
            class_rho_i = self.road.vehicleType_dict[vehicleType_ID]['road_density']
            out_array = np.zeros(self.road.road_cells, dtype=np.float64)
            agg_beta_i += np.multiply(np.divide(class_rho_i, agg_rho_i, out=out_array, where=agg_rho_i!=0), class_beta_i)
        agg_beta_im1 = np.roll(agg_beta_i, 1); agg_beta_im1[0]=0 # Value to the left

        # Get station to road 
        agg_r_on_i = np.zeros(self.road.road_cells)
        # Sum over all loaded vehicleTypes
        for vehicleType_ID in self.road.vehicleType_dict.keys():
            agg_r_on_i += self.road.vehicleType_dict[vehicleType_ID]['on_ramp_flow']
        agg_r_on_ip1 = np.roll(agg_r_on_i, -1); agg_r_on_i[-1]=0 # Value to the right

        # Calculate aggregate downstream flux
        out_array = np.zeros(self.road.road_cells, dtype=np.float64)
        q_i_plus = np.minimum(Vf*agg_rho_i_min, np.divide(W*(P-agg_rho_ip1_max)-agg_r_on_ip1, 1-agg_beta_i, out=out_array, where=agg_beta_i<1))
        q_im1_plus = np.roll(q_i_plus, 1) # Value to the left
        # Get the total flow injection
        q_1m1_plus_0 = 0
        for vehicleType_ID in self.road.vehicleType_dict.keys():
            # Only search through user defined vehicleTypes
            if self.road.vehicleType_dict[vehicleType_ID]['assigned_by'] == 'user':
                boundary_df = self.road.vehicleType_dict[vehicleType_ID]['boundary_df']
                # Check if the current_timestamp is unique
                if boundary_df[boundary_df[boundary_df.columns[0]] == self.road.current_timestamp].shape[0] == 1:
                    # Extract the value from the second column at the current timestep
                    q_1m1_plus_0 += boundary_df.loc[boundary_df[boundary_df.columns[0]] == self.road.current_timestamp, boundary_df.columns[1]].iloc[0]
                else:
                    # Throw an error if the current timestep is not unique
                    raise ValueError(f"The timestep {self.road.current_timestamp} is not unique in the DataFrame.")
        # Assign the total flow to the first element
        q_im1_plus[0] = q_1m1_plus_0

        # Calculate aggregate upstream flux
        q_i_minus = q_im1_plus*(1 - agg_beta_im1) + agg_r_on_i

        # Aggregate update step
        next_agg_rho_i = agg_rho_i + T/L*(q_i_minus - q_i_plus)

        # Save the results in the results matrices
        self.results_dict['aggregate_density'] = pd.concat([self.results_dict['aggregate_density'],
                                                            self.result_dataframe_row(next_agg_rho_i)])
        
        # Calculate vehicle specific updates
        for vehicleType_ID in self.road.vehicleType_dict.keys():
            # Define local copies
            class_beta_i = self.road.vehicleType_dict[vehicleType_ID]['off_ramp_ratio']
            class_beta_im1 = np.roll(class_beta_i, 1); class_beta_im1[0]=0
            #
            class_r_on_i = self.road.vehicleType_dict[vehicleType_ID]['on_ramp_flow']
            #
            class_eps_i = self.road.vehicleType_dict[vehicleType_ID]['road_SoC']
            class_eps_im1 = np.roll(class_eps_i, 1)
            # Get the class SoC injection
            if self.road.vehicleType_dict[vehicleType_ID]['assigned_by'] == 'user':
                boundary_df = self.road.vehicleType_dict[vehicleType_ID]['boundary_df']
                # Check if the current_timestamp is unique
                if boundary_df[boundary_df[boundary_df.columns[0]] == self.road.current_timestamp].shape[0] == 1:
                    # Extract the value from the second column at the current timestep
                    class_eps_im1_0 = T*boundary_df.loc[boundary_df[boundary_df.columns[0]] == self.road.current_timestamp, boundary_df.columns[2]].iloc[0]
                else:
                    # Throw an error if the current timestep is not unique
                    raise ValueError(f"The timestep {self.road.current_timestamp} is not unique in the DataFrame.")
            else: 
                class_eps_im1_0 = 0

            # Assign the SoC injection to the first element
            class_eps_im1[0] = class_eps_im1_0
            #
            class_eps_on_i = self.road.vehicleType_dict[vehicleType_ID]['on_ramp_mean_SoC']

            # Calculate the road discharge field
            class_Df = self.road.vehicleType_dict[vehicleType_ID]['object'].discharge_rate_freeflow
            class_Dj = self.road.vehicleType_dict[vehicleType_ID]['object'].discharge_rate_jam
            class_Dscale = self.road.vehicleType_dict[vehicleType_ID]['discharge_tuple'][0]
            class_Dshift = self.road.vehicleType_dict[vehicleType_ID]['discharge_tuple'][1]
            # Initialize the discharge field
            class_d_i = np.zeros(self.road.road_cells)
            # Conditions
            freeflowing = (agg_rho_i <= sigma) & (agg_rho_i != 0)
            congested = (agg_rho_i > sigma) & (agg_rho_i < P)
            jammed = agg_rho_i >= P
            # Apply conditions
            out_array = np.zeros(self.road.road_cells, dtype=np.float64)
            class_d_i = np.divide(class_Dscale, agg_rho_i, out=out_array, where=(agg_rho_i!=0))
            class_d_i[congested] += class_Dshift
            class_d_i[freeflowing] = class_Df
            class_d_i[jammed] = class_Dj
            class_d_im1 = np.roll(class_d_i, 1); class_d_im1[0] = 0

            # Get the flux values per vehicleType
            class_rho_i = self.road.vehicleType_dict[vehicleType_ID]['road_density']
            # Calculate class downstream density flux
            out_array = np.zeros(self.road.road_cells, dtype=np.float64)
            class_q_i_plus = np.divide(class_rho_i, agg_rho_i, out=out_array, where=agg_rho_i!=0)*q_i_plus
            class_q_im1_plus = np.roll(class_q_i_plus, 1) # Value to the left
            # Get the class flow injection
            if self.road.vehicleType_dict[vehicleType_ID]['assigned_by'] == 'user':
                boundary_df = self.road.vehicleType_dict[vehicleType_ID]['boundary_df']
                # Check if the current_timestamp is unique
                if boundary_df[boundary_df[boundary_df.columns[0]] == self.road.current_timestamp].shape[0] == 1:
                    # Extract the value from the second column at the current timestep
                    class_q_1m1_plus_0 = boundary_df.loc[boundary_df[boundary_df.columns[0]] == self.road.current_timestamp, boundary_df.columns[1]].iloc[0]
                else:
                    # Throw an error if the current timestep is not unique
                    raise ValueError(f"The timestep {self.road.current_timestamp} is not unique in the DataFrame.")
            else: class_q_1m1_plus_0 = 0
            # Assign the total flow to the first element
            class_q_im1_plus[0] = class_q_1m1_plus_0

            # Calculate class upstream density flux
            class_q_i_minus = np.multiply(class_q_im1_plus, 1 - class_beta_im1) + class_r_on_i

            # Calculate flows without beta, with beta, subtract to get flows to station. Scale the flows to the station later.
            self.road.vehicleType_dict[vehicleType_ID]['TEMP_off_ramp_flow'] = np.zeros(self.road.road_cells)
            for stationType_ID in self.road.stationType_dict.keys():
                flow_without_beta = np.sum(np.multiply(class_q_im1_plus, np.roll(self.road.stationType_dict[stationType_ID]['off_ramp_index_array'], 1)))
                flow_without_beta *= self.road.stationType_dict[stationType_ID]['off_ramp_index_array']
                flow_with_beta = np.sum(np.multiply(np.multiply(class_q_im1_plus, 1 - class_beta_im1), np.roll(self.road.stationType_dict[stationType_ID]['off_ramp_index_array'], 1)))
                flow_with_beta *= self.road.stationType_dict[stationType_ID]['off_ramp_index_array']
                self.road.vehicleType_dict[vehicleType_ID]['TEMP_off_ramp_flow'] += flow_without_beta - flow_with_beta

            # Calculate class upstream energy flux
            class_phi_i_minus = np.multiply(class_q_i_minus - class_r_on_i, class_eps_im1 + class_d_im1*T) + np.multiply(class_r_on_i, class_eps_on_i)
            
            # Calculate class downstream energy flux
            class_phi_i_plus = np.multiply(class_q_i_plus, class_eps_i + class_d_i*T)

            # vehicleType density update step
            next_class_rho_i = class_rho_i + T/L*(class_q_i_minus - class_q_i_plus)

            if self.road.vehicleType_dict[vehicleType_ID]['object'].battery_capacity is not None:
                # vehicleType energy density update step
                next_class_rho_eps_i = np.multiply(class_rho_i, class_eps_i + class_d_i*T) + T/L*(class_phi_i_minus - class_phi_i_plus)

                # vehicleType SoC update step
                out_array = np.zeros(self.road.road_cells, dtype=np.float64)
                next_class_eps_i = np.divide(next_class_rho_eps_i, next_class_rho_i, out=out_array, where=next_class_rho_i!=0)
                # If the number of vehicles in the cell indicates less than 1 car on the whole road, there shouldn't be an SoC value
                # next_class_eps_i[next_class_rho_i<1/L]=0 
                # There shouldn't be negative SoC values
                next_class_eps_i[next_class_eps_i<0]=0

            else:
                next_class_rho_eps_i = np.zeros(self.road.road_cells)
                next_class_eps_i = np.zeros(self.road.road_cells)

            """# Create a dictionary of arrays to check for NaN values
            arrays_to_check = {
                'class_rho_i': class_rho_i,
                'class_q_i_plus': class_q_i_plus,
                'class_q_i_minus': class_q_i_minus,
                'class_eps_i': class_eps_i,
                'class_d_i': class_d_i,
                'class_phi_i_minus': class_phi_i_minus,
                'class_phi_i_plus': class_phi_i_plus,
                'next_class_rho_i': next_class_rho_i,
                'next_class_rho_eps_i': next_class_rho_eps_i,
                'next_class_eps_i': next_class_eps_i
            }

            # Perform the check
            check_for_nans(arrays_to_check)"""

            # Save the results in the results matrices
            self.results_dict['vehicleType_results'][vehicleType_ID]['road_density'] = pd.concat(
                [self.results_dict['vehicleType_results'][vehicleType_ID]['road_density'],
                 self.result_dataframe_row(next_class_rho_i)])
            self.results_dict['vehicleType_results'][vehicleType_ID]['road_SoC'] = pd.concat(
                [self.results_dict['vehicleType_results'][vehicleType_ID]['road_SoC'],
                 self.result_dataframe_row(next_class_eps_i)])
            self.results_dict['vehicleType_results'][vehicleType_ID]['road_energy_density'] = pd.concat(
                [self.results_dict['vehicleType_results'][vehicleType_ID]['road_energy_density'],
                 self.result_dataframe_row(next_class_rho_eps_i)])

        for stationType_ID in self.road.stationType_dict.keys():
            # Get local copy
            c_max = self.road.stationType_dict[stationType_ID]['object'].max_charging_rate

            # Obtain inflows per station
            mu_in_j = np.zeros(self.road.stationType_dict[stationType_ID]['SoC_cells'])
            # Sum over all appropriate vehicleTypes
            for vehicleType_ID in self.road.vehicleType_dict.keys():
                if (self.road.stationType_dict[stationType_ID]['object'].vehicleType_ID in vehicleType_ID and 
                    self.road.vehicleType_dict[vehicleType_ID]['object'].battery_capacity is not None):

                    off_ramp_flow = np.sum(np.multiply(self.road.vehicleType_dict[vehicleType_ID]['TEMP_off_ramp_flow'],
                                                      self.road.stationType_dict[stationType_ID]['off_ramp_index_array']))
                    class_mu_in_j = self.road.vehicleType_dict[vehicleType_ID][stationType_ID]['class_mu_in_j']
                    if np.sum(class_mu_in_j) > 0: 
                        mu_in_j += class_mu_in_j*off_ramp_flow/np.sum(class_mu_in_j)

            mu_out_j = self.road.stationType_dict[stationType_ID]['mu_out_j']

            # Get all charging coefficients
            full_length = self.road.stationType_dict[stationType_ID]['SoC_cells'] # Get the length of the SoC array
            taper_start_index = int(full_length * 0.8)
            # Create the first segment with c_max
            first_segment = np.full(taper_start_index, c_max)
            # Create the second segment that tapers down linearly from c_max to 0
            taper_length = full_length - taper_start_index
            second_segment = np.linspace(c_max, 0, taper_length)
            # Concatenate the two segments
            c_j = np.concatenate((first_segment, second_segment))
            c_jm1 = np.roll(c_j, 1); c_jm1[0] = 0 # Roll to the left
            c_jp1 = np.roll(c_j, -1); c_jp1[-1] = 0 # Roll to the right
            c_jm1_max = np.maximum(0, c_jm1)
            c_jp1_min = np.minimum(0, c_jp1)
            c_j_abs = np.abs(c_j)

            # Get all vehicle counts per SoC
            eta_j = self.road.stationType_dict[stationType_ID]['SoC_vehicles']
            eta_jm1 = np.roll(eta_j, 1); eta_jm1[0] = 0 # Roll to the left
            eta_jp1 = np.roll(eta_j, -1); eta_jp1[-1] = 0 # Roll to the right
            L_s = self.road.stationType_dict[stationType_ID]['SoC_cell_resolution']

            # stationType SoC update step
            next_eta_j = eta_j + T/L_s*(np.multiply(c_jm1_max, eta_jm1) - 
                                        np.multiply(c_j_abs, eta_j) - 
                                        np.multiply(c_jp1_min, eta_jp1)) + T*(mu_in_j - mu_out_j)
            
            # Remove negative values (sometimes created due to rounding issues)
            next_eta_j[next_eta_j<0]=0

            # Save the results in the results matrices
            self.results_dict['stationType_results'][stationType_ID]['SoC_vehicles'] = pd.concat(
                [self.results_dict['stationType_results'][stationType_ID]['SoC_vehicles'],
                 self.result_dataframe_row(next_eta_j, stationType_ID)])
            
        # Copy the results to the road instance after all calculations are done
        for vehicleType_ID in self.road.vehicleType_dict.keys():
            # Extract the last row for each metric
            final_road_density = self.results_dict['vehicleType_results'][vehicleType_ID]['road_density'].iloc[-1:].values.flatten()
            final_road_SoC = self.results_dict['vehicleType_results'][vehicleType_ID]['road_SoC'].iloc[-1].values.flatten()
            final_road_energy_density = self.results_dict['vehicleType_results'][vehicleType_ID]['road_energy_density'].iloc[-1].values.flatten()

            # Update the road with the final results
            self.road.vehicleType_dict[vehicleType_ID]['road_density'] = final_road_density
            self.road.vehicleType_dict[vehicleType_ID]['road_SoC'] = final_road_SoC
            self.road.vehicleType_dict[vehicleType_ID]['road_energy_density'] = final_road_energy_density

        for stationType_ID in self.road.stationType_dict.keys():
            # Extract the last row
            final_eta_j = self.results_dict['stationType_results'][stationType_ID]['SoC_vehicles'].iloc[-1:].values.flatten()
            self.road.stationType_dict[stationType_ID]['SoC_vehicles'] = final_eta_j # Update the road

    def simulate(self):
        # Iterate through timesteps until finished
        while self.road.current_timestamp < self.road.end_timestamp:
            self.nextStep()
        
        # Perform post simulation calculations
        for stationType_ID in self.road.stationType_dict.keys():
            c_max = self.road.stationType_dict[stationType_ID]['object'].max_charging_rate
            # Get all charging coefficients
            full_length = self.road.stationType_dict[stationType_ID]['SoC_cells'] # Get the length of the SoC array
            taper_start_index = int(full_length * 0.8)
            # Create the first segment with c_max
            first_segment = np.full(taper_start_index, c_max)
            # Create the second segment that tapers down linearly from c_max to 0
            taper_length = full_length - taper_start_index
            second_segment = np.linspace(c_max, 0, taper_length)
            # Concatenate the two segments
            c_j = np.concatenate((first_segment, second_segment))

            df = self.results_dict['stationType_results'][stationType_ID]['SoC_vehicles']
            # Calculate the total number of EVs at the station, at each timestep
            total_EVs_charging = df.sum(axis=1).to_frame(name="total_EVs_charging")
            self.results_dict['stationType_results'][stationType_ID]['total_EVs_charging'] = total_EVs_charging

            # Calculate the total power at each station, at each timestep
            total_power = df.multiply(c_j, axis=1).sum(axis=1).to_frame(name="total_power")
            self.results_dict['stationType_results'][stationType_ID]['total_power'] = total_power

    def get_available_categories(self):
        return list(self.results_dict.keys())

    def get_available_instances(self, category):
        if category in self.results_dict:
            return list(self.results_dict[category].keys())
        else:
            return []

    def get_available_plot_types(self, category, instance):
        if category in self.results_dict and instance in self.results_dict[category]:
            return list(self.results_dict[category][instance].keys())
        else:
            return []

    def validate_input(self, value, available_options, input_name):
        if value not in available_options:
            raise KeyError(f"Invalid {input_name}: {value}. Available options are: {available_options}")

    def plot(self, category:str = None, instance:str = None, plotType:str = None,
             width = 10, height = 8):
        # Validate category
        available_categories = self.get_available_categories()
        if category not in available_categories:
            raise KeyError(f"Please indicate which category of instance results you want to access using the category argument. Available categories are: {available_categories}")
        
        ylabel = 'Time'
        xlabel = 'Road Segments, km'
        P       = self.road.trafficBehaviour.jam_density
        sigma   = self.road.trafficBehaviour.critical_density
        colors = ["green", "yellow", "red"]
        #nodes = [0.0, sigma/P, 1.0]
        nodes = [0.0, 0.5, 1.0]
        norm = None #mcolors.Normalize(vmin=0, vmax=P)

        if category == 'aggregate_density':
            df = self.results_dict[category]
            title = 'Aggregate Road Density Results'
            clabel = 'veh/km'
        else:
            # Validate instance
            available_instances = self.get_available_instances(category)
            self.validate_input(instance, available_instances, 'instance')
            
            # Validate plotType
            available_plot_types = self.get_available_plot_types(category, instance)
            self.validate_input(plotType, available_plot_types, 'plotType')
            
            df = self.results_dict[category][instance][plotType]

        if plotType == 'road_density':
            title = f'Density of {instance} Vehicles on the Road'
            clabel = 'veh/km'
        elif plotType == 'road_energy_density':
            title = f'Energy Density of {instance} Vehicles on the Road'
            clabel = 'kW*veh/km'
            nodes = [0.0, 0.5, 1.0]
            norm = None
        elif plotType == 'road_SoC':
            title = f'Mean State of Charge (SoC) of {instance} Vehicles on the Road'
            clabel = 'kW'
            nodes = [0.0, 0.5, 1.0]
            norm = None
        elif plotType == 'SoC_vehicles':
            xlabel = 'State of Charge (SoC), kW'
            title = f'Binned State of Charge Profile of Vehicles at Station {instance}'
            clabel = 'veh'
            nodes = [0.0, 0.5, 1.0]
            norm = None

        plt.figure(figsize=(width, height))
        if plotType == 'total_EVs_charging':
            plt.plot(df.index, df['total_EVs_charging'], label='Total EVs Charging', color='blue')
            plt.xlabel('Time')  # Assuming the index is datetime
            plt.ylabel('Total EVs Charging')
            plt.title(f'Total EVs Charging Over Time at Station {instance}')
            plt.grid(True)  # Add gridlines for better readability
            plt.legend()  # Show legend
            plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
            plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
            plt.show()

        elif plotType == 'total_power':
            plt.plot(df.index, df['total_power'], label='Total Power', color='red')
            plt.xlabel('Time')  # Assuming the index is datetime
            plt.ylabel('Total Power, kW')
            plt.title(f'Total Power Consumption Over Time at Station {instance}')
            plt.grid(True)  # Add gridlines for better readability
            plt.legend()  # Show legend
            plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
            plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
            plt.show()

        else:
            custom_colormap = mcolors.LinearSegmentedColormap.from_list("custom_colormap", list(zip(nodes, colors)))
            ax = sns.heatmap(df[::-1], cmap=custom_colormap, norm=norm, cbar_kws={'label': clabel}, annot=False)

            # Reducing y-tick clutter and setting y-tick labels
            y_tick_freq = len(df) // 10  # Adjust the frequency of y-ticks here
            y_tick_labels = [label.strftime('%H:%M') for label in df.index[::y_tick_freq]]  # Generate labels
            y_ticks = np.arange(len(df.index), 0, -y_tick_freq)  # Calculate tick positions
            ax.set_yticks(y_ticks + 0.5)  # Apply tick positions, adjust by 0.5 for centering
            ax.set_yticklabels(y_tick_labels, rotation=0)  # Set the labels

            # Reducing x-tick clutter (Optional, adjust based on your dataset)
            x_tick_freq = max(len(df.columns) // 10, 1)  # Ensure at least 1 to avoid division by zero
            x_ticks = np.arange(0, len(df.columns), x_tick_freq)
            ax.set_xticks(x_ticks + 0.5)
            ax.set_xticklabels(df.columns[x_ticks], rotation=45)

            # Add labels and plot
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.show()

def check_for_nans(arrays_dict):
    """
    Check each provided array for NaN values and print the first occurrence.

    Parameters:
    arrays_dict (dict): A dictionary where keys are descriptive names of the arrays
                        and values are the arrays themselves.
    """
    for name, array in arrays_dict.items():
        if np.isnan(array).any():
            first_nan_index = np.where(np.isnan(array))[0][0]
            raise ValueError(f"NaN found in {name} at index: {first_nan_index}")
        else:
            if verbose: print(f"No NaN values in {name}.")

def format_boundary_df(df):
    # Ensure the timestamp column is in datetime64 format
    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])

    # Calculate differences between consecutive timestamps in minutes
    time_diffs = df.iloc[:, 0].diff().dt.total_seconds().div(60).dropna()

    # Find the most common duration between timestamps in minutes
    most_common_duration = time_diffs.mode().iloc[0]

    # Calculate the multiplier to convert quantities to per hour
    # Assuming the most common duration is in minutes
    multiplier = 60 / most_common_duration

    # Convert quantities from per duration to per hour
    df[df.columns[1]] = df[df.columns[1]] * multiplier
    df[df.columns[2]] = df[df.columns[2]] * multiplier

    return df, most_common_duration / 60 # in hours

def int_rebin_dataframe(df, current_bin_size_mins, new_bin_size_mins):
    """
    Re-bin the second column of a DataFrame into sub bins.

    df: The input DataFrame with the first column 'Timestamp', 
               the second column as binned vehicles, 
               and the third column with other properties.
    time_bin: The current binning interval as a pd.Timedelta (e.g., pd.Timedelta(minutes=10))
    sub_bin_factor: The factor by which to divide the data bin size (e.g., 2 for half-size bins)
    return: A new DataFrame with updated bins.
    """

    # Create a new DataFrame to hold the rebinned data
    rebinned_data = []

    # Iterate through each row of the DataFrame
    for index, row in df.iterrows():
        # Calculate the number of new bins needed for this row
        num_new_bins = int(current_bin_size_mins / new_bin_size_mins)

        # Split the second column value evenly across the new bins
        split_value = row.iloc[1] / num_new_bins

        # Generate new timestamps and fill the rebinned_data list
        for i in range(num_new_bins):
            new_timestamp = row.iloc[0] + pd.Timedelta(minutes=i*new_bin_size_mins)
            rebinned_data.append([new_timestamp, split_value, row.iloc[2]])

    # Create the new DataFrame from the rebinned data
    new_df = pd.DataFrame(rebinned_data, columns=df.columns)

    return new_df