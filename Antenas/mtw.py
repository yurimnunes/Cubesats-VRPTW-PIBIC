# %%
import json
import random
import uuid
from datetime import datetime, timedelta

# Convert a datetime to minutes since epoch
def datetime_to_minutes(dt):
    return int(dt.timestamp() // 60)

# Generate random minutes within a given range
def generate_random_minutes(start_minutes, end_minutes):
    return int(random.uniform(start_minutes, end_minutes))

# Generate view periods for resources
def generate_view_periods(start_minutes, end_minutes, num_periods=3):
    periods = []
    period_duration = (end_minutes - start_minutes) // num_periods
    for i in range(num_periods):
        rise = start_minutes + period_duration * i
        set_time = rise + int(random.uniform(60, 240))  # Duration between 1 to 4 hours in minutes
        periods.append({
            "RISE": rise,
            "SET": set_time,
            "TRX ON": rise,
            "TRX OFF": set_time
        })
    return periods

# Generate track data for a given week and year
def generate_track_data(week, year, num_tracks):
    job_id = 0
    start_week = datetime.strptime(f'{year} {week} 1', '%Y %W %w')
    start_week_minutes = 0
    data = []
    for _ in range(num_tracks):
        duration = int(random.uniform(60, 2*60))  # Duration between 1 to 3.5 hours in minutes
        setup_time = 60
        teardown_time = 15
        track_id = job_id
        job_id += 1
        time_window_start = start_week_minutes + generate_random_minutes(0, 6.5 * 24 * 60 * 1)  # Random start within the week
        time_window_end = time_window_start + generate_random_minutes(1 * 60, 1.5 * 60)  # Random end 1 to 2 hours later
        
        require_two_antennas = random.random() < 0.1  # 20% chance of requiring exactly two antennas

        if require_two_antennas:
            # Select exactly two antennas for this job
            resources = random.sample(["DSS-24", "DSS-26", "DSS-34", "DSS-36", "DSS-54"], k=2)
            # concatenate both antennas in a single string with _ separator
            resources = ["_".join(resources)]
        else:
            # Select between 1 to 5 antennas randomly for regular jobs
            resources = random.sample(["DSS-24", "DSS-26", "DSS-34", "DSS-36", "DSS-54"], k=random.randint(1, 5))

        resource_vp_dict = {res: generate_view_periods(time_window_start, time_window_end) for res in resources}
        
        track = {
            "subject": 521,
            "user": "521_0",
            "week": week,
            "year": year,
            "duration": duration,
            "duration_min": duration,
            "resources": [[res] for res in resources],
            "track_id": track_id,
            "setup_time": setup_time,
            "teardown_time": teardown_time,
            "time_window_start": time_window_start,
            "time_window_end": time_window_end,
            "resource_vp_dict": resource_vp_dict
        }
        data.append(track)
    return data


# Generate DSN data for a range of weeks and a given year
def generate_dsn_data(weeks, year, num_tracks_per_week):
    dsn_data = {}
    for week in weeks:
        week_key = f"W{week}_{year}"
        dsn_data[week_key] = generate_track_data(week, year, num_tracks_per_week)
    return dsn_data

if __name__ == "__main__":
    weeks = [10]
    year = 2018
    num_tracks_per_week = 50
    dsn_data = generate_dsn_data(weeks, year, num_tracks_per_week)
    
    with open("Antenas/build/toy_problem.json", "w") as file:
        json.dump(dsn_data, file, indent=2)
    
    print("DSN test data generated successfully.")


# %%
def read_jobs_from_file(filename, jobs, sync_jobs):
    with open(filename, 'r') as file:
        j = json.load(file)

    for item_key, item_value in j.items():
        if item_key != "W10_2018":
            continue
        for req in item_value:
            job = Job()
            job.track_id = req["track_id"]
            job.subject = req["subject"]
            job.week = req["week"]
            job.year = req["year"]
            job.duration = req["duration"]
            job.duration_min = req["duration_min"]
            job.setup_time = req["setup_time"]
            job.teardown_time = req["teardown_time"]
            job.time_window_start = req["time_window_start"]
            job.time_window_end = req["time_window_end"]

            for antenna_name, vp_list in req["resource_vp_dict"].items():
                print(antenna_name)
                if '_' in antenna_name:
                    print(f"Antenna {antenna_name} has multiple antennas")
                    single_antennas = antenna_name.split('_')
                    for single_antenna in single_antennas:
                        add_view_periods_to_job(job, single_antenna, vp_list)
                    sync_jobs.append(job.track_id)
                else:
                    add_view_periods_to_job(job, antenna_name, vp_list)

            jobs.append(job)

def distribute_jobs_to_antennas(jobs):
    antenna_jobs = {}

    for job in jobs:
        # print(f"Distributing job {job.track_id} to antennas")
        for antenna, view_periods in job.antenna_view_periods.items():
            # print(f"Antenna {antenna}")
            if antenna not in antenna_jobs:
                antenna_jobs[antenna] = []
            antenna_jobs[antenna].append(job)

    return antenna_jobs

# %%
class ViewPeriod:
    def __init__(self):
        self.rise = None
        self.set = None
        self.trx_on = None
        self.trx_off = None

class Job:
    def __init__(self):
        self.track_id = ""
        self.subject = 0
        self.week = 0
        self.year = 0
        self.duration = 0.0
        self.duration_min = 0.0
        self.setup_time = 0
        self.teardown_time = 0
        self.time_window_start = 0
        self.time_window_end = 0
        self.antenna_view_periods = {}

def add_view_periods_to_job(job, antenna_name, vp_list):
    for vp in vp_list:
        view_period = ViewPeriod()
        if "RISE" in vp: view_period.rise = vp["RISE"]
        if "SET" in vp: view_period.set = vp["SET"]
        if "TRX ON" in vp: view_period.trx_on = vp["TRX ON"]
        if "TRX OFF" in vp: view_period.trx_off = vp["TRX OFF"]
        if antenna_name not in job.antenna_view_periods:
            job.antenna_view_periods[antenna_name] = []
        job.antenna_view_periods[antenna_name].append(view_period)


# %%
jobs = []
synchronized_jobs = []
print("Reading jobs from file")
read_jobs_from_file("Antenas/build/toy_problem.json", jobs, synchronized_jobs)
print(synchronized_jobs)
print(len(synchronized_jobs))
antenna_jobs = distribute_jobs_to_antennas(jobs)
antennas = list(antenna_jobs.keys())

# %%
# get each job.track_id from antenna_jobs
all_jobs = []
for antenna in antenna_jobs:
    for job in antenna_jobs[antenna]:
        all_jobs.append(job.track_id)

# check if there are any duplicate track_ids
all_jobs = set(all_jobs)

# %%
antennas

# %%
for antenna in antennas:
    print(f"Antenna {antenna}")
    for job in antenna_jobs[antenna]:
        print(f"  Job {job.track_id}")
        for antenna_name, vp_list in job.antenna_view_periods.items():
            print(f"    Antenna {antenna_name}")
            for vp in vp_list:
                print(f"      Rise {vp.rise} Set {vp.set}")

# # %%
# import sys
# import os
# sys.path.append(os.path.abspath("build"))


# %%
import pybaldes as baldes


# %%
from collections import defaultdict

# Dictionary to store nodes_baldes for each antenna
antenna_nodes = defaultdict(list)

# Iterate over each antenna and their corresponding jobs
for antenna in antennas:
    print(f"Antenna {antenna}")

    # Create a new list of nodes for this antenna
    antenna_nodes[antenna] = []
    
    # Iterate over jobs for the current antenna
    for job in antenna_jobs[antenna]:
        #print(f"  Job {job.track_id}")

        # Access the view periods of the job for this specific antenna
        if antenna in job.antenna_view_periods:
            node = baldes.VRPNode()
            node.duration = job.duration  # Customize as needed
            node.identifier = str(job.track_id)  # Customize as needed
            node.cost = 0  # Customize as needed
            node.demand = 0  # Customize as needed
            node.consumption = [0]  # Customize as needed
            node.set_location(random.randint(0, 100), random.randint(0, 100))  # Set random location
            node.id = len(antenna_nodes[antenna])  # Unique ID within this antenna list
            vp_list = job.antenna_view_periods[antenna]
            node.mtw_lb = []
            node.mtw_ub = []
            
            # Create nodes for each view period and assign attributes
            for vp in vp_list:
                #print(f"    Rise {vp.rise}, Set {vp.set}")
                
                # Create a new VRPNode and assign rise and set times
                node.mtw_lb = node.mtw_lb + [vp.rise]
                node.mtw_ub = node.mtw_ub + [vp.set]
                #print(f"      Node {node.id} -> lb: {node.mtw_lb[-1]}, ub: {node.mtw_ub[-1]}")
                
            # Append the node to the list for the current antenna
            antenna_nodes[antenna].append(node)
                
                #print(f"      Node {node.id} -> lb: {node.lb[0]}, ub: {node.ub[0]}, duration: {node.duration}")

# Example: Accessing nodes for a specific antenna
for antenna, nodes in antenna_nodes.items():
    print(f"\nNodes for Antenna {antenna}:")
    for node in nodes:
        print(f"  Node {node.id}: lb={node.mtw_lb}, ub={node.mtw_ub}, duration={node.duration}")

#####################################################
# Greedy Heuristic
#####################################################

def greedy_heuristic(nodes):
    # Sort nodes based on their earliest lower bound (mtw_lb)
    sorted_nodes = sorted(nodes, key=lambda node: min(node.mtw_lb))
    
    selected_jobs = []
    selected_node_ids = set()  # Keep track of selected node IDs to avoid repetition
    last_end_time = 0  # Keeps track of the latest end time of the selected jobs

    for node in sorted_nodes:
        if node.id in synchronized_jobs:
            continue
        # Skip nodes that have already been selected
        if node.id in selected_node_ids:
            continue

        # Try to select a feasible time window for the current node
        for i in range(len(node.mtw_lb)):
            start_time = node.mtw_lb[i]
            end_time = node.mtw_ub[i]
            
            # Check if the current time window is feasible
            if start_time >= last_end_time:
                # Select this time window
                selected_jobs.append({
                    "node_id": node.id,
                    "selected_lb": start_time,
                    "selected_ub": end_time,
                    "duration": node.duration,
                    "identifier": node.identifier
                })
                
                # Mark the node as selected
                selected_node_ids.add(node.id)
                
                # Update the last end time
                last_end_time = end_time
                break  # Move to the next node after selecting one time window

    return selected_jobs

initial_solution = {}
for antenna, nodes in antenna_nodes.items():
    print(f"\nNodes for Antenna {antenna}:")
    selected_jobs = greedy_heuristic(nodes)
    initial_solution[antenna] = selected_jobs

    # Display the selected jobs
    for job in selected_jobs:
        print(f"Selected Job - Node {job['node_id']}: Start {job['selected_lb']}, End {job['selected_ub']}, Duration {job['duration']}")

# create array of job['identifier'] for each antenna
antenna_initial_jobs = {}
for antenna in antennas:
    antenna_initial_jobs[antenna] = []
    for job in initial_solution[antenna]:
        antenna_initial_jobs[antenna].append(job['identifier'])

colunas = {}
for antenna in antennas:
    colunas[antenna] = []

for antenna in initial_solution:
    sol = []

    for job in initial_solution[antenna]:
        sol.append(job['identifier'])
    colunas[antenna].append(sol)

########################################################
# Define master problem
########################################################

def create_master_problem(colunas):
    # create master problem with gurobipy
    import gurobipy as gp
    master = gp.Model()
    master.Params.OutputFlag = 0

    # create one variable for each job in initial solution of each antenna
    x = {}
    for i in range(len(antennas)):
        name = antennas[i]
        
        counter = 0
        for col in colunas[name]:
            n_jobs = len(col)
            x[name, counter] = master.addVar(lb=0, ub=1, vtype=gp.GRB.CONTINUOUS, name=f"x_{name}_{counter}", obj=n_jobs)
            counter += 1

    master.update()
    y = {}
    for job in synchronized_jobs:
        y[job] = master.addVar(lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name=f"y_{job}")
    # add constraints to ensure that each job is attended at most once
    ctrs = []
    for job in all_jobs:
        if job in synchronized_jobs:
            # Sum of x variables for this job across all antennas
            lhs = 0
            for antenna in antennas:
                counter = 0
                for col in colunas[antenna]:
                    if str(job) in col:
                        lhs += x[antenna, counter]
                    counter += 1

            # Add constraints to enforce synchronization
            # If y[job] == 1, sum of x variables must be 2
            # If y[job] == 0, sum of x variables must be 0
            ctrs.append(master.addConstr(lhs >= 2 * y[job], name=f"sync_{job}"))
        else:
            lhs = 0;
            for antenna in antennas:
                counter = 0
                for col in colunas[antenna]:
                    if str(job) in col:
                        lhs += x[antenna, counter]
                    counter += 1
            ctrs.append(master.addConstr(lhs <= 1, name=f"job_{job}_once"))

    # add constraints to ensure that each antenna has at most one counter
    for antenna in antennas:
        lhs = 0
        counter = 0
        for col in colunas[antenna]:
            lhs += x[antenna, counter]
            counter += 1
        ctrs.append(master.addConstr(lhs <= 1, name=f"antenna_{antenna}_once"))


    # set objective as maximization
    master.modelSense = gp.GRB.MAXIMIZE
    return master

master = create_master_problem(colunas)
###############################################
# Get Duals
###############################################

duals = {}
for constr in master.getConstrs():
    duals[constr.ConstrName] = constr.Pi

################################################
# Solve the bucket graph for each antenna
################################################

import numpy as np
buckets = {}
nos = {}
for antenna in antennas:
    print(antenna)
    nodes = antenna_nodes[antenna]
    
    maximo = 0
    for node in nodes:
        print(node.mtw_ub)
        aux = np.max(node.mtw_ub)
        if aux > maximo:
            maximo = aux
    print(maximo)
    # create noode "start" to represent the start of the route
    start = baldes.VRPNode()
    start.duration = 0
    start.identifier = 'start'
    start.cost = 0
    start.demand = 0
    start.consumption = [0]
    start.set_location(0, 0)
    start.mtw_lb = [0]
    start.mtw_ub = [maximo]
    start.lb = [0]
    start.ub = [maximo]

    nodes = [start] + nodes
    # create noode "end" to represent the end of the route
    end = baldes.VRPNode()
    end.duration = 0
    end.identifier = 'end'
    end.cost = 0
    end.demand = 0
    end.consumption = [0]
    end.set_location(0, 0)
    end.mtw_lb = [0]
    end.mtw_ub = [maximo]
    end.lb = [0]
    end.ub = [maximo]

    # add start and end nodes to the list of nodes
    nodes.append(end)

    # generate distance matrix between all nodes, with 60 minutes as the time to travel between nodes
    distance_matrix = []
    for i in range(len(nodes)):
        row = []
        for j in range(len(nodes)):
            row.append(0)
        distance_matrix.append(row)

    # set the main diagonal as 0
    for i in range(len(nodes)):
        distance_matrix[i][i] = 0

    pos_end = len(nodes) - 1
    options = baldes.BucketOptions()
    options.depot = 0
    options.end_depot = pos_end
    options.size = len(nodes)
    options.resource_type = [3]
    options.bucket_fixing = False

    # for each node, define lb as the lowest mtw_lb and ub as the highest mtw_ub
    for node in nodes:
        node.lb = [min(node.mtw_lb)]
        node.ub = [max(node.mtw_ub)]


    counter = 0
    for node in nodes:
        node.id = counter
        #print lb and ub for each node
        counter += 1

    bg = baldes.BucketGraph(nodes, int(maximo)+1000, 1)
    bg.setOptions(options)
    bg.set_distance_matrix(distance_matrix)
    bg.setup()

    nos[antenna] = nodes
    buckets[antenna] = bg

def map_to_identifier(coluna,all_jobs,antenna):
    sol = []
    nodes = nos[antenna]
    for item in coluna:
        identificador = nodes[item].identifier
        if identificador == 'start' or identificador == 'end':
            continue
        sol.append(int(identificador))
    return sol
################################################
# Solve the bucket graph for each antenna
################################################

print(len(all_jobs))
for i in range(1):

    print('-----------------------------------')
    print('ITERATION', i)

    master = create_master_problem(colunas)
    master.optimize()
    duals = [ctr.Pi for ctr in master.getConstrs()]

    for var in master.getVars():
        if var.x > 0:
            print(f"{var.varName} = {var.x}")

    print("Obj Val", master.objVal)

    cols_added = 0
    for antenna in antennas:
        print(antenna)
        duals_antenna = []
        for node in nos[antenna]:
            if (node.identifier == 'start' or node.identifier == 'end'):
                continue
            if node.identifier in synchronized_jobs:
                duals_antenna.append(duals[int(node.identifier)])
            else:
                duals_antenna.append(duals[int(node.identifier)])
        bg = buckets[antenna]
        bg.set_duals(duals_antenna)
        print(duals_antenna)
        labels = bg.solve()
        for label in labels:
            print(label.nodes_covered())
            coluna = map_to_identifier(label.nodes_covered(),all_jobs,antenna)
            if len(coluna) == 0:
                continue
            print('label', label.nodes_covered)
            print('coluna', coluna)
            colunas[antenna].append(coluna)
            cols_added += 1
    if cols_added == 0:
        break

