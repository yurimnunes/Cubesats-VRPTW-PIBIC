import pybaldes as baldes
import json
import random
import uuid
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict



# # Convert a datetime to minutes since epoch
# def datetime_to_minutes(dt):
#     return int(dt.timestamp() // 60)

# # Generate random minutes within a given range
# def generate_random_minutes(start_minutes, end_minutes):
#     return int(random.uniform(start_minutes, end_minutes))

# # Generate view periods for resources
# def generate_view_periods(start_minutes, end_minutes, num_periods=3):
#     periods = []
#     period_duration = (end_minutes - start_minutes) // num_periods
#     for i in range(num_periods):
#         rise = start_minutes + period_duration * i
#         set_time = rise + int(random.uniform(60, 240))  # Duration between 1 to 4 hours in minutes
#         periods.append({
#             "RISE": rise,
#             "SET": set_time,
#             "TRX ON": rise,
#             "TRX OFF": set_time
#         })
#     return periods

# # Generate track data for a given week and year
# def generate_track_data(week, year, num_tracks):
#     start_week = datetime.strptime(f'{year} {week} 1', '%Y %W %w')
#     start_week_minutes = 0
#     data = []
#     for _ in range(num_tracks):
#         duration = int(random.uniform(60, 2*60))  # Duration between 1 to 3.5 hours in minutes
#         setup_time = 60
#         teardown_time = 15
#         track_id = str(uuid.uuid4())
#         time_window_start = start_week_minutes + generate_random_minutes(0, 6.5 * 24 * 60 * 1)  # Random start within the week
#         time_window_end = time_window_start + generate_random_minutes(1 * 60, 1.5 * 60)  # Random end 1 to 2 hours later
        
#         if random.random() < 0.0:
#             resources = random.sample(["DSS-24", "DSS-26", "DSS-34", "DSS-36", "DSS-54"], k=2)
#             # concatenate name with _
#             resources = ["_".join(resources)]
#         else:
#             resources = random.sample(["DSS-24", "DSS-26", "DSS-34", "DSS-36", "DSS-54"], k=random.randint(1, 5))
#         resource_vp_dict = {res: generate_view_periods(time_window_start, time_window_end) for res in resources}
        
#         track = {
#             "subject": 521,
#             "user": "521_0",
#             "week": week,
#             "year": year,
#             "duration": duration,
#             "duration_min": duration,
#             "resources": [[res] for res in resources],
#             "track_id": track_id,
#             "setup_time": setup_time,
#             "teardown_time": teardown_time,
#             "time_window_start": time_window_start,
#             "time_window_end": time_window_end,
#             "resource_vp_dict": resource_vp_dict
#         }
#         data.append(track)
#     return data


# # Generate DSN data for a range of weeks and a given year
# def generate_dsn_data(weeks, year, num_tracks_per_week):
#     dsn_data = {}
#     for week in weeks:
#         week_key = f"W{week}_{year}"
#         dsn_data[week_key] = generate_track_data(week, year, num_tracks_per_week)
#     return dsn_data

# weeks = [1]
# year = 2018
# num_tracks_per_week = 20
# dsn_data = generate_dsn_data(weeks, year, num_tracks_per_week)

# with open("Antenas/build/toy_problem.json", "w") as file:
#     json.dump(dsn_data, file, indent=2)

#     print("DSN test data generated successfully.")

# import json

class Instance:
    def __init__(self):
        self.tracks = []          # List of track IDs
        self.resources = set()    # Set of antenna resources
        self.track_nodes = {}     # {track_id: node_data}
        self.sync_groups = []     # Groups requiring simultaneous scheduling

    def load_data(self, file_path):
        """Load and parse DSN scheduling data from JSON file"""
        self.__init__()
        with open(file_path) as f:
            raw_data = json.load(f)
            
        for week_data in raw_data.values():
            for track in week_data:
                self._process_track(track)

    def _process_track(self, track):
        """Create node with embedded resource-timewindow data"""
        track_id = track['track_id']
        resource_windows = {}
        current_sync_groups = []

        # Process each resource group
        for res_group in track['resources']:
            res = res_group[0]  # Get resource string
            
            if '_' in res:  # Handle sync groups
                antennas = res.split('_')
                current_sync_groups.append(antennas)
                
                # Add windows to each antenna in group
                combined_periods = track['resource_vp_dict'].get(res, [])
                for antenna in antennas:
                    self.resources.add(antenna)
                    periods = [(p['TRX ON'], p['TRX OFF']) for p in combined_periods]
                    resource_windows.setdefault(antenna, []).extend(periods)
            else:  # Single antenna
                self.resources.add(res)
                periods = [(p['TRX ON'], p['TRX OFF']) 
                          for p in track['resource_vp_dict'].get(res, [])]
                resource_windows[res] = periods

        # Update global sync groups
        for group in current_sync_groups:
            if group not in self.sync_groups:
                self.sync_groups.append(group)

        # Store track data
        self.track_nodes[track_id] = {
            'duration': track['duration_min'],
            'setup': track['setup_time'],
            'teardown': track['teardown_time'],
            'mission_window': (track['time_window_start'], track['time_window_end']),
            'resource_windows': resource_windows,
            'sync_groups': current_sync_groups
        }
        self.tracks.append(track_id)

# Create and load instance
# dsn_instance = Instance()
# dsn_instance.load_data("build/toy_problem.json")
# dsn_instance.load_data("build/dsn_schedule.json")

import itertools

class Graph:
    def __init__(self, instance):
        self.instance = instance
        self.nodes = {}       # Track nodes + virtual nodes
        self.edges = {}       # Edges with travel times
        self.resources = set()

        self._build_nodes()
        self._build_edges()

    def _build_nodes(self):
        """Add virtual start/end nodes and store resource windows in track nodes"""
        # Regular track nodes with resource-specific time windows
        for track_id in self.instance.tracks:
            track_data = self.instance.track_nodes[track_id]
            self.nodes[track_id] = {
                'type': 'track',
                'duration': track_data['duration'],
                'setup': track_data['setup'],
                'teardown': track_data['teardown'],
                'mission_window': track_data['mission_window'],
                'resource_windows': track_data['resource_windows'],
                'resources': list(track_data['resource_windows'].keys())
            }
            self.resources.update(track_data['resource_windows'].keys())

        # Virtual nodes for each antenna
        for antenna in self.resources:
            vs_node = f"vs_{antenna}"
            ve_node = f"ve_{antenna}"
            
            self.nodes[vs_node] = {
                'type': 'virtual_start',
                'setup': 0,
                'teardown': 0,
                'duration': 0,
                'time_window': (0, float('inf')),
                'resources': [antenna]
            }
            
            self.nodes[ve_node] = {
                'type': 'virtual_end',
                'setup': 0,
                'teardown': 0,
                'duration': 0,
                'time_window': (0, float('inf')),
                'resources': [antenna]
            }

    def _get_travel_time(self, u, v):
        """Calculate transition time between nodes"""
        if self.nodes[u]['type'] == 'virtual_start':
            return self.nodes[v]['setup']
        elif self.nodes[v]['type'] == 'virtual_end':
            return self.nodes[u]['teardown']
        return self.nodes[u]['teardown'] + self.nodes[v]['setup']

    def _can_follow(self, u, v, antenna):
        """Check if v can follow u on antenna, considering resource windows"""
        if 'virtual' in u or 'virtual' in v:
            return False

        # Both tracks must support this antenna
        if antenna not in self.nodes[u]['resource_windows'] or \
           antenna not in self.nodes[v]['resource_windows']:
            return False

        # Check time window compatibility
        for u_window in self.nodes[u]['resource_windows'][antenna]:
            u_trx_on, u_trx_off = u_window
            u_total_end = u_trx_on + self.nodes[u]['duration']
            min_v_start = u_total_end + self._get_travel_time(u, v)

            for v_window in self.nodes[v]['resource_windows'][antenna]:
                v_trx_on, v_trx_off = v_window
                latest_v_start = v_trx_off - self.nodes[v]['duration']

                if min_v_start <= latest_v_start:
                    return True
        return False

    def _build_edges(self):
        """Build edges considering resource-specific windows"""
        for antenna in self.resources:
            vs_node = f"vs_{antenna}"
            ve_node = f"ve_{antenna}"

            # Add idle edge
            self._add_edge(vs_node, ve_node, antenna, 0)

            # Get tracks supporting this antenna
            tracks_on_a = [
                t for t in self.instance.tracks 
                if antenna in self.nodes[t]['resource_windows']
            ]

            # Track-to-track edges
            for u, v in itertools.permutations(tracks_on_a, 2):
                if self._can_follow(u, v, antenna):
                    self._add_edge(u, v, antenna, self._get_travel_time(u, v))

            # Connect virtual nodes
            for track in tracks_on_a:
                if self._can_start(track, antenna):
                    self._add_edge(vs_node, track, antenna, self.nodes[track]['setup'])
                self._add_edge(track, ve_node, antenna, self.nodes[track]['teardown'])

    def _can_start(self, track, antenna):
        """Check if track can be first on antenna"""
        setup_time = self.nodes[track]['setup']
        return any(
            setup_time <= window[0] 
            for window in self.nodes[track]['resource_windows'][antenna]
        )

    def _add_edge(self, u, v, antenna, travel_time):
        """Store edge with transition metadata"""
        key = (u, v)
        if key not in self.edges:
            self.edges[key] = {}
        self.edges[key][antenna] = travel_time

    def get_feasible_successors(self, u, antenna):
        """Get all valid next nodes (including ve_a)"""
        successors = []
        for (src, dest), antennas in self.edges.items():
            if src == u and antenna in antennas:
                successors.append(dest)
        return successors

    def get_feasible_predecessors(self, v, antenna):
        """Get all valid previous nodes (including vs_a)"""
        predecessors = []
        for (src, dest), antennas in self.edges.items():
            if dest == v and antenna in antennas:
                predecessors.append(src)
        return predecessors
    
    def build_distance_matrix(self):
        """Build a distance matrix for the graph"""
        num_nodes = len(self.nodes)
        distance_matrix = []
        for i in range(num_nodes):
            row = []
            for j in range(num_nodes):
                if i == j:
                    row.append(0)
                else:
                    u = list(self.nodes.keys())[i]
                    v = list(self.nodes.keys())[j]
                    travel_time = self._get_travel_time(u, v)
                    row.append(travel_time)
            distance_matrix.append(row)
        return distance_matrix
    
    def print_graph(self):
        """Print graph structure with nodes and edges, including resource windows"""
        print("="*40)
        print("Graph Nodes:")
        for node_id, data in self.nodes.items():
            node_type = data['type']
            if node_type == 'track':
                print(f"\nTrack {node_id}:")
                print(f"  Duration: {data['duration']} min")
                print(f"  Setup: {data['setup']} min | Teardown: {data['teardown']} min")
                print(f"  Mission Window: {data['mission_window'][0]}–{data['mission_window'][1]}")
                print("  Resource Windows:")
                for resource, windows in data['resource_windows'].items():
                    window_str = ", ".join([f"{start}–{end}" for start, end in windows])
                    print(f"    {resource}: {window_str}")
                print(f"  Sync Groups: {self.instance.track_nodes[node_id].get('sync_groups', [])}")
            else:
                print(f"\nVirtual {node_type.split('_')[-1]} Node {node_id}:")
                print(f"  Resources: {data['resources']}")

        print("\n" + "="*40)
        print("Graph Edges:")
        for (src, dest), antennas in self.edges.items():
            for antenna, travel_time in antennas.items():
                arrow = f"[{antenna}] {travel_time}min"
                print(f"{src: <20} —— {arrow: <25} → {dest}")

instance = Instance()
instance.load_data("Antenas/build/toy_problem.json")
graph = Graph(instance)
graph.print_graph()

from collections import defaultdict
import numpy as np
import gurobipy as gp
import pybaldes as baldes
import random

class ColumnGenerationScheduler:
    def __init__(self, graph):
        self.graph = graph
        self.antenna_jobs = self._distribute_jobs_to_antennas()
        self.antennas = list(self.antenna_jobs.keys())
        self.all_jobs = self._get_all_jobs()
        self.solution = {}
        
        # Data structures
        self.antenna_nodes = defaultdict(list)
        self.colunas = defaultdict(list)
        self.buckets = {}
        self.nos = {}
        
        # Build initial solution
        self._create_vrp_nodes()
        self._run_greedy_heuristic()
        self._create_initial_columns()
        self._setup_bucket_graphs()

    def _distribute_jobs_to_antennas(self):
        """Map graph nodes to antenna-specific jobs"""
        antenna_jobs = defaultdict(list)
        for track_id in self.graph.nodes:
            node_data = self.graph.nodes[track_id]
            if node_data['type'] != 'track':
                continue
            
            for antenna in node_data['resources']:
                antenna_jobs[antenna].append({
                    'track_id': track_id,
                    'duration': node_data['duration'],
                    'resource_windows': node_data['resource_windows'][antenna],
                    'setup': node_data['setup'],
                    'teardown': node_data['teardown']
                })
        return antenna_jobs

    def _get_all_jobs(self):
        return {job['track_id'] for antenna in self.antennas for job in self.antenna_jobs[antenna]}

    def _create_vrp_nodes(self):
        """Create VRP nodes using graph data"""
        for antenna in self.antennas:
            self.antenna_nodes[antenna] = []
            for job in self.antenna_jobs[antenna]:
                node = baldes.VRPNode()
                node.duration = job['duration']
                node.identifier = job['track_id']
                node.mtw_lb = [tw[0] for tw in job['resource_windows']]
                node.mtw_ub = [tw[1] for tw in job['resource_windows']]
                node.set_location(random.randint(0, 100), random.randint(0, 100))
                node.cost = 0  
                node.demand = 0  
                node.consumption = [0] 
                node.id = len(self.antenna_nodes[antenna])
                self.antenna_nodes[antenna].append(node)
                print(f"Node {node.identifier} created for antenna {antenna} with windows {node.mtw_lb} to {node.mtw_ub}")
        

    def _run_greedy_heuristic(self):
        """Generate initial solution using graph-compatible greedy heuristic"""
        self.already_selected = []
        self.initial_solution = {}
        for antenna, nodes in self.antenna_nodes.items():
            sorted_nodes = sorted(nodes, key=lambda n: min(n.mtw_lb))
            selected = []
            last_end = 0
            antenna_counter = 0
            for node in sorted_nodes:
                if antenna_counter >= 1:
                    break
                if node.identifier in self.already_selected:
                    continue
                for i in range(len(node.mtw_lb)):
                    if node.mtw_lb[i] >= last_end:
                        selected.append({
                            'identifier': node.identifier,
                            'start': node.mtw_lb[i],
                            'end': node.mtw_ub[i]
                        })
                        self.already_selected.append(node.identifier)
                        last_end = node.mtw_ub[i]
                        antenna_counter += 1
                        break
            self.initial_solution[antenna] = selected
            print(f"Greedy solution for antenna {antenna}: {selected}")
        print("Greedy heuristic completed.")

    def _create_initial_columns(self):
        """Initialize columns from greedy solution"""
        for antenna in self.antennas:
            self.colunas[antenna] = [
                [job['identifier'] for job in self.initial_solution[antenna]]]
                
    def _setup_bucket_graphs(self):
        """Build bucket graphs using graph edges"""
        for antenna in self.antennas:
            vs_node = f"vs_{antenna}"
            ve_node = f"ve_{antenna}"
            track_nodes = self.antenna_nodes[antenna]
            ub = max(
                max(node.mtw_ub) for node in track_nodes
            ) + 1000
            print(f"Max time for antenna {antenna}: {ub}")
            # Create virtual nodes from graph
            start = baldes.VRPNode()
            start.duration = 0
            start.identifier = 'start'
            start.cost = 0
            start.demand = 0
            start.consumption = [0]
            start.set_location(0, 0)
            start.mtw_lb = [0]
            start.mtw_ub = [ub]
            
            end = baldes.VRPNode()
            end.duration = 0
            end.identifier = 'end'
            end.cost = 0
            end.demand = 0
            end.consumption = [0]
            end.set_location(0, 0)
            end.mtw_lb = [0]
            end.mtw_ub = [ub]

            full_nodes = [start] + track_nodes + [end]
            
            # Build distance matrix from graph edges
            size = len(full_nodes)
            distance_matrix = self.graph.build_distance_matrix()
            distance_matrix = [[0 for _ in range(size)] for _ in range(size)]
            # for i in range(size):
            #     for j in range(size):
            #         if i != j:
            #             distance_matrix[i][j] = random.randint(1, 100)

            # Configure bucket graph
            options = baldes.BucketOptions()
            options.depot = 0
            options.end_depot = len(full_nodes)-1
            options.size = len(full_nodes)
            options.resource_type = [3]
            options.bucket_fixing = False
            
            max_time = max(
                max(node.mtw_ub) for node in track_nodes
             for track in self.graph.instance.tracks) + 1000
            
            for node in full_nodes:
                node.lb = [min(node.mtw_lb)]
                node.ub = [max(node.mtw_ub)]

            counter = 0
            for node in full_nodes:
                node.id = counter
                counter += 1
            
            bg = baldes.BucketGraph(full_nodes, int(ub)+10000, 1)
            bg.setOptions(options)
            bg.set_distance_matrix(distance_matrix)
            bg.setup()
            
            self.buckets[antenna] = bg
            self.nos[antenna] = full_nodes

    def _create_master_problem(self):
        """Build master problem with graph-compatible constraints"""
        master = gp.Model()
        master.Params.OutputFlag = 0
        
        # Create variables
        x = {}
        for antenna in self.antennas:
            for idx, col in enumerate(self.colunas[antenna]):
                obj_value = sum(
                    1 for job in col 
                    if job not in {"start", "end"}
                )
                x[(antenna, idx)] = master.addVar(
                    vtype=gp.GRB.CONTINUOUS,
                    name=f"x_{antenna}_{idx}",
                    obj=obj_value
                )

        # Job coverage constraints
        for job in self.all_jobs:
            lhs = gp.quicksum(
                x[(antenna, idx)]
                for antenna in self.antennas
                for idx, col in enumerate(self.colunas[antenna])
                if job in col
            )
            master.addConstr(lhs <= 1, f"job_{job}")

        # Antenna constraints
        for antenna in self.antennas:
            master.addConstr(
                gp.quicksum(x[(antenna, idx)] 
                for idx in range(len(self.colunas[antenna]))) <= 1,
                f"antenna_{antenna}"
            )

        master.modelSense = gp.GRB.MAXIMIZE
        master.write("master.lp")
        return master

    def run_column_generation(self, max_iterations=50):
        """Run column generation with graph-based pricing"""
        for iter in range(max_iterations):
            print("Iter-----------------------------------------:", iter)  
            master = self._create_master_problem()
            master.optimize()
            
            if master.status != gp.GRB.OPTIMAL:
                break

            # Get dual values
            duals = {c.ConstrName: c.Pi for c in master.getConstrs()}
            self.solution = {
                v.VarName: v.X for v in master.getVars() if v.X > 0
            }
            self.master_obj = master.ObjVal
            print("Master objective value:", self.master_obj)
            
            # Solve pricing problems
            added = 0
            for antenna in self.antennas:
                dual_values = [
                    duals.get(f"job_{node.identifier}", 0)
                    for node in self.nos[antenna][1:-1]
                ]
                print(dual_values)
                # if iter == 0: 
                #     # dual_values = [2000 for i in range(len(dual_values))]
                print("Dual values:", dual_values)
                print("Number of nodes:", len(self.nos[antenna]))
                self.buckets[antenna].set_duals(dual_values)
                labels = []
                labels = self.buckets[antenna].solve()
                print(labels)
                for label in labels:
                    print(label.nodes_covered())
                    column = [
                        self.nos[antenna][i].identifier 
                        for i in label.nodes_covered()
                        if self.nos[antenna][i].identifier not in {f"vs_{antenna}", f"ve_{antenna}"}
                    ]
                    if column and column not in self.colunas[antenna]:
                        self.colunas[antenna].append(column)
                        added += 1
            
            if added == 0:
                break

    
    def print_instance(self):
        """Print the instance with antenna jobs and columns"""
        print("\n=== Antenna Jobs ===")
        for antenna, jobs in self.antenna_jobs.items():
            print(f"\nAntenna {antenna}:")
            for job in jobs:
                print(f"  Track ID: {job['track_id']}, Duration: {job['duration']}min, Windows: {job['resource_windows']}")
        
        print("\n=== Columns ===")
        for antenna, cols in self.colunas.items():
            print(f"\nAntenna {antenna}:")
            for col in cols:
                print(f"  Column: {col}")

    def print_nodes(self):
        """Print the nodes of the bucket graphs"""
        for antenna, nodes in self.nos.items():
            print(f"\nNodes for antenna {antenna}:")
            for node in nodes:
                print(f"  Node ID: {node.id}, Identifier: {node.identifier}, LB: {node.lb}, UB: {node.ub}")
        print("\n=== Bucket Graphs ===")
        for antenna, bg in self.buckets.items():
            print(f"\nBucket graph for antenna {antenna}:")
            # print(f"  Size: {bg.size}, Max time: {bg.max_time}")
            # print(f"  Depot: {bg.depot}, End depot: {bg.end_depot}")
            # print(f"  Resource type: {bg.resource_type}")
            # print(f"  Bucket fixing: {bg.bucket_fixing}")

instance = Instance()
instance.load_data("Antenas/build/toy_problem.json")
graph = Graph(instance)
scheduler = ColumnGenerationScheduler(graph)
scheduler.print_instance()
scheduler.run_column_generation()
scheduler.print_nodes()
