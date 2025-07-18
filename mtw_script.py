import pandas as pd

# Function to parse the time windows column
def parse_time_windows(tw_str):
    if isinstance(tw_str, str):
        tws = tw_str.split(';')
        time_windows = []
        for tw in tws:
            if ',' in tw:
                start, end = tw.split(',')
                time_windows.append((int(start), int(end)))
            else:
                time_windows.append((int(tw), int(tw))) 
        return time_windows
    else:
        return [] 

# Reading the file and parsing the data
def read_instance_file(file_path):
    df = pd.read_csv(file_path, delimiter='\t')
    df['TWS'] = df['TWS'].apply(parse_time_windows)
    return df


import copy
import numpy as np
import math
#Define a customer class
class Customer:
    def __init__(self,custumer_number=0, x=0.0, y=0.0, demand=0, duration=0, time_windows=None, associated_custumer=None):
        self.custumer_number = custumer_number
        self.x = x
        self.y = y
        self.demand = demand
        self.duration = duration
        self.time_windows = time_windows if time_windows is not None else []
        self.associated_custumer = associated_custumer

    # def __repr__(self):
    #     return (
    #         f"Job(custumer_no={self.custumer_number}, x_coord={self.x}, ycoord={self.y}, "
    #         f"demand={self.demand}, time_windows={self.time_windows}, associated_custumer={self.associated_custumer})"
    #     )
    def __repr__(self):
        return (
            f"Job(custumer_no={self.custumer_number})"
        )
    

def read_jobs_from_instance_file(file_path):
    # Define the columns expected in the file
    #columns = ['XCOORD', 'YCOORD', 'DEMAND', 'TWNUM', 'TWS']
    
    # Read the file into a dataframe
    df = pd.read_csv(file_path, delimiter='\t')
    for i in range(len(df)):
        df.at[i, 'CUST NO.'] = i
    # List to store all jobs
    custumers = []
    # Iterate over the dataframe rows and create Job objects
    for _, row in df.iterrows():
        time_windows = parse_time_windows(row['TWS'])
        custumer = Customer(
            custumer_number=int(row['CUST NO.']),
            x=row['XCOORD.'],
            y=row['YCOORD.'],
            demand=row['DEMAND'],
            duration=1,
            time_windows=time_windows
        )
        custumers.append(custumer)

    return custumers

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def create_distance_matrix(custumers):
    n = len(custumers)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i, j] = euclidean_distance(
                    custumers[i].x, custumers[i].y, custumers[j].x, custumers[j].y
                )
    return distance_matrix

custumers = read_jobs_from_instance_file('SVRP-SPM-master/RD01N10.txt')
distance_matrix = create_distance_matrix(custumers)
custumers[2].time_windows

import math
import random
import matplotlib.pyplot as plt
import networkx as nx


class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = {}
        self.pairs = set()
        self.nodes_with_loads = []  # Expanded nodes with loads
        self.edges_with_loads = {}  # Expanded edges with loads
        self.distance_matrix = {} 
        self.demand_expansion_map = {} 

    def euclidean_distance(self, node1, node2):
        """
        Computes the Euclidean distance between two nodes.
        """
        return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

    def compute_distance_matrix(self, use_load_nodes=False):
        """
        Computes the distance matrix and stores it as a class attribute.

        Parameters:
            use_load_nodes (bool): If True, uses `self.nodes_with_loads` for the distance matrix,
                                   otherwise uses `self.nodes`.
        """
        nodes = self.nodes_with_loads if use_load_nodes else self.nodes
        self.distance_matrix = {}  # Initialize an empty dictionary for the distance matrix

        for i, node1 in enumerate(nodes):
            self.distance_matrix[node1.custumer_number] = {}
            for j, node2 in enumerate(nodes):
                if i == j:
                    self.distance_matrix[node1.custumer_number][node2.custumer_number] = 0
                else:
                    dist = self.euclidean_distance(node1, node2)
                    self.distance_matrix[node1.custumer_number][node2.custumer_number] = dist

    def get_distance(self, from_node, to_node):
        """
        Retrieves the distance between two nodes from the distance matrix.

        Parameters:
            from_node (Customer): The starting node.
            to_node (Customer): The destination node.

        Returns:
            float: The distance between from_node and to_node.
        """
        return self.distance_matrix.get(from_node.custumer_number, {}).get(to_node.custumer_number, float('inf'))
    
    def phi(self, u):
        """
        Returns the associated customer of a load vertex u.

        Parameters:
            u (Customer): The load vertex or customer node.

        Returns:
            int: The customer number of the associated original customer.
        """
        # If `u` has an `associated_custumer` (meaning it's an expanded node),
        # return that. Otherwise, return `u`'s own custumer_number.
        return u.associated_custumer if u.associated_custumer is not None else u.custumer_number

    def add_nodes_from_customers(self, customers):
        """
        Adds customers to the graph's nodes, with Customer 0 as the start depot and an identical end depot.
        """
        self.custumers = customers[1:]
        self.nodes = custumers
        self.edges = {node: [] for node in self.nodes}

        # Add a duplicate of Customer 0 as the last node (end depot)
        if self.nodes:
            end_depot = copy.deepcopy(self.nodes[0])
            end_depot.custumer_number = len(self.nodes)  # Give it a unique number as the end depot
            self.nodes.append(end_depot)
            self.edges[end_depot] = []

    def add_edge(self, from_node, to_node, weight):
        # Store edges as tuples (to_node, weight) where to_node is a Customer object
        self.edges[from_node].append((to_node, weight))

    def get_edges(self, node):
        return self.edges[node]

    def euclidean_distance(self, node1, node2):
        """
        Computes the Euclidean distance between two nodes, which we use as travel time.
        """
        return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

    def create_custumer_graph(self):
        for cust_A in self.nodes[:-1]:  # Skip the end depot
            for cust_B in self.nodes[:-1]:  # Skip the end depot
                if cust_A.custumer_number != cust_B.custumer_number:
                    for tw1_start, tw1_end in cust_A.time_windows:
                        for tw2_start, tw2_end in cust_B.time_windows:
                            if tw2_start >= tw1_start:
                                transition_time = self.euclidean_distance(cust_A, cust_B)
                                self.add_edge(cust_A, cust_B, transition_time)
                                self.pairs.add((cust_A.custumer_number, cust_B.custumer_number))

        # Connect the end depot to all expanded nodes with the same distances as start depot
        self.start_depot = self.nodes[0]
        self.end_depot = self.nodes[-1]
        for node in self.nodes[1:-1]:  # Skip depots
            depot_weight = self.euclidean_distance(self.start_depot, node)
            self.add_edge(self.start_depot, node, depot_weight)
            self.add_edge(node, self.start_depot, depot_weight)
            self.add_edge(self.end_depot, node, depot_weight)
            self.add_edge(node, self.end_depot, depot_weight)

    def create_load_vertices(self):
        self.nodes_with_loads = []  # Initialize expanded nodes with loads
        self.edges_with_loads = {}  # Initialize expanded edges with loads

        # Use the first and last depot nodes
        self.nodes_with_loads.append(self.start_depot)
        self.edges_with_loads[self.start_depot] = []
        self.edges_with_loads[self.end_depot] = []  # Initialize edges for end_depot

        self.demand_expansion_map = {}

        # Expand all nodes except the depots
        for node in self.nodes[1:-1]:  # Skip the first and last nodes (depots)
            expanded_nodes = []
            for i in range(1, node.demand + 1):
                new_node = Customer(
                    custumer_number=f"{node.custumer_number}_{i}",
                    x=node.x,
                    y=node.y,
                    demand=i,
                    duration=node.duration,
                    time_windows=node.time_windows,
                    associated_custumer=node.custumer_number
                )
                expanded_nodes.append(new_node)
                self.nodes_with_loads.append(new_node)
                self.edges_with_loads[new_node] = []

                # Add edges between start and end depots and each expanded node
                depot_weight_start = self.euclidean_distance(self.start_depot, node)
                depot_weight_end = self.euclidean_distance(self.end_depot, node)
                self.edges_with_loads[self.start_depot].append((new_node, depot_weight_start))
                self.edges_with_loads[new_node].append((self.start_depot, depot_weight_start))
                self.edges_with_loads[self.end_depot].append((new_node, depot_weight_end))
                self.edges_with_loads[new_node].append((self.end_depot, depot_weight_end))

            self.demand_expansion_map[node] = expanded_nodes

        # Use a set to track all added edges globally across nodes
        added_edges = set()

        # Replicate edges only between expanded nodes based on the original connections
        for original_from_node, edges in self.edges.items():
            for original_to_node, weight in edges:
                expanded_from_nodes = self.demand_expansion_map.get(original_from_node, [])
                expanded_to_nodes = self.demand_expansion_map.get(original_to_node, [])
                
                for expanded_from_node in expanded_from_nodes:
                    for expanded_to_node in expanded_to_nodes:
                        # Define a unique key for each edge using custumer_number
                        edge_key = (expanded_from_node.custumer_number, expanded_to_node.custumer_number)
                        if edge_key not in added_edges:
                            self.edges_with_loads[expanded_from_node].append((expanded_to_node, weight))
                            added_edges.add(edge_key)

        # Add the end depot as the final node in nodes_with_loads
        self.nodes_with_loads.append(self.end_depot)
    

    def get_subnodes(self, node):
        """
        Returns the sub-nodes (load nodes) associated with a given customer node.

        Parameters:
            node (Customer): The original customer node for which sub-nodes are required.

        Returns:
            list: A list of sub-nodes associated with the given node.
        """
        return self.demand_expansion_map.get(node, [])
    
    def V_plus(self, u, g):
        """
        Computes the succeeding set V^+(u) for a given node u.

        Parameters:
            u (Customer): The load vertex for which to compute the succeeding set.
            g (float): The time cost of delivering 1 unit of demand.
            
        Returns:
            set: The succeeding set V^+(u).
        """
        succeeding_set = set()
        u_customer = self.phi(u)  # Associated customer of u

        # Minimum start time for the service window of customer associated with u
        min_service_time_u = min(ei for ei, li in u.time_windows)

        for v in self.nodes_with_loads:
            v_customer = self.phi(v)  # Associated customer of v
            if v_customer == u_customer:
                continue  # Skip nodes associated with the same customer as u

            # Fetch precomputed travel time from u to v
            travel_time_uv = self.get_distance(u, v)
            
            # Minimum and maximum service times for v's customer
            max_service_time_v = max(li - g * v.demand for ei, li in v.time_windows)

            # Condition for v to be in V^+(u)
            if min_service_time_u + g * u.demand + travel_time_uv <= max_service_time_v:
                succeeding_set.add(v)
        
        return succeeding_set

    def V_minus(self, u, g):
        """
        Computes the preceding set V^-(u) for a given node u.

        Parameters:
            u (Customer): The load vertex for which to compute the preceding set.
            g (float): The time cost of delivering 1 unit of demand.
            
        Returns:
            set: The preceding set V^-(u).
        """
        preceding_set = set()
        u_customer = self.phi(u)  # Associated customer of u

        # Minimum and maximum service times for u's customer
        min_service_time_u = min(ei for ei, _ in u.time_windows)
        max_service_time_u = max(li - g * u.demand for _, li in u.time_windows)

        for v in self.nodes_with_loads:
            v_customer = self.phi(v)  # Associated customer of v
            if v_customer == u_customer:
                continue  # Skip nodes associated with the same customer as u

            # Fetch precomputed travel time from v to u
            travel_time_vu = self.get_distance(v, u)
            
            # Minimum service time for v's customer
            min_service_time_v = min(ei for ei, _ in v.time_windows)

            # Condition for v to be in V^-(u)
            if min_service_time_v + g * v.demand + travel_time_vu <= max_service_time_u:
                preceding_set.add(v)
        
        return preceding_set


# Sample usage with debugging
graph = Graph()
graph.add_nodes_from_customers(custumers)  # Assuming 'custumers' is a list of Customer objects
graph.create_custumer_graph()
graph.create_load_vertices()

# Get the sub-nodes for a specific node
target_node = graph.nodes[1]  # Example: choose the first customer after the depot
subnodes = graph.get_subnodes(target_node)
# print("Sub-nodes for customer node", target_node.custumer_number, ":", subnodes)

## test V_plus
# Choose a load vertex to test V^+(u)
u = subnodes[0]  # Choose the first sub-node of the target node
g = 1  # Assume the time cost of delivering 1 unit of demand
succeeding_set = graph.V_plus(u, g)
print("Succeeding set V^+(", u.custumer_number, "):", succeeding_set)

# Assuming graph is an instance of Graph with nodes added and expanded nodes created

# Compute the distance matrix using either the original nodes or expanded nodes
graph.compute_distance_matrix(use_load_nodes=True)

print("Distance matrix:")
for i, row in graph.distance_matrix.items():
    print(i, ":", row)


import gurobipy as gp
from gurobipy import GRB

class VRPModel:
    def __init__(self, graph, num_vehicles):
        self.graph = graph
        self.model = gp.Model("Vehicle Routing Problem with Time Windows")
        self.num_vehicles = num_vehicles
        self.g = 1
        self.Q = 100


        self.x = {}  # x_{u,v,k}: Binary variable for edge traversal
        self.z = {}  # z_{u,w,k}: Binary variable for visiting vertex with time window selection
        self.y = {}  # y_{i,w}: Binary variable for time window selection
        self.tau = {}  # tau_{u,k}: Continuous variable for service starting time

        self._initialize_variables()

    def _initialize_variables(self):
        for u in self.graph.nodes_with_loads:
            for v in self.graph.nodes_with_loads:
                if u != v:  # Only add if u and v are different
                    for k in range(self.num_vehicles):
                        self.x[u.custumer_number, v.custumer_number, k] = self.model.addVar(
                            vtype=GRB.BINARY,
                            name=f"x_{u.custumer_number}_{v.custumer_number}_{k}"
                        )

        for u in self.graph.nodes_with_loads:
            for w, _ in enumerate(u.time_windows):  # Use index w for each time window
                for k in range(self.num_vehicles):
                    self.z[u.custumer_number, w, k] = self.model.addVar(
                        vtype=GRB.BINARY,
                        name=f"z_{u.custumer_number}_{w}_{k}"
                        )   

        # y_{i,w}: 1 if time window w is selected for customer i
        for i in self.graph.custumers:
            for w, _ in enumerate(i.time_windows):  # Use index w for each time window
                self.y[i.custumer_number, w] = self.model.addVar(
                    vtype=GRB.BINARY,
                    name=f"y_{i.custumer_number}_{w}"
                )

        # tau_{u,k}: Service starting time of vehicle k at load vertex u
        for u in self.graph.nodes_with_loads:
            for k in range(self.num_vehicles):
                self.tau[u.custumer_number, k] = self.model.addVar(
                    vtype=GRB.CONTINUOUS,
                    name=f"tau_{u.custumer_number}_{k}"
                )

        # Integrate the new variables into the model
        self.model.update()

    def set_objective(self):
        # Objective: Minimize the total travel cost
        objective = gp.quicksum(
            self.graph.get_distance(u, v) * self.x[u.custumer_number, v.custumer_number, k]
            for k in range(self.num_vehicles)
            for u in self.graph.nodes_with_loads if u != self.graph.end_depot
            for v in self.graph.V_plus(u, g=1) if v != self.graph.start_depot  # Assume g is 1 if not defined
            if v != u  # Avoid self-loops
        )
        
    def demand_constraint(self):
    
        for custumer in self.graph.custumers:
             self.model.addConstr(gp.quicksum(
                u.demand * self.z[u.custumer_number, w, k]
                for k in range(self.num_vehicles)
                for u in self.graph.get_subnodes(custumer)
                for w, _ in enumerate(u.time_windows))
                >= custumer.demand, f"demand_constraint_{custumer.custumer_number}")
    
    def route_structure_contraint(self):
        num_vehicles = self.num_vehicles

        # 1. Each vehicle starts from the depot (node 0) exactly once
        for k in range(num_vehicles):
            self.model.addConstr(
                gp.quicksum(self.x[self.graph.start_depot.custumer_number, u.custumer_number, k] 
                            for u in self.graph.nodes_with_loads if u != self.graph.start_depot) == 1,
                name=f"start_from_depot_vehicle_{k}"
            )

        for custumer in self.graph.custumers:
            for k in range(num_vehicles):
                for u in self.graph.get_subnodes(custumer):
                    outgoing_flow = gp.quicksum(self.x[u.custumer_number, v.custumer_number, k] for v in self.graph.V_plus(u, self.g) if v != self.graph.start_depot)
                    incoming_flow = gp.quicksum(self.x[v.custumer_number, u.custumer_number, k] for v in self.graph.V_minus(u, self.g) if v != self.graph.end_depot)
                    self.model.addConstr(outgoing_flow == incoming_flow, f"flow_conservation_{u.custumer_number}_{k}")
                    self.model.addConstr(outgoing_flow <= 1, f"visit_once_{u.custumer_number}_{k}")

        # 3. Each vehicle ends at the final depot (node n+1) exactly once
        for k in range(num_vehicles):
            self.model.addConstr(
                gp.quicksum(self.x[u.custumer_number, self.graph.end_depot.custumer_number, k] 
                            for u in self.graph.nodes_with_loads if u != self.graph.end_depot) == 1,
                name=f"end_at_depot_vehicle_{k}"
            )

        #4. Each customer can only be visited at most once by the same vehicle
        for custumer in self.graph.custumers:
            for k in range(num_vehicles):
                self.model.addConstr(gp.quicksum(self.x[u.custumer_number,v.custumer_number,k]
                                     for u in self.graph.get_subnodes(custumer)
                                     for v in self.graph.V_plus(u, self.g)) <= 1, f"customer_{custumer.custumer_number}_vehicle_{k}_visit_at_most_once")
        #5. x and z link
        for custumer in self.graph.custumers:
            for k in range(num_vehicles):
                for u in self.graph.get_subnodes(custumer):
                    self.model.addConstr(gp.quicksum(self.x[u.custumer_number,v.custumer_number,k]
                                                     for v in self.graph.V_plus(u, self.g) if v != self.graph.start_depot)
                                        == gp.quicksum(self.z[u.custumer_number,w,k] for w, _ in enumerate(custumer.time_windows)), f"u_z_link_{u.custumer_number}_vehicle_{k}")
                    self.model.addConstr(gp.quicksum(self.z[u.custumer_number,w,k] for w, _ in enumerate(custumer.time_windows)) <= 1)
        #6. vehicle capacity
        for k in range(num_vehicles):
            self.model.addConstr(gp.quicksum(self.z[u.custumer_number,w,k] * u.demand
                                             for custumer in self.graph.custumers
                                             for u in self.graph.get_subnodes(custumer)
                                             for w, _ in enumerate(custumer.time_windows)) <= self.Q, f"vehicle_capacity_{k}")
    
    def synchronization_constraints(self):
        for custumer in self.graph.custumers:
            self.model.addConstr(gp.quicksum(self.y[custumer.custumer_number, w] for w, _ in enumerate(custumer.time_windows)) == 1, f"synchronization_constraint_1_{custumer.custumer_number}")

        for custumer in self.graph.custumers:
            for k in range(self.num_vehicles):
                for w in range(len(custumer.time_windows)):
                    for u in self.graph.get_subnodes(custumer):
                        self.model.addConstr(self.z[u.custumer_number, w, k] <= self.y[custumer.custumer_number, w], f"synchronization_constraint_2_{custumer.custumer_number}_{u.custumer_number}_{k}_{w}")
                        self.model.addConstr(self.z[u.custumer_number, w, k] >= self.y[custumer.custumer_number, w]
                                             + gp.quicksum(self.x[u.custumer_number,v.custumer_number,k] for v in self.graph.V_plus(u, self.g) if v != self.graph.start_depot) - 1, f"synchronization_constraint_3_{custumer.custumer_number}_{u.custumer_number}_{k}_{w}")

    def add_time_window_constraints(self):
        M = 1e6  # Large constant for big-M constraints

        for custumer in self.graph.custumers + [self.graph.start_depot]:
            for k in range(self.num_vehicles):
                for u in self.graph.get_subnodes(custumer):
                    for v in self.graph.V_plus(u, self.g):
                        if v != self.graph.start_depot:
                            travel_time = self.graph.get_distance(u, v)
                            self.model.addConstr(self.tau[v.custumer_number,k] >= self.tau[u.custumer_number,k] + self.g * u.demand + travel_time + M * (self.x[u.custumer_number, v.custumer_number,k] - 1),
                                                 name=f"time_consistency_{u.custumer_number}_{v.custumer_number}_{k}")

        for custumer in self.graph.custumers:
            for k in range(self.num_vehicles):
                for u in self.graph.get_subnodes(custumer):
                    for w, (e_w, _) in enumerate(custumer.time_windows):
                        self.model.addConstr(self.tau[u.custumer_number, k] >= e_w + M * (self.z[u.custumer_number, w, k] -1), name=f"start_time_window_{u.custumer_number}_{w}_{k}")

        for custumer in self.graph.custumers:
            for k in range(self.num_vehicles):
                for w, (_, l_w) in enumerate(custumer.time_windows):
                    for u in self.graph.get_subnodes(custumer):
                        self.model.addConstr(self.tau[u.custumer_number, k] + self.g * u.demand <= l_w + M * (1 - self.z[u.custumer_number, w, k]), name=f"end_time_window_{u.custumer_number}_{w}_{k}")
    
    def solve(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            print("Optimal solution found.")
        elif self.model.status == GRB.INFEASIBLE:
            print("Model is infeasible. Computing IIS...")
            self.model.computeIIS()  # Compute the IIS
            print("The following constraints and variables contribute to the infeasibility:")
            for c in self.model.getConstrs():
                if c.IISConstr:  # Check if the constraint is part of the IIS
                    print(f"Infeasible constraint: {c.ConstrName}")
            for v in self.model.getVars():
                if v.IISLB or v.IISUB:  # Check if the variable bounds are part of the IIS
                    bound_type = "lower bound" if v.IISLB else "upper bound"
                    print(f"Infeasible {bound_type} for variable: {v.VarName}")
        elif self.model.status == GRB.UNBOUNDED:
            print("Model is unbounded.")
        else:
            print(f"Optimization ended with status {self.model.status}")
    
    def print_routes(self):
        if self.model.status != GRB.OPTIMAL:
            print("No optimal solution available to print routes.")
            return

        for k in range(self.num_vehicles):
            route = []
            current_node = self.graph.start_depot
            while current_node != self.graph.end_depot:
                route.append(current_node.custumer_number)
                # Find the next node in the route
                next_node = None
                for v in self.graph.nodes_with_loads:
                    if (current_node.custumer_number, v.custumer_number, k) in self.x:
                        if self.x[current_node.custumer_number, v.custumer_number, k].x > 0.5:
                            next_node = v
                            break
                if next_node is None:
                    print(f"No valid route found for vehicle {k}")
                    break
                current_node = next_node
            
            # Add end depot to complete the route
            route.append(self.graph.end_depot.custumer_number)
            print(f"Route for vehicle {k}: {route}")
                            
        

num_vehicles = 5  # Example
vrp_model = VRPModel(graph, num_vehicles)   
vrp_model.set_objective()
vrp_model.demand_constraint()
vrp_model.route_structure_contraint()
vrp_model.synchronization_constraints()
vrp_model.add_time_window_constraints()

# Solve the model
vrp_model.solve()

# Print the routes for each vehicle
vrp_model.print_routes()
