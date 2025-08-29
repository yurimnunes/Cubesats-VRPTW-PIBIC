#%%
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np

class Graph:
    def __init__(self, instance):
        self.instance = instance
        self.nodes = {}       # Track nodes + virtual nodes
        self.edges = {}       # Edges with travel times
        self.resources = set()
        self._build_nodes()
        self._build_edges()
        self._antennas_jobs_dict()

    def _antennas_jobs_dict(self):
        """Create a dictionary mapping antennas to their jobs"""
        self.antennas_jobs = {}
        for antenna in self.resources:
            self.antennas_jobs[antenna] = []
            for track_id in self.instance.tracks:
                if antenna in self.instance.track_nodes[track_id]['resource_windows']:
                    self.antennas_jobs[antenna].append(track_id)
        return self.antennas_jobs

    def print_nodes_and_instance(self):
        """Print nodes and instance data for debugging"""
        print("Instance Data:")
        print(self.instance.track_nodes)
        print("\nGraph Nodes:")
        for node_id, data in self.nodes.items():
            print(f"{node_id}: {data}")
        print("\nAntennas Jobs Dictionary:")
        print(self.antennas_jobs)

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
        # return 0
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
            # for track in tracks_on_a:
            #     if self._can_start(track, antenna):
            #         self._add_edge(vs_node, track, antenna, self.nodes[track]['setup'])
            #     self._add_edge(track, ve_node, antenna, self.nodes[track]['teardown'])
            for track in tracks_on_a:
                if self._can_start(track, antenna):
                    self._add_edge(vs_node, track, antenna, 0) # TEmporario
                self._add_edge(track, ve_node, antenna, 0)

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


from gurobipy import Model, GRB, quicksum

class OptimizationProblem:
    def __init__(self, graph):
        self.graph = graph
        self.model = Model("DSN_Scheduling")
        self.M = 1e5 # Big M 
        
        # Decision variables from Table 2

        # var y podera ser removida, e o modelo de tracks ser alterado considerando a saida de var x
        self.y = {}          # y_i (track scheduled)
        self.x = {}          # x^a_{u,v} (antenna flow)
        self.z = {}          # z^a_{u,w} (window selection)
        self.tau = {}        # τ_{u,a} (start time)
        
        self._create_variables()
        self._build_constraints()

    def _create_variables(self):
        """Create all decision variables"""
        # Activity scheduling variables
        for track in self.graph.instance.tracks:
            self.y[track] = self.model.addVar(vtype=GRB.BINARY, name=f"y_{track}")

        # Antenna flow variables
        for (u, v), antennas in self.graph.edges.items():
            for a in antennas:
                self.x[(u, v, a)] = self.model.addVar(
                    vtype=GRB.BINARY, name=f"x_{u}_{v}_{a}"
                )

        # Time window selection variables
        for track in self.graph.instance.tracks:
            for a in self.graph.nodes[track]['resources']:
                windows = self.graph.nodes[track]['resource_windows'][a]
                for w_idx in range(len(windows)):
                    self.z[(track, a, w_idx)] = self.model.addVar(
                        vtype=GRB.BINARY, name=f"z_{track}_{a}_{w_idx}"
                    )

        # Start time variables
        for track in self.graph.instance.tracks:
            for a in self.graph.nodes[track]['resources']:
                self.tau[(track, a)] = self.model.addVar(
                    vtype=GRB.CONTINUOUS, name=f"tau_{track}_{a}"
                )

    def _build_constraints(self):
        """Build all constraints"""
        self._build_flow_constraints()
        print("Flow constraints built.")
        self._build_window_selection_constraints()
        print("Window selection constraints built.")
        self._build_timing_constraints()
        print("Timing constraints built.")
        # self._build_sync_constraints()
        print("Sync constraints built.")

    def _build_sync_constraints(self):
        """Constraints for synchronized antenna groups"""
        for track_id in self.graph.instance.tracks:
            track_data = self.graph.instance.track_nodes[track_id]
            
            # Skip non-synchronized tracks
            if not track_data['sync_groups']:
                continue
            else:
                print(f"Sync groups for track {track_id}: {track_data['sync_groups']}")
                
            # For each sync group (e.g., ["DSS-26", "DSS-24"])
            for group in track_data['sync_groups']:
                print(f"Processing sync group {group} for track {track_id}")
                a1, a2 = group

                # 1. Both antennas must be assigned to this track
                self.model.addConstr(
                    quicksum(self.x[(u, track_id, a1)] 
                    for u in self.graph.get_feasible_predecessors(track_id, a1))
                    == self.y[track_id],
                    name=f"sync_assign_{track_id}_{a1}"
)
                
                self.model.addConstr(
                    quicksum(self.x[(u, track_id, a2)] 
                    for u in self.graph.get_feasible_predecessors(track_id, a2))
                    == self.y[track_id],
                    name=f"sync_assign_{track_id}_{a2}"
                )
                
                self.model.addConstr(
                    self.tau[(track_id, a1)] == self.tau[(track_id, a2)],
                    name=f"sync_time_{track_id}"
                )

                # 3. Same window selection (using window indices)
                windows_a1 = track_data['resource_windows'][a1]
                windows_a2 = track_data['resource_windows'][a2]
                for w in range(len(windows_a1)):
                    self.model.addConstr(
                        self.z[(track_id, a1, w)] == self.z[(track_id, a2, w)],
                        name=f"sync_window_{track_id}_{w}"
                    )
                    


    def _build_flow_constraints(self):
        """Antenna path and flow conservation constraints"""
        # Original flow constraints from previous implementation
        for antenna in self.graph.resources:
            vs = f"vs_{antenna}"
            ve = f"ve_{antenna}"
            
            # Start/end node constraints
            self.model.addConstr(
                quicksum(self.x[(vs, v, antenna)] for v in self.graph.get_feasible_successors(vs, antenna)) == 1,
                name=f"start_{antenna}"
            )
            self.model.addConstr(
                quicksum(self.x[(u, ve, antenna)] for u in self.graph.get_feasible_predecessors(ve, antenna)) == 1,
                name=f"end_{antenna}"
            )
            # Flow conservation constraints

        # Link flow variables to activity scheduling
        for u in self.graph.nodes:
            if self.graph.nodes[u]['type'] == 'track':
                for a in self.graph.nodes[u]['resources']:
                    self.model.addConstr(
                        quicksum(self.x[(v, u, a)] for v in self.graph.get_feasible_predecessors(u,a))
                        == quicksum(self.x[(u, v, a)] for v in self.graph.get_feasible_successors(u,a)),
                        name=f"flow_conservation_{u}_{a}"
                    )
                    ###### Essa Constraint pode ser removida... porem em fluxo deve ser garantido
                    self.model.addConstr(
                        quicksum(self.x[(u, v, a)] for v in self.graph.get_feasible_successors(u,a)) <= self.y[u],
                        name=f"flow_conservation_{u}_{a}_less_than_1"
                    )
            
        
        #### Isso pode ser removido tambem.
        for u in self.graph.nodes:
            if self.graph.nodes[u]['type'] == 'track':
                track_data = self.graph.instance.track_nodes[u]
                if not track_data['sync_groups']:
                    self.model.addConstr(
                        quicksum(self.x[(u, v, a)] for a in self.graph.nodes[u]['resources'] for v in self.graph.get_feasible_successors(u,a)) 
                        >= self.y[u],
                        name=f"activity_scheduling_{u}"
                    )
                else:
                    self.model.addConstr(
                        quicksum(self.x[(u, v, a)] for a in self.graph.nodes[u]['resources'] for v in self.graph.get_feasible_successors(u,a)) 
                        == 2 * self.y[u],
                        name=f"activity_scheduling_{u}"
                    )

    # Pode ser removido tambem, porem precisa ajustar calculos posteriores de time windows
    def _build_window_selection_constraints(self):
        """Time window selection constraints"""
        for u in self.graph.nodes:
            if self.graph.nodes[u]['type'] == 'track':
                for a in self.graph.nodes[u]['resources']:
                    windows = self.graph.nodes[u]['resource_windows'][a]
                    self.model.addConstr(
                        quicksum(self.z[(u, a, w)] for w in range(len(windows))) 
                        == quicksum(self.x[(u, v, a)] for v in self.graph.get_feasible_successors(u,a)),
                        name=f"window_select_{u}_{a}"
                    )
        for u in self.graph.nodes:
            if self.graph.nodes[u]['type'] == 'track':
                    # windows = self.graph.nodes[u]['resource_windows'][a]
                    self.model.addConstr(
                        quicksum(self.z[(u, a, w)] for a in self.graph.nodes[u]['resources'] for w in range(len(self.graph.nodes[u]['resource_windows'][a]))) 
                        <= self.y[u],
                        name=f"window_select_{u}_less_than_1"
                    )
        

    def _build_timing_constraints(self):
        """Timing constraints with window enforcement"""
        for u in self.graph.nodes:
            if self.graph.nodes[u]['type'] == 'track':
                for a in self.graph.nodes[u]['resources']:
                    for v in self.graph.get_feasible_successors(u, a):
                        if self.graph.nodes[v]['type'] == 'track':
                            travel_time = self.graph.edges[(u, v)][a]
                            # for w_idx, (trx_on, trx_off) in enumerate(self.graph.nodes[u]['resource_windows'][a]):
                            self.model.addConstr(
                                self.tau[(v, a)] >= self.tau[(u, a)] + self.graph.nodes[u]['duration'] + travel_time + self.M * (self.x[(u, v, a)] - 1),
                                #name=f"timing_{u}_{v}_{a}_{w_idx}"
                                name=f"timing_{u}_{v}_{a}"
                                )
                            
        for u in self.graph.nodes:
            if self.graph.nodes[u]['type'] == 'track':
                for a in self.graph.nodes[u]['resources']:
                    for w_idx, (trx_on, trx_off) in enumerate(self.graph.nodes[u]['resource_windows'][a]):
                        self.model.addConstr(
                            self.tau[(u, a)] >= trx_on + self.M * (self.z[(u, a, w_idx)] - 1),
                            name=f"window_start_{u}_{a}_{w_idx}"
                        )
                        self.model.addConstr(
                            self.tau[(u, a)] + self.graph.nodes[u]['duration'] <= trx_off + self.M * (1 - self.z[(u, a, w_idx)]),
                            name=f"window_end_{u}_{a}_{w_idx}"
                        )

    def set_objective(self):
        """Example objective: Maximize total scheduled duration"""
        total_duration = quicksum(
            self.y[track] 
            # * self.graph.nodes[track]['duration']
            for track in self.graph.instance.tracks
        )
        self.model.setObjective(total_duration, GRB.MAXIMIZE)

    def solve(self):
        """Solve and return status"""
        self.model.write("original.lp")
        self.model.optimize()
        return self.model.status
    
    def print_solution(self):
        """Print the solution with sync track handling"""
        if self.model.status != GRB.OPTIMAL:
            print("No optimal solution found.")
            return

        def time_str(mins):
            return f"{int(mins//60):02d}:{int(mins%60):02d}"

        print("\n=== Optimal Schedule ===")
        scheduled_sync_tracks = set()

        # Print synchronized tracks first
        print("\nSYNCHRONIZED TRACKS:")
        for track_id in self.graph.instance.tracks:
            # Get data from GRAPH nodes, not instance
            track_node = self.graph.nodes[track_id]
            if not track_node.get('sync_groups') or self.y[track_id].X < 0.5:
                continue

            scheduled_sync_tracks.add(track_id)
            group = track_node['sync_groups'][0]
            a1, a2 = group[0], group[1]

            # Get common timing information
            start = self.tau[(track_id, a1)].X
            end = start + track_node['duration']
            w_idx = next(i for i in range(len(track_node['resource_windows'][a1])) 
                        if self.z[(track_id, a1, i)].X > 0.5)
            window = track_node['resource_windows'][a1][w_idx]

            print(f"\n🚀 Track {track_id} (SYNC)")
            print(f"  Antennas: {', '.join(group)}")
            print(f"  Start: {time_str(start)}")
            print(f"  End: {time_str(end)}")
            print(f"  Duration: {track_node['duration']}min")
            print(f"  Selected Window: {time_str(window[0])}-{time_str(window[1])}")

        # Print regular tracks
        print("\nREGULAR TRACKS:")
        for track_id in self.graph.instance.tracks:
            if self.y[track_id].X < 0.5 or track_id in scheduled_sync_tracks:
                continue

            # Get data from GRAPH nodes
            track_node = self.graph.nodes[track_id]
            print(f"\n📡 Track {track_id}")
            print(f"  Duration: {track_node['duration']}min")
            # Access resources from graph node
            # print(f"  Resource Windows:")
            # print("Track node resources:")
            # print(track_node['resources'])
            # print("Track node resource windows:")
            # print(track_node['resource_windows'])
            # print("z variable for this track node")
            
            # for a in track_node['resources']:
            #     "print all z variables related to this track node"
            #     for w_idx in range(len(track_node['resource_windows'][a])):
            #         print(f"z_{track_id}_{a}_{w_idx}: {self.z[(track_id, a, w_idx)].X}")
            for a in track_node['resources']:
                # for w_idx, (trx_on, trx_off) in enumerate(track_node['resource_windows'][a]):
                #     print(f"    {a} - Window {w_idx}: {time_str(trx_on)}-{time_str(trx_off)}")
                if any(self.x[(u, track_id, a)].X > 0.5 for u in self.graph.get_feasible_predecessors(track_id, a)):
                    # print("Predecessors:")
                    # for u in self.graph.get_feasible_predecessors(track_id, a):
                    #     print(f"  {u}")
                    # print("Successors:")
                    # for v in self.graph.get_feasible_successors(track_id, a):
                    #     print(f"  {v}")
                    w_idx = next(i for i in range(len(track_node['resource_windows'][a])) 
                            if self.z[(track_id, a, i)].X > 0.5)
                    start = self.tau[(track_id, a)].X
                    end = start + track_node['duration']
                    window = track_node['resource_windows'][a][w_idx]
                    
                    print(f"  On {a}:")
                    print(f"    TRX: {time_str(start)}-{time_str(end)}")
                    print(f"    Window: {time_str(window[0])}-{time_str(window[1])}")

    def print_antenna_paths(self):
        """Print antenna paths with sync track awareness in a circular plot spanning 168 hours"""
        if self.model.status != GRB.OPTIMAL:
            print("No solution found")
            return

        def time_str(mins):
            """Formata o tempo para HH:MM (apenas horário, sem dias)"""
            hours = int(mins // 60)
            minutes = int(mins % 60)
            return f"{hours:02d}:{minutes:02d}"

        print("\nANTENNA PATHS:")
        
        # Configuração do plot circular
        fig, ax = plt.subplots(figsize=(16, 16), subplot_kw={'projection': 'polar'})
        
        # Cores para diferentes antenas
        colors = plt.cm.tab10.colors
        antenna_colors = {}
        
        # Para cada antena, vamos desenhar seu caminho
        for idx, antenna in enumerate(self.graph.resources):
            # Atribuir uma cor para esta antena
            antenna_colors[antenna] = colors[idx % len(colors)]
            
            current_node = f"vs_{antenna}"
            print(f"\n📻 {antenna} Timeline:")
            print(f"╠═ {current_node} [Start]")
            
            # Lista para armazenar os pontos do caminho
            angles = []
            radii = []
            labels = []
            
            # Adicionar o nó virtual de início
            angles.append(0)  # Ângulo 0 para o início
            radii.append(idx + 1)  # Raio baseado no índice da antena
            labels.append(f"{antenna} Start")
            
            while True:
                # Encontrar próximo nó
                next_node = None
                for (u, v, a) in self.x:
                    if u == current_node and a == antenna and self.x[(u, v, a)].X > 0.5:
                        next_node = v
                        break

                if not next_node or next_node.startswith("ve_"):
                    break

                # Processar nó de track
                track_id = next_node
                track_data = self.graph.instance.track_nodes[track_id]
                start = self.tau[(track_id, antenna)].X
                end = start + track_data['duration']
                
                # Converter tempo para ângulo (0-168 horas para 0-2π)
                angle = (start / (168*60)) * 2 * math.pi  # 168 horas em minutos
                
                angles.append(angle)
                radii.append(idx + 1)  # Mesmo raio para esta antena
                
                # Formatar label como solicitado: apenas horário (HH:MM-HH:MM)
                labels.append(f"{time_str(start)}-{time_str(end)}")
                
                # Informações para console (mantemos com dias para debug)
                def console_time_str(mins):
                    days = int(mins // (24*60))
                    hours = int((mins % (24*60)) // 60)
                    minutes = int(mins % 60)
                    return f"D{days} {hours:02d}:{minutes:02d}"
                    
                w_idx = next(i for i in range(len(track_data['resource_windows'][antenna])) 
                        if self.z[(track_id, antenna, i)].X > 0.5)
                window = track_data['resource_windows'][antenna][w_idx]
                
                sync_note = ""
                if track_data['sync_groups']:
                    sync_partners = [a for a in track_data['sync_groups'][0] if a != antenna]
                    sync_note = f" (SYNC with {', '.join(sync_partners)})"

                print(f"╠═ Track {track_id}{sync_note}")
                print(f"║  ├─ Window: {console_time_str(window[0])}-{console_time_str(window[1])}")
                print(f"║  ├─ TRX: {console_time_str(start)}-{console_time_str(end)}")
                print(f"║  └─ Duration: {track_data['duration']}min")
                
                current_node = track_id

            print(f"╚═ ve_{antenna} [End]")
            
            # Plotar o caminho desta antena
            ax.plot(angles, radii, 'o-', color=antenna_colors[antenna], label=antenna, linewidth=2, markersize=8)
            
            # Adicionar labels aos pontos (apenas para tracks, não para o início)
            for i, (angle, radius, label) in enumerate(zip(angles, radii, labels)):
                if i >= 0:  # Não mostrar label para o ponto de início
                    # Determinar a posição do texto para evitar sobreposição
                    text_radius = radius + 0.15
                    
                    # Alternar entre inside e outside para evitar sobreposição
                    if i % 2 == 0:
                        text_radius = radius - 0.15
                        va = 'top'
                    else:
                        text_radius = radius + 0.15
                        va = 'bottom'
                    
                    ax.text(angle, text_radius, label, 
                        ha='center', va=va, fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=antenna_colors[antenna], alpha=0.8))
        
        # Configurar o gráfico polar
        ax.set_theta_offset(math.pi/2)  # 0° no topo
        ax.set_theta_direction(-1)  # Sentido horário
        
        # Configurar os ticks do ângulo (dias da semana)
        days = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
        day_ticks = np.linspace(0, 2*math.pi, 7, endpoint=False)
        ax.set_xticks(day_ticks)
        ax.set_xticklabels(days)
        
        # Adicionar linhas de grade para os dias
        ax.grid(True, which='major', axis='x', linestyle='-', alpha=0.7)
        
        # Configurar os ticks do raio (antenas)
        ax.set_yticks(range(1, len(self.graph.resources) + 1))
        ax.set_yticklabels([f'Antena {a}' for a in self.graph.resources])
        ax.set_ylim(0, len(self.graph.resources) + 1)
        
        # Adicionar legenda
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Adicionar título
        plt.title('Cronograma Semanal das Antenas (168 horas)', pad=20, fontsize=16)
        
        # Ajustar layout
        plt.tight_layout()
        plt.show() 
    def print_variables(self):
        """Print all decision variables with non-zero/selected values"""
        if self.model.status != GRB.OPTIMAL:
            print("No solution to print.")
            return

        print("\n=== Decision Variables ===")
        
        # Track scheduling variables (y)
        print("\nTrack Scheduling (y):")
        for track, var in self.y.items():
            if var.X > 0.5:
                print(f"{var.VarName}: {var.X}")

        # Antenna flow variables (x)
        print("\nAntenna Flow (x):")
        for (u, v, a), var in self.x.items():
            if var.X > 0.5:
                print(f"{var.VarName}: {var.X}")

        # Window selection variables (z)
        print("\nWindow Selection (z):")
        for (track, a, w), var in self.z.items():
            if var.X > 0.5:
                print(f"{var.VarName}: {var.X}")

        # Start time variables (tau)
        print("\nStart Times (tau):")
        for (track, a), var in self.tau.items():
            if var.X > 0:  # Print if start time is assigned
                print(f"{var.VarName}: {var.X:.2f}")
#%%
                


#### SOLVE

from read_problem import Instance

instance = Instance()

#instance.load_data("build/dsn_schedule.json")
instance.load_data("build/dsn_tracks50_antenas3.json")
#instance.load_data("build/toy_problem.json")
graph = Graph(instance)

#%%

# graph.print_graph()
# graph.print_nodes_and_instance()

optimizer = OptimizationProblem(graph)
optimizer.set_objective()  # Use default objective (maximize duration)


status = optimizer.solve()

#%%
#optimizer.print_variables()
# 5. Process results
if status == GRB.OPTIMAL:
    optimizer.print_solution()
    optimizer.print_antenna_paths()
# %%