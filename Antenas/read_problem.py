
import json

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