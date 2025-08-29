#%%
import json
import random
import uuid
from datetime import datetime, timedelta

random.seed(42)  # For reproducibility

def datetime_to_minutes(dt):
    return int(dt.timestamp() // 60)

def generate_track_start(start_week_minutes, end_week_minutes):
    while True:
        candidate = random.randint(start_week_minutes, end_week_minutes - 1)
        dt = datetime.fromtimestamp(candidate * 60)
        hour = dt.hour
        if 8 <= hour < 22:
            if random.random() < 0.7:
                return candidate
        else:
            if random.random() < 0.3:
                return candidate

def generate_duration():
    r = random.random()
    if r < 0.8:
        return random.randint(60, 120)
    elif r < 0.95:
        return random.randint(121, 240)
    else:
        return random.randint(241, 480)

def generate_setup_teardown():
    return random.randint(10, 30), random.randint(5, 15)

def generate_view_periods(time_window_start, time_window_end, setup, teardown):
    periods = []
    for _ in range(3):  # Generate 3 periods per original format
        extra_before = random.randint(0, 60)
        extra_after = random.randint(0, 60)
        periods.append({
            "RISE": time_window_start - setup - extra_before,
            "SET": time_window_end + teardown + extra_after,
            "TRX ON": time_window_start,
            "TRX OFF": time_window_end
        })
    return periods

def generate_track_data(week, year, num_tracks, antennas):
    #start_week = datetime.strptime(f'{year}-W{week}-1', "%Y-W%W-%w")
    #start_week_minutes = datetime_to_minutes(start_week)
    #end_week_minutes = start_week_minutes + 7 * 24 * 60
    start_week_minutes =(7 * 24 * 60)*(week -1) 
    end_week_minutes = (7 * 24 * 60)*(week)
    tracks = []
  
    for _ in range(num_tracks):
        time_window_start = generate_track_start(start_week_minutes, end_week_minutes)
        duration = generate_duration()
        time_window_end = time_window_start + duration
        if time_window_end >= end_week_minutes:
            time_window_end = end_week_minutes - 1
            duration = time_window_end - time_window_start
            if duration <= 0: continue

        setup, teardown = generate_setup_teardown()
        num_ants = random.choices([1,2], weights=[0.99,0.01])[0]
        selected_ants = random.sample(antennas, k=num_ants) if num_ants <= len(antennas) else antennas
        
        # Modified resource handling
        if num_ants > 1:
            sync_group = "_".join(selected_ants)
            resources = [[sync_group]]
            resource_vp = {
                sync_group: generate_view_periods(time_window_start, time_window_end, setup, teardown)
            }
        else:
            resources = [[ant] for ant in selected_ants]
            resource_vp = {
                ant: generate_view_periods(time_window_start, time_window_end, setup, teardown)
                for ant in selected_ants
            }

        tracks.append({
            "subject": 521,
            "user": "521_0",
            "week": week,
            "year": year,
            "duration": duration,
            "duration_min": duration,
            "track_id": str(uuid.uuid4()),
            "setup_time": setup,
            "teardown_time": teardown,
            "time_window_start": time_window_start,
            "time_window_end": time_window_end,
            "resources": resources,
            "resource_vp_dict": resource_vp
        })
    return tracks

def generate_dsn_data(weeks, year, tracks_per_week, antennas):
    return {
        f"W{week}_{year}": generate_track_data(week, year, tracks_per_week, antennas)
        for week in weeks
    }

if __name__ == "__main__":
    # # Configuration
    WEEKS = list(range(1, 3))
    YEAR = 2024
    TRACKS_PER_WEEK = 50
    ANTENNAS = [f"DSS-{i:02d}" for i in range(1, 10)]  # 6 antennas
    print(f"Numero de antenas é {ANTENNAS}")

    dsn_data = generate_dsn_data(WEEKS, YEAR, TRACKS_PER_WEEK, ANTENNAS)

    import os

    relative_path = os.path.dirname(os.path.abspath(__file__)) + "/build/dsn_schedule.json"
    
    with open(relative_path, "w") as file:
        json.dump(dsn_data, file, indent=2)

    print("Realistic DSN schedule generated successfully.")
    #print(dsn_data)
# %%
