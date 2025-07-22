
import json
import random
import uuid
from datetime import datetime, timedelta
import os


random.seed(42)  # For reproducibility

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
    start_week = datetime.strptime(f'{year} {week} 1', '%Y %W %w')
    start_week_minutes = 0
    data = []
    for _ in range(num_tracks):
        duration = int(random.uniform(60, 2*60))  # Duration between 1 to 3.5 hours in minutes
        setup_time = 60
        teardown_time = 15
        track_id = str(uuid.uuid4())
        time_window_start = start_week_minutes + generate_random_minutes(0, 6.5 * 24 * 60 * 1)  # Random start within the week
        time_window_end = time_window_start + generate_random_minutes(1 * 60, 1.5 * 60)  # Random end 1 to 2 hours later
        
        if random.random() < 0.9:
            resources = random.sample(["DSS-24", "DSS-26", "DSS-34", "DSS-36", "DSS-54"], k=2)
            # concatenate name with _
            resources = ["_".join(resources)]
        else:
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
    
    import os

    relative_path = os.path.dirname(os.path.abspath(__file__)) + "/build/toy_problem.json"
    
    with open(relative_path, "w") as file:
        json.dump(dsn_data, file, indent=2)
    
    print("DSN test data generated successfully.")

    #print(dsn_data)