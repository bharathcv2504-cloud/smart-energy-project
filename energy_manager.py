import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def train_forecasting_model():
    """
    Loads mock data and trains a simple model to predict
    energy load based on the hour of the day.
    """
    print("Training AI forecasting model...")
    
    # Load the mock data
    try:
        data = pd.read_csv('mock_energy_data.csv')
    except FileNotFoundError:
        print("Error: 'mock_energy_data.csv' not found!")
        print("Please create it first (see Phase 2, Step 7).")
        return None

    # Define our features (X) and target (y)
    # We want to predict 'base_load_kw' using 'hour' and 'day_of_week'
    features = ['hour', 'day_of_week']
    target = 'base_load_kw'

    X = data[features]
    y = data[target]

    # We must "one-hot encode" categorical features like 'hour'
    # This turns "hour 5" into a format the model understands
    categorical_features = ['hour', 'day_of_week']
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Create a "pipeline" that first processes the data, then runs the model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Train the model!
    model.fit(X, y)
    
    print("Model training complete.")
    return model

# Define our Time-of-Use (TOU) electricity pricing
# Prices are in $ per kWh
# 0-6 (Off-Peak), 7-16 (Mid-Peak), 17-21 (On-Peak), 22-23 (Off-Peak)
TOU_PRICES = {
    'off_peak': 0.10,
    'mid_peak': 0.18,
    'on_peak': 0.30
}

def get_price(hour):
    """Returns the electricity price for a given hour."""
    if 0 <= hour <= 6 or 22 <= hour <= 23:
        return TOU_PRICES['off_peak']
    elif 7 <= hour <= 16:
        return TOU_PRICES['mid_peak']
    elif 17 <= hour <= 21:
        return TOU_PRICES['on_peak']

# Define our deferrable (schedulable) appliances
# We list their name, total energy need (kWh), and how long they must run.
APPLIANCES = [
    {
        'name': 'Electric Vehicle (EV) Charger',
        'total_kwh': 20.0,  # Needs 20 kWh of charge
        'duration_hours': 4   # Must run for 4 consecutive hours
    },
    {
        'name': 'Dishwasher',
        'total_kwh': 1.5,
        'duration_hours': 2
    },
    {
        'name': 'Washing Machine',
        'total_kwh': 1.0,
        'duration_hours': 1
    }
]

def find_optimal_schedule(model):
    """
    The main logic. Finds the cheapest time to run appliances.
    """
    if model is None:
        print("Model is not trained. Exiting.")
        return

    print("\nFinding optimal energy schedule...")
    
    # 1. Get our 24-hour "base load" forecast from the AI model
    # We create a dummy DataFrame for the next 24 hours (today, day 3)
    forecast_input = pd.DataFrame({
        'hour': range(24),
        'day_of_week': [3] * 24  # Assuming "today" is day 3 (Thursday)
    })
    
    # The 'base_load_forecast' is now a 24-element array
    base_load_forecast = model.predict(forecast_input)
    
    # This will hold our final, total load (base + appliances)
    optimized_load_schedule = np.copy(base_load_forecast)
    
    # This will hold our human-readable schedule
    final_schedule_plan = {}

    # 2. Loop through each appliance to find its best slot
    for appliance in APPLIANCES:
        name = appliance['name']
        duration = appliance['duration_hours']
        # Energy per hour for this appliance
        kwh_per_hour = appliance['total_kwh'] / duration
        
        best_start_hour = -1
        lowest_cost = float('inf') # Start with an impossibly high cost

        # 3. Try every possible start time in the 24-hour window
        # (e.g., 0, 1, 2, ... up to 23 - duration)
        for start_hour in range(24 - duration + 1):
            end_hour = start_hour + duration
            total_cost_for_window = 0

            # 4. Calculate the *total* cost for this time window
            for hour in range(start_hour, end_hour):
                # Cost = (Base Load + Appliance Load) * Price at that hour
                current_load = optimized_load_schedule[hour] # Use the *updated* schedule
                price = get_price(hour)
                cost = (current_load + kwh_per_hour) * price
                total_cost_for_window += cost
            
            # 5. Check if this window is the cheapest one found so far
            if total_cost_for_window < lowest_cost:
                lowest_cost = total_cost_for_window
                best_start_hour = start_hour
        
        # 6. "Book" this appliance into the schedule
        if best_start_hour != -1:
            print(f"  -> Scheduling '{name}' to run from {best_start_hour}:00 to {best_start_hour + duration}:00.")
            final_schedule_plan[name] = f"{best_start_hour}:00 - {best_start_hour + duration}:00"
            
            # Add this appliance's load to our main schedule
            # so the *next* appliance knows the grid is busier
            for hour in range(best_start_hour, best_start_hour + duration):
                optimized_load_schedule[hour] += kwh_per_hour
        else:
            print(f"  -> Could not find a schedule for '{name}'.")
    
    return final_schedule_plan, base_load_forecast, optimized_load_schedule

def print_results(schedule, base_load, optimized_load):
    """
    Prints a nice summary of the results.
    """
    print("\n--- 📈 AI-Based Smart Home Energy Schedule ---")
    
    print("\n**Appliance Schedule:**")
    if not schedule:
        print("No appliances were scheduled.")
    for name, time_slot in schedule.items():
        print(f"  - {name}: {time_slot}")

    print("\n**Cost Analysis:**")
    base_cost = 0
    optimized_cost = 0
    
    for hour in range(24):
        price = get_price(hour)
        base_cost += base_load[hour] * price
        optimized_cost += optimized_load[hour] * price
        
    savings = base_cost - optimized_cost # This will be negative since we added load
    total_appliance_cost = optimized_cost - base_cost

    print(f"  - Base Load Cost (before appliances): ${base_cost:.2f}")
    print(f"  - Total Optimized Cost (with appliances): ${optimized_cost:.2f}")
    print(f"  - **Cost of running all appliances: ${total_appliance_cost:.2f}**")
    
    print("\n--- End of Report ---")

# This special line means "run this code only when the file is executed directly"
if __name__ == "__main__":
    # 1. Train model
    ai_model = train_forecasting_model()
    
    # 2. Find schedule
    final_plan, base_forecast, optimized_schedule = find_optimal_schedule(ai_model)
    
    # 3. Print the report
    if final_plan:
        print_results(final_plan, base_forecast, optimized_schedule)