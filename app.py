import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor  # <-- NEW: Import advanced AI
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- 1. AI MODEL & FORECASTING ---
# (This is now UPDATED to support two different models)

def train_forecasting_model(use_advanced_model=False):
    """
    Loads mock data and trains a model to predict
    energy load based on the hour of the day.
    """
    try:
        data = pd.read_csv('mock_energy_data.csv')
    except FileNotFoundError:
        st.error("Error: 'mock_energy_data.csv' not found!")
        st.stop()

    features = ['hour', 'day_of_week']
    target = 'base_load_kw'
    X = data[features]
    y = data[target]

    categorical_features = ['hour', 'day_of_week']
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
        remainder='passthrough'
    )

    # --- NEW: Choose which AI model to use ---
    if use_advanced_model:
        st.write("📈 Using Advanced AI (Random Forest)...")
        regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        st.write("📈 Using Simple AI (Linear Regression)...")
        regressor = LinearRegression()
    # --- End of new section ---

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)  # Use the selected regressor
    ])

    model.fit(X, y)
    return model

# --- 2. SMART HOME PARAMETERS ---
# (This is unchanged)

TOU_PRICES = {
    'off_peak': 0.10,
    'mid_peak': 0.18,
    'on_peak': 0.30
}

def get_price(hour):
    if 0 <= hour <= 6 or 22 <= hour <= 23:
        return TOU_PRICES['off_peak']
    elif 7 <= hour <= 16:
        return TOU_PRICES['mid_peak']
    elif 17 <= hour <= 21:
        return TOU_PRICES['on_peak']

# --- 3. OPTIMIZATION LOGIC ---
# (This is UPDATED to accept the appliance list and EV constraint)

def find_optimal_schedule(model, appliances_to_run, ev_finish_by=24):
    """
    Finds the cheapest time to run the provided list of appliances.
    """
    forecast_input = pd.DataFrame({
        'hour': range(24),
        'day_of_week': [3] * 24  # Assuming "today" is day 3 (Thursday)
    })
    
    base_load_forecast = model.predict(forecast_input)
    optimized_load_schedule = np.copy(base_load_forecast)
    final_schedule_plan = {}
    
    # --- UPDATED: Loop over the list passed from the UI ---
    for appliance in appliances_to_run:
        name = appliance['name']
        duration = appliance['duration_hours']
        
        # Skip appliances that have no duration
        if duration <= 0:
            continue
            
        kwh_per_hour = appliance['total_kwh'] / duration
        
        best_start_hour = -1
        lowest_cost = float('inf')

        # Try every possible start time
        for start_hour in range(24 - duration + 1):
            end_hour = start_hour + duration
            
            # --- NEW: Check EV 'finish by' constraint ---
            if name == 'Electric Vehicle (EV) Charger' and end_hour > ev_finish_by:
                continue  # This time slot finishes too late, skip it
            # --- End of new section ---

            total_cost_for_window = 0
            for hour in range(start_hour, end_hour):
                current_load = optimized_load_schedule[hour]
                price = get_price(hour)
                cost = (current_load + kwh_per_hour) * price
                total_cost_for_window += cost
            
            if total_cost_for_window < lowest_cost:
                lowest_cost = total_cost_for_window
                best_start_hour = start_hour
        
        # "Book" this appliance into the schedule
        if best_start_hour != -1:
            final_schedule_plan[name] = f"{best_start_hour}:00 - {best_start_hour + duration}:00"
            for hour in range(best_start_hour, best_start_hour + duration):
                optimized_load_schedule[hour] += kwh_per_hour
        else:
            # Could not find a valid time (e.g., constraint was impossible)
            final_schedule_plan[name] = "COULD NOT SCHEDULE"
            st.warning(f"Could not find a valid schedule for '{name}' with your constraints.")
            
    return final_schedule_plan, base_load_forecast, optimized_load_schedule

# --- 4. STREAMLIT WEB APP UI ---
# (This is UPDATED with sidebar controls)

def main():
    st.set_page_config(page_title="Smart Energy Manager", page_icon="💡", layout="wide")
    st.title("💡 AI-Based Smart Home Energy Manager")
    
    st.write("Welcome! Use the sidebar to set your preferences and find the cheapest energy schedule.")

    # --- NEW: Sidebar controls ---
    st.sidebar.header("Options")
    
    # AI Model Selection
    use_advanced_ai = st.sidebar.checkbox("🤖 Use Advanced 'Random Forest' AI")

    # Appliance Controls
    st.sidebar.header("Appliance Controls")
    
    # EV Slider
    ev_kwh = st.sidebar.slider("⚡ EV Charge Needed (kWh)", 
                               min_value=0.0, 
                               max_value=40.0, 
                               value=20.0,  # Default value
                               step=1.0)
    
    # EV Constraint
    ev_finish_by = st.sidebar.number_input("🔋 EV must be charged by (hour, 0-24)", 
                                           min_value=0, 
                                           max_value=24, 
                                           value=24) # Default 24 (no constraint)

    # Other Appliances
    run_dishwasher = st.sidebar.checkbox("🍽️ Run Dishwasher", value=True)
    run_washer = st.sidebar.checkbox("👚 Run Washing Machine", value=True)
    # --- End of new section ---


    # A button to run the optimization
    if st.button("Optimize My 24-Hour Schedule"):
        
        # --- NEW: Build appliance list dynamically ---
        appliances_to_run = []
        
        if ev_kwh > 0:
            # Assume a 5kW charger, calculate duration
            charge_rate_kw = 5.0 
            duration = int(np.ceil(ev_kwh / charge_rate_kw))
            appliances_to_run.append({'name': 'Electric Vehicle (EV) Charger', 'total_kwh': ev_kwh, 'duration_hours': duration})
        
        if run_dishwasher:
            appliances_to_run.append({'name': 'Dishwasher', 'total_kwh': 1.5, 'duration_hours': 2})
        
        if run_washer:
            appliances_to_run.append({'name': 'Washing Machine', 'total_kwh': 1.0, 'duration_hours': 1})
        # --- End of new section ---

        
        # 1. Run the AI model (with a loading spinner)
        with st.spinner("Training AI model and forecasting load..."):
            
            # --- UPDATED: Pass settings to functions ---
            ai_model = train_forecasting_model(use_advanced_model=use_advanced_ai)
            final_plan, base_load, optimized_load = find_optimal_schedule(ai_model, appliances_to_run, ev_finish_by)
            # --- End of update ---
        
        st.success("Optimization Complete!")
        st.balloons()

        # 2. Display the Appliance Schedule
        st.header("Optimal Appliance Schedule")
        
        if not appliances_to_run:
            st.info("No appliances were selected to run.")
        elif final_plan:
            plan_df = pd.DataFrame(list(final_plan.items()), columns=['Appliance', 'Scheduled Time'])
            st.dataframe(plan_df, use_container_width=True)
        else:
            st.write("No appliances could be scheduled.")

        # 3. Display the Cost Analysis
        st.header("Cost Analysis")
        
        base_cost = 0
        optimized_cost = 0
        for hour in range(24):
            price = get_price(hour)
            base_cost += base_load[hour] * price
            optimized_cost += optimized_load[hour] * price
        
        appliance_cost = optimized_cost - base_cost
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Base Load Cost", f"${base_cost:.2f}", help="Cost of your base load (lights, fridge, etc.) without new appliances.")
        col2.metric("Total Appliance Cost", f"${appliance_cost:.2f}", help="The *additional* cost from running the scheduled appliances.")
        col3.metric("Total Optimized Cost", f"${optimized_cost:.2f}", help="Your new total electricity bill for the day.")

        # 4. Display the Visual Chart
        st.header("Energy Load Forecast (kW)")
        
        chart_data = pd.DataFrame({
            'Hour': range(24),
            'Base Load': base_load,
            'Optimized Load (with Appliances)': optimized_load
        }).set_index('Hour')
        
        st.line_chart(chart_data)

# This is the new way to run our main function
if __name__ == "__main__":
    main()