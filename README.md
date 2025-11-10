# 💡 AI-Based Smart Home Energy Manager

This is a Python-based web application that uses machine learning to forecast a home's base energy load and then optimizes the schedule for high-energy appliances to run at the cheapest possible times, based on Time-of-Use (TOU) electricity pricing.

The entire app is built with **Streamlit**, **Pandas**, and **Scikit-learn**.

## 📸 Preview
![Screenshot 2025-11-10 110956](https://github.com/user-attachments/assets/9230bfd9-398e-4dbe-9c81-246013964c35)

![Screenshot 2025-11-10 111116](https://github.com/user-attachments/assets/9e64c4d8-07c3-401e-b684-80cda499656a)




---

## ✨ Features

* **AI-Powered Forecasting:** Uses a machine learning model (Linear Regression or an advanced Random Forest) to predict your 24-hour base energy load.
* **Cost Optimization:** Automatically schedules appliances (EV, Dishwasher, etc.) to run during the cheapest "Off-Peak" hours.
* **Interactive Dashboard:** A simple and clean web interface built with Streamlit.
* **User Controls (in Sidebar):**
    * **EV Charge Slider:** Set exactly how many kWh your Electric Vehicle needs.
    * **Appliance Toggles:** Enable or disable the Dishwasher and Washing Machine from the schedule.
    * **AI Model Selection:** Switch between a "Simple" or "Advanced" AI model.
    * **EV Charge Constraint:** Set a "Must be charged by" hour to ensure your EV is ready for your morning commute.
* **Dynamic Cost Analysis:** Instantly see your base cost, appliance cost, and total optimized cost for the day.
* **Visual Load Chart:** See a 24-hour chart of your base load vs. your new *optimized* load with appliances.

---


