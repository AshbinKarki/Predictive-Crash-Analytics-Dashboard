🚦 Crash Analytics Dashboard

An interactive dashboard built to analyze when crashes occur, which conditions increase risk, and how vehicle characteristics relate to crash severity, using real-world crash data.

📌 Overview

This project explores crash patterns across time, weather, road conditions, and vehicles.
The dashboard enables interactive exploration through filters and visualizations, helping uncover conditions that lead to higher injury severity.

Built with Python, Pandas, Plotly, and Dash.

❓ Key Questions

When are crashes most frequent during the day and week?

Which weather and lighting conditions are associated with higher crash severity?

How do road surface conditions affect outcomes?

Do vehicle age and type influence injury severity?

📊 Features

Hourly, weekly, and monthly crash trends

Crash patterns by lighting condition

Weather vs injury severity analysis

Weather vs road surface interaction

Vehicle year vs injury severity

Vehicle body type vs collision type

🛠️ Tech Stack

Python

Pandas, NumPy

Plotly, Dash

▶️ Run Locally
git clone https://github.com/your-username/crash-analytics-dashboard.git
cd crash-analytics-dashboard
pip install pandas numpy plotly dash
python app.py


Open: http://127.0.0.1:8050/

💡 Key Insights

Crashes peak during weekday evening rush hours.

Fog and snow/ice show a higher share of severe injuries despite fewer crashes.

Unlit dark conditions increase crash severity.

Older vehicles are associated with more severe outcomes.

📁 Files
├── app.py
├── Crash_Reporting_-_Drivers_Data.csv
├── README.md

🎯 Why This Project

Demonstrates end-to-end data analysis, cleaning, feature engineering, and interactive visualization skills using a large real-world dataset.
