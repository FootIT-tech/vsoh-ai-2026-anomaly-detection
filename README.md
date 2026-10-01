# vsoh-ai-2026-anomaly-detection
Machine Learning solution using Isolation Forest for Unsupervised Anomaly Detection in spacecraft telemetry data. Developed for the All-Russian AI Olympiad 2026.

# All-Russian AI Olympiad 2026 🚀
## Task F: "Silence AZ-26" — Unsupervised Anomaly Detection in Space Telemetry

This repository contains a high-performing Machine Learning solution for **Task F ("Silence AZ-26")** of the Main Stage at the All-Russian Olympiad in Artificial Intelligence 2026. 

The goal of this challenge is to detect anomalous behavior and failures in spacecraft equipment using multi-channel sensor telemetry time-series data.

### 🛠️ Tech Stack & Architecture
* **Language:** Python 3
* **Data Processing:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (`IsolationForest`, `RobustScaler`)

### 🧠 Feature Engineering & Core Approach
Instead of relying on raw sensor values, the solution engineers robust statistical features to capture temporal and structural deviations:
1. **Session-wise Median Deviation:** Calculates the absolute distance of each sensor reading from its median within a specific communication session to capture sudden spikes.
2. **Rolling Volatility (`rolling std`):** Computes a moving standard deviation (window size = 5) to capture destabilization trends and irregular fluctuations over time.
3. **Robust Scaling:** Uses `RobustScaler` to normalize features, ensuring the model remains resilient against extreme outliers in the training data.
4. **Isolation Forest:** Deploys an ensemble of Isolation Trees to flag anomalies in a high-dimensional feature space without relying on explicit target labels (Unsupervised/Semi-supervised approach).

### 📈 Results
The algorithm achieved a solid score of **158 points** on the official VK All Cups evaluation platform, securing a guaranteed spot in the next stage of the competition.
