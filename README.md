# 🤖 AI-Based Intelligent System for ROP Optimization (Algorithmic Core)

[![Python Version](https://img.shields.io/badge/python-3.13%2B-blue)](https://www.python.org/)
[![University](https://img.shields.io/badge/SUST-Sudan%20University-green)](https://sust.edu.sd/)


Welcome to the official repository of the **Intelligent Drilling Performance Optimizer**. This project represents a modern approach to Petroleum Engineering, integrating the classical **Galle-Woods** analytical model with **Advanced Pythonic Algorithms** for intelligent decision support in drilling operations.

## 🌟 Project Overview
This system is designed to optimize the **Rate of Penetration (dF/dT)** by analyzing real-time drilling parameters (WOB, RPM, Depth, and Time) specifically for the xxx well. The core of this research is to bridge the gap between classical Reservoir/Drilling physics and modern algorithmic optimization.

---

## 🧠 Core Intelligence: The Algorithmic Pillars
The intelligence of this system is built into its autonomous processing layers, developed as part of the **Gali Wadouz (Galle-Woods) Simulator**:

### 1. Stochastic Optimization (Monte Carlo Simulation)
A major innovation within this research is the implementation of **Monte Carlo Simulation** for **Uncertainty Modeling**. 
* The system generates thousands of probabilistic drilling scenarios to assess rock formation responses.
* It identifies the "Sweet Spot" for operational parameters under uncertain field conditions.

### 2. Engineering Data Firewall (`DataCleaner`)
To ensure high data integrity, the system implements a professional purging pipeline:
* **Strict Outlier Removal (3x IQR):** Uses a `3.0 * Interquartile Range` threshold to target catastrophic data entry errors while preserving natural geological variances.
* **Smart Imputation:** Missing values are filled using **Median-based filling** to maintain the statistical validity of the Tawakul-1 dataset.

### 3. System Protection Framework (SPF)
As detailed in the **Methodology (Chapter 3)**, the system is fortified with:
* **Math Protection:** Conditional validation and `try-except` blocks to prevent runtime arithmetic errors (Division by Zero, etc.).
* **UI Protection (Asynchronous Execution):** Utilizing Python’s `threading` library to offload intensive Monte Carlo computations to background processes, ensuring the **CustomTkinter** interface remains responsive.
* **Centralized Logging:** Continuous telemetry captured in `drilling_app.log` for diagnostic and academic auditing.

---

## 🛠️ Technical Workflow & Architecture
1. **Data Ingestion:** Importing field data via Excel using the `Pandas` engine.
2. **Algorithmic Cleaning:** Automatic 3x IQR filtering and Median imputation.
3. **Analytical Modeling:** Calculating `dF/dT` and `dF/dD` using derived Galle-Woods coefficients.
4. **Stochastic Iteration:** Running high-speed Monte Carlo loops.
5. **Heuristic Output:** Generating the "Top 3 Optimized Drilling Strategies."

---

## 📂 Repository Structure
* 📁 `src/`: Core Python source code (`main.py`) containing the UI and Optimization Logic.
* 📁 `data/`: Sample drilling datasets used for model calibration (Tawakul-1).
* 📄 `drilling_app.log`: Auto-generated telemetry file storing session histories.
* 📄 `requirements.txt`: (CustomTkinter, Pandas, Numpy, Matplotlib, Openpyxl).

---

## 🎓 Researcher & Academic Affiliation
* **Researcher:** Mohamed Elamin Ali
* **Institution:** Sudan University of Science and Technology (SUST)
* **Project Date:** 2026

---
*Developed as a functional integration of physical models and digital technologies to support Decision Making in the Oil & Gas industry.*
