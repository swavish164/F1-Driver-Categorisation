# F1 Driver Behaviour Analysis And Categorisation 

A Formula 1 telemetry analysis project that processes race session data to extract lap-level driving behaviour and race context. The system uses official F1 telemetry to classify laps (clean, attacking, defending), engineer driving features, and store the results for single-race or multi-race analysis.

---

## Overview

### Python data pipeline
Loads race sessions, laps, and telemetry data using the FastF1 library. Cleans and filters laps to remove invalid data, safety car, VSC, and yellow flag running before processing telemetry inputs.

### Telemetry feature extraction
Extracts driving behaviour features from raw telemetry, including throttle usage, braking patterns, cornering behaviour, and coasting. Corner data is identified and analysed to capture driver input through different phases of the lap.

### Race context detection
Classifies laps based on on-track context by detecting attacking and defending scenarios using driver proximity and positional data, separating traffic-affected laps from clean running.

### Data storage
Stores structured race, driver, lap, and feature data in a SQLite database. The database supports both per-lap analysis for individual races and averaged driver behaviour across multiple races.

### Analysis-ready output
Produces clean, structured datasets suitable for visualisation or unsupervised learning techniques such as PCA and clustering to compare driving styles across drivers and races.

---

## Core Features
- FastF1-based race session and telemetry ingestion  
- Lap filtering by race conditions (SC, VSC, yellow flags, deleted laps)  
- Telemetry-driven feature engineering (throttle, braking, cornering, coasting)  
- Attacking and defending lap classification using driver proximity  
- Relational SQLite database for structured storage  
- Single-race or multi-race averaged analysis support  


