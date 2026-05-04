# AI-Based Pedestrian Safety System

This project implements a real-time pedestrian detection system using computer vision to enhance road safety by monitoring both driver and pedestrian environments.

# Features
- Real-time pedestrian detection using OpenCV
- Driver-side alert system for nearby pedestrians
- Crossing-side monitoring for safe road crossing
- Image preprocessing to improve detection accuracy
- Detection using video streams or images

# Technologies Used
- Python
- OpenCV
- Flask
- HTML, CSS

# Project Structure
- driver_app.py → Driver-side detection system
- crossing_app.py → Pedestrian crossing detection
- driver.html → Driver interface
- crossing.html → Crossing interface

# How It Works
- Captures real-time video input (camera/video file)
- Applies computer vision techniques to detect pedestrians
- Draws bounding boxes around detected pedestrians
- Triggers alerts based on movement and proximity

# Applications
- Driver assistance systems
- Smart traffic management
- Pedestrian safety monitoring
- Accident prevention systems
