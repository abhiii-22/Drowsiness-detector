# Drowsiness Detector Pro

A real-time drowsiness detection system that monitors eye activity through your webcam to determine attentiveness. Using computer vision and AI techniques, it detects eye closure, blinks, and alerts the user with visual cues and audible alarms when signs of drowsiness are detected.

## Features

- Detects eye open/closed state with high accuracy using MediaPipe Face Mesh  
- Counts blinks and tracks total drowsiness duration  
- Displays professional status indicators with colored text and boxes  
- Sounds a continuous beep alarm when eyes remain closed beyond a threshold  
- Logs drowsiness events with timestamps and duration in a CSV file  
- Handles no-person detection with clear messaging  

## How It Works

The system calculates the Eye Aspect Ratio (EAR) to quantify eye openness. If EAR falls below a threshold for more than one second, the user is marked inattentive, triggering the alarm and logging.

## Installation

Make sure you have Python 3 installed. Then install the required packages:

```bash
pip install -r requirements.txt
