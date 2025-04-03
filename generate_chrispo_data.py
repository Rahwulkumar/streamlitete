import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Define possible values for each category
colleges = ["Alpha College", "Beta Institute", "Gamma University", "Delta Academy", "Epsilon College",
            "Zeta University", "Eta Institute", "Theta College", "Iota Academy", "Kappa University"]

states = ["California", "Texas", "New York", "Florida", "Illinois"]

sports = ["Basketball", "Soccer", "Tennis", "Volleyball", "Cricket", 
          "Swimming", "Badminton", "Table Tennis", "Athletics", "Chess"]

participation_statuses = ["Participated", "Absent", "Reserve"]

# Generate 300 rows of data
num_records = 300
data = []

for i in range(1, num_records + 1):
    participant_id = i
    college = random.choice(colleges)
    state = random.choice(states)
    sport = random.choice(sports)
    day = random.randint(1, 5)
    participation_status = random.choice(participation_statuses)
    
    # Score is only assigned if the participant actually participated
    score = round(random.uniform(0, 100), 2) if participation_status == "Participated" else np.nan
    
    # Generate feedback based on sport
    feedback_templates = {
        "Basketball": ["Great teamwork!", "Needs improvement in shooting.", "Excellent defensive play."],
        "Soccer": ["Outstanding dribbling skills.", "Poor passing accuracy.", "Great goalkeeping performance."],
        "Tennis": ["Strong serve!", "Needs better footwork.", "Great backhand shots."],
        "Volleyball": ["Powerful spikes!", "Better coordination needed.", "Good blocking skills."],
        "Cricket": ["Amazing batting!", "Could improve bowling accuracy.", "Excellent fielding."],
        "Swimming": ["Fast and smooth strokes.", "Needs better stamina.", "Great diving technique."],
        "Badminton": ["Quick reflexes!", "Lacks consistency in smashes.", "Strong net play."],
        "Table Tennis": ["Great spin control.", "Struggles with backhand shots.", "Excellent rally skills."],
        "Athletics": ["Fast sprint!", "Better endurance needed.", "Strong jumping ability."],
        "Chess": ["Strategic and calm.", "Needs to think ahead more.", "Good opening moves."]
    }
    
    feedback = random.choice(feedback_templates[sport])
    
    # Image filename
    image_filename = f"{sport.lower().replace(' ', '_')}_{day}_{participant_id}.jpg"
    
    # Generate a random registration date between '2024-01-01' and '2024-03-31'
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 3, 31)
    random_days = random.randint(0, (end_date - start_date).days)
    registration_date = start_date + timedelta(days=random_days)
    
    # Append to the data list
    data.append([
        participant_id, college, state, sport, day, participation_status,
        score, feedback, image_filename, registration_date.strftime("%Y-%m-%d")
    ])

# Create DataFrame
columns = ["ParticipantID", "College", "State", "Sport", "Day", "ParticipationStatus", 
           "Score", "Feedback", "ImageFilename", "RegistrationDate"]

df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv("chrispo_data.csv", index=False)

print("Dataset generated and saved as chrispo_data.csv!")
