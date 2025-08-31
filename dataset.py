
import pandas as pd
import numpy as np

np.random.seed(42)
n = 200

attendance = np.random.randint(50, 100, n)
study_hours = np.random.randint(1, 15, n)
past_score = np.random.randint(40, 100, n)

final_score = (0.3 * attendance + 0.4 * study_hours*5 + 0.3 * past_score +
               np.random.normal(0, 5, n))

df = pd.DataFrame({
    "Attendance": attendance,
    "StudyHours": study_hours,
    "PastScore": past_score,
    "FinalScore": final_score
})

df.to_csv("student_performance.csv", index=False)
print(" Dataset created: student_performance.csv")
