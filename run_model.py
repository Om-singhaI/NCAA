import pandas as pd
import xgboost as xgb
import numpy as np

# ==========================================
# 1. CONFIGURATION (UPDATE THESE!)
# ==========================================
# Replace these strings with the ACTUAL column names from your CSV
features = [
    'NET_Rank', 'KPI_Rank', 'SOR_Rank',  # Examples: Replace with actual metric columns
    'Q1_Wins', 'Q2_Wins', 'SOS', 
    'Road_Wins', 'Conf_Record' 
]
target = 'Seed'  # The column you are predicting (e.g., 1-16, or 'Is_Selected')
season_col = 'Season' # The year column
team_id_col = 'TeamID'

# ==========================================
# 2. LOAD DATA
# ==========================================
print("Loading data...")
df_train = pd.read_csv('NCAA_Seed_Training_Set.csv')
df_test = pd.read_csv('NCAA_Seed_Test_Set.csv')
submission = pd.read_csv('submission_template.csv')

# Handle missing values (XGBoost handles NaNs, but clean data is better)
df_train = df_train.fillna(0)
df_test = df_test.fillna(0)

# ==========================================
# 3. PREPARE FOR RANKING
# ==========================================
# XGBoost Ranker needs data sorted by Group (Season)
df_train = df_train.sort_values(by=season_col)

X_train = df_train[features]
y_train = df_train[target]

# INVERSE LOGIC CHECK: 
# If the target is "Seed" (1 is best, 16 is worst), lower is better.
# If the target is "Selected" (1=Yes, 0=No), higher is better.
# For Ranking, we usually want Higher = Better relevance.
# If target is Seed, let's invert it so Rank 1 becomes a high score.
# y_train = 17 - y_train  # Uncomment this if your target is Seed (1-16)

# Create Groups
groups = df_train.groupby(season_col).size().to_list()

# ==========================================
# 4. TRAIN MODEL
# ==========================================
print(f"Training on {len(df_train)} teams across {len(groups)} seasons...")

model = xgb.XGBRanker(
    objective='rank:pairwise',
    learning_rate=0.05,
    n_estimators=500,
    max_depth=6,
    subsample=0.8,
    random_state=42
)

model.fit(X_train, y_train, group=groups, verbose=True)

# ==========================================
# 5. PREDICT & FORMAT SUBMISSION
# ==========================================
print("Predicting on Test Set...")

# Predict scores (Relevance scores, not seeds directly)
scores = model.predict(df_test[features])
df_test['Predicted_Score'] = scores

# Now we must map these scores to the Submission Template format.
# Usually, you need to output a Seed or a Probability.

# Example: Sorting the test set by Score to assign Seeds
# (This part depends heavily on if the Test Set is one year or multiple)
df_test = df_test.sort_values(by=['Predicted_Score'], ascending=False)

# Create the submission file
# Assuming the template wants ID and Prediction
submission_output = df_test[[team_id_col, 'Predicted_Score']]
submission_output.to_csv('my_submission.csv', index=False)

print("✅ Submission file 'my_submission.csv' generated!")