import pandas as pd

# Load the datasets
train = pd.read_csv('NCAA_Seed_Training_Set2.0.csv')
test = pd.read_csv('NCAA_Seed_Test_Set2.0.csv')
dictionary = pd.read_excel('FFAC Data Dictionary.xlsx')

print("--- TRAINING DATA COLUMNS ---")
print(train.columns.tolist())

print("\n--- DATA DICTIONARY PREVIEW ---")
print(dictionary.head(10))

print("\n--- SUBMISSION FORMAT ---")
sub_template = pd.read_csv('submission_template2.0.csv')
print(sub_template.head())