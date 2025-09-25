import pandas as pd
import numpy as np
import os


if not os.path.exists('adult.data.csv'):
    print("Downloading dataset...")
    import urllib.request
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    urllib.request.urlretrieve(url, 'adult.data.csv')
    print("Dataset downloaded!")


column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary'
]

df = pd.read_csv('adult.data.csv', header=None, names=column_names, skipinitialspace=True)


for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip()

# 1. How many people of each race?
race_count = df['race'].value_counts()

# 2. Average age of men
average_age_men = round(df[df['sex'] == 'Male']['age'].mean(), 1)

# 3. Percentage with Bachelor's degree
total_people = len(df)
bachelors_count = len(df[df['education'] == 'Bachelors'])
percentage_bachelors = round((bachelors_count / total_people) * 100, 1)

# 4. Percentage with advanced education making >50K
advanced_education = ['Bachelors', 'Masters', 'Doctorate']
advanced_edu_df = df[df['education'].isin(advanced_education)]
advanced_edu_rich = len(advanced_edu_df[advanced_edu_df['salary'] == '>50K'])
higher_education_rich = round((advanced_edu_rich / len(advanced_edu_df)) * 100, 1)

# 5. Percentage without advanced education making >50K
no_advanced_edu_df = df[~df['education'].isin(advanced_education)]
no_advanced_edu_rich = len(no_advanced_edu_df[no_advanced_edu_df['salary'] == '>50K'])
lower_education_rich = round((no_advanced_edu_rich / len(no_advanced_edu_df)) * 100, 1)

# 6. Minimum hours per week
min_work_hours = df['hours-per-week'].min()

# 7. Percentage of min workers with >50K salary
min_workers = df[df['hours-per-week'] == min_work_hours]
min_workers_rich = len(min_workers[min_workers['salary'] == '>50K'])
rich_percentage = round((min_workers_rich / len(min_workers)) * 100, 1)

# 8. Country with highest percentage of >50K earners
country_stats = df.groupby('native-country')['salary'].apply(
    lambda x: (x == '>50K').mean() * 100
).round(1)
highest_earning_country = country_stats.idxmax()
highest_earning_country_percentage = country_stats.max()

# 9. Most popular occupation for >50K earners in India
india_high_earners = df[(df['native-country'] == 'India') & (df['salary'] == '>50K')]
top_IN_occupation = india_high_earners['occupation'].value_counts().idxmax()

# Print results
print("Demographic Data Analysis Results:")
print("=" * 50)
print(f"1. Race count:\n{race_count}")
print(f"\n2. Average age of men: {average_age_men}")
print(f"3. Percentage with Bachelor's degree: {percentage_bachelors}%")
print(f"4. Percentage with advanced education earning >50K: {higher_education_rich}%")
print(f"5. Percentage without advanced education earning >50K: {lower_education_rich}%")
print(f"6. Minimum work hours per week: {min_work_hours}")
print(f"7. Percentage of min workers earning >50K: {rich_percentage}%")
print(f"8. Country with highest percentage of >50K earners: {highest_earning_country} ({highest_earning_country_percentage}%)")
print(f"9. Top occupation in India for >50K earners: {top_IN_occupation}")