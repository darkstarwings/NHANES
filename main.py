import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import os
import subprocess

# Install openpyxl
subprocess.call("pip install openpyxl", shell=True)

# Function to download XPT file
def download_xpt(url, filename):
    if os.path.exists(filename):
        print(f"File {filename} already exists. Skipping download.")
        return
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {filename}")
    else:
        print(f"Failed to download {filename}. Status code:", response.status_code)

# Function to fetch and download specific XPT files from a given URL
def fetch_and_download_specific_xpt(base_url, component, specific_xpt_files):
    # Construct the URL
    url = f"{base_url}&Component={component}"

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.content, "html.parser")

        # Find all <a> tags with href containing ".XPT"
        xpt_links = soup.find_all("a", href=lambda href: href and href.endswith('.XPT'))

        if xpt_links:
            for xpt_link in xpt_links:
                # Extract the filename from the URL
                filename = xpt_link["href"].split("/")[-1]

                # Check if the filename contains any of the specific XPT files
                for specific_file in specific_xpt_files:
                    if specific_file in filename:
                        # Construct the full URL of the XPT file
                        full_xpt_url = "https://wwwn.cdc.gov" + xpt_link["href"]

                        # Download the XPT file
                        download_xpt(full_xpt_url, filename)
                        break
        else:
            print("XPT file links not found on the webpage.")
    else:
        print("Failed to fetch the webpage. Status code:", response.status_code)

# Base URL for NHANES data
base_url = "https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?CycleBeginYear=2013"

# List of components along with specific XPT files to download
components = {
    "Demographics": ["DEMO_H"],
    "Examination": ["BMX_H", "BPX_H"],
    "Laboratory": ["TRIGLY_H", "GLU_H", "HDL_H"],
    "Questionnaire": ["OCQ_H", "BPQ_H", "DIQ_H", "INQ_H", "MCQ_H"]
}

# Loop through each component and fetch specific XPT files
for component, specific_xpt_files in components.items():
    fetch_and_download_specific_xpt(base_url, component, specific_xpt_files)

# Load the NHANES datasets
DEMO_H = pd.read_sas("DEMO_H.XPT")
OCQ_H = pd.read_sas("OCQ_H.XPT")
BPX_H = pd.read_sas("BPX_H.XPT")
INQ_H = pd.read_sas("INQ_H.XPT")
BMX_H = pd.read_sas("BMX_H.XPT")
TRIGLY_H = pd.read_sas("TRIGLY_H.XPT")
HDL_H = pd.read_sas("HDL_H.XPT")
BPX_H = pd.read_sas("BPX_H.XPT")
GLU_H = pd.read_sas("GLU_H.XPT")
BPQ_H = pd.read_sas("BPQ_H.XPT")
MCQ_H = pd.read_sas("MCQ_H.XPT")
DIQ_H = pd.read_sas("DIQ_H.XPT")

# Merge the datasets on SEQN (unique identifier for each participant)
merged_data = pd.merge(OCQ_H, DEMO_H, on="SEQN", how="left")
merged_data = pd.merge(merged_data, BPX_H, on="SEQN", how="left")
merged_data = pd.merge(merged_data, INQ_H, on="SEQN", how="left")
merged_data = pd.merge(merged_data, BMX_H, on="SEQN", how="left")
merged_data = pd.merge(merged_data, TRIGLY_H, on="SEQN", how="left")
merged_data = pd.merge(merged_data, HDL_H, on="SEQN", how="left")
merged_data = pd.merge(merged_data, GLU_H, on="SEQN", how="left")
merged_data = pd.merge(merged_data, BPQ_H, on="SEQN", how="left")
merged_data = pd.merge(merged_data, MCQ_H, on="SEQN", how="left")
merged_data = pd.merge(merged_data, DIQ_H, on="SEQN", how="left")

# Count of participants in merged dataset
key_count = merged_data['SEQN'].nunique()
print("\nParticipant Count: ", key_count, "\n")

# Function to create age groups
def create_age_groups(age):
    if pd.isnull(age):
        return "Missing"
    elif age < 80:
        return f"{int((age // 10) * 10)}-{int((age // 10) * 10 + 9)}"
    else:
        return "≥80"

# Create age groups
merged_data["AgeGroup"] = merged_data["RIDAGEYR"].apply(create_age_groups)

# Coded values and missing value for gender according to the codebook
gender_coded_values = {1: "1_Male", 2: "2_Female", "": "99_Missing"}

# Coded values and missing value for race according to the codebook
race_coded_values = {1: "1_Mexican American", 
                     2: "2_Other Hispanic", 
                     3: "3_Non-Hispanic White", 
                     4: "4_Non-Hispanic Black", 
                     6: "6_Non-Hispanic Asian", 
                     7: "7_Other Race - Including Multi-Racial",
                     "": "Missing"}

# Coded values and missing value for INDFMMPC according to the codebook
indfmmpc_coded_values = {1: "1_Monthly poverty level index <= 1.30", 
                         2: "2_1.30 < Monthly poverty level index <= 1.85", 
                         3: "3_Monthly poverty level index > 1.85", 
                         7: "4_Refused", 
                         9: "5_Don't Know",
                         "": "Missing"}

# Criteria for metabolic syndrome
metabolic_syndrome_criteria = {
    "Abdominal obesity": {"Male": 102, "Female": 88},
    "High triglyceride level": 150,
    "Low HDL cholesterol": {"Male": 40, "Female": 50},
    "High blood pressure": {"SBP": 130, "DBP": 85},
    "High fasting glucose": 100
}

# Function to calculate metabolic syndrome status
def metabolic_syndrome_status(row):
    criteria_met = 0

    # Abdominal obesity
    if row["RIAGENDR"] == 1:
        if pd.notnull(row["BMXWAIST"]) and row["BMXWAIST"] >= metabolic_syndrome_criteria["Abdominal obesity"]["Male"]:
            criteria_met += 1
    elif row["RIAGENDR"] == 2:
        if pd.notnull(row["BMXWAIST"]) and row["BMXWAIST"] >= metabolic_syndrome_criteria["Abdominal obesity"]["Female"]:
            criteria_met += 1

    # High triglyceride level
    if pd.notnull(row["LBXTR"]) and row["LBXTR"] >= metabolic_syndrome_criteria["High triglyceride level"]:
        criteria_met += 1

    # Low HDL cholesterol
    if pd.notnull(row["LBDHDD"]) and pd.notnull(metabolic_syndrome_criteria["Low HDL cholesterol"].get(row["RIAGENDR"])):
        if row["LBDHDD"] < metabolic_syndrome_criteria["Low HDL cholesterol"][row["RIAGENDR"]]:
            criteria_met += 1

    # High blood pressure
    if pd.notnull(row["BPXSY1"]) and pd.notnull(row["BPXDI1"]) and pd.notnull(metabolic_syndrome_criteria["High blood pressure"]["SBP"]) and pd.notnull(metabolic_syndrome_criteria["High blood pressure"]["DBP"]):
        if row["BPXSY1"] >= metabolic_syndrome_criteria["High blood pressure"]["SBP"] or row["BPXDI1"] >= metabolic_syndrome_criteria["High blood pressure"]["DBP"]:
            criteria_met += 1

    # High fasting glucose
    if pd.notnull(row["LBXGLU"]):
        if row["LBXGLU"] >= metabolic_syndrome_criteria["High fasting glucose"]:
            criteria_met += 1

    # Categorize metabolic syndrome status
    if criteria_met == 0:
        return "1_No MetS"
    elif criteria_met <= 2:
        return "2_At risk of MetS"
    else:
        return "3_MetS"

# Apply the metabolic syndrome status calculation
merged_data["MetabolicSyndromeStatus"] = merged_data.apply(metabolic_syndrome_status, axis=1)

# Function to calculate count of industry subgroups
def calculate_count(data, column, missing_value):
    # Copy the original dataframe
    count_df = data.copy()

    # Replace missing values
    count_df[column] = count_df[column].fillna(missing_value)

    # Convert the column to string
    count_df[column] = count_df[column].astype(str)

    # Calculate count of each industry subgroup
    count_series = count_df[column].value_counts()

    # Convert series to dataframe
    count_df = count_series.reset_index()

    # Convert the industry subgroup column to numeric for sorting
    count_df[column] = pd.to_numeric(count_df[column], errors='coerce')

    #count_df[column].replace(np.nan, "Missing")

    # Sort the dataframe by industry subgroup
    count_df = count_df.sort_values(by=column)

    # Rename the columns
    count_df.columns = [column, "Count"]

    # Convert NaN to "Missing"
    count_df[column] = count_df[column].fillna("Missing")

    # Create a new dataframe with two rows
    count_data = pd.DataFrame({
        "OCD21": count_df[column].tolist() + ["Total"],
        "Count": count_df["Count"].tolist() + [len(data)]
    })

    # Transpose the dataframe
    count_data = count_data.T

    # Set the first row as column headers
    count_data.columns = count_data.iloc[0]

    # Drop the first row
    count_data = count_data[1:]

    #count_data.replace(np.nan, "Missing")

    return count_data


# Function to calculate distribution by subgroup
def calculate_distribution_by_subgroup(data, column, coded_values, missing_value, subgroup_column):
    # Copy the original dataframe
    distribution_df = data.copy()

    # Replace coded values with actual categories
    if coded_values:
        distribution_df[column] = distribution_df[column].map(coded_values)

    # Replace missing values
    distribution_df[column] = distribution_df[column].fillna(missing_value)
    distribution_df[subgroup_column] = distribution_df[subgroup_column].fillna(missing_value)

    # Group by subgroup and the column, calculate the count, and unstack the result
    distribution_df = distribution_df.groupby([subgroup_column, column]).size().unstack(fill_value=0)

    # Convert the count to a percentage
    #distribution_df = distribution_df.apply(lambda x: x / x.sum(), axis=1)

    # Add a row for total count
    distribution_df.loc["Total"] = distribution_df.sum()

    # Transpose the dataframe
    distribution_df = distribution_df.transpose()

    # Calculate percentages and combine with count
    for col in distribution_df.columns:
        total_count = distribution_df[col].sum()
        percentages = (distribution_df[col] / total_count * 100).round(1)
        distribution_df[col] = distribution_df[col].astype(str) + ' (' + percentages.astype(str) + '%)'

    return distribution_df


# Calculate the count of each industry subgroup
industry_count = calculate_count(merged_data, "OCD231", "Missing")

# Calculate the distribution of gender (RIAGENDR)
gender_distribution = calculate_distribution_by_subgroup(merged_data, "RIAGENDR", gender_coded_values, "Missing", "OCD231")

# Calculate the distribution of age (AgeGroup)
age_distribution = calculate_distribution_by_subgroup(merged_data, "AgeGroup", None, "Missing", "OCD231")

# Calculate the distribution of race (RIDRETH3)
race_distribution = calculate_distribution_by_subgroup(merged_data, "RIDRETH3", race_coded_values, "Missing", "OCD231")

# Calculate the distribution of family monthly poverty to income level (INDFMMPC)
poverty_distribution = calculate_distribution_by_subgroup(merged_data, "INDFMMPC", indfmmpc_coded_values, "Missing", "OCD231")

# Apply the metabolic syndrome status calculation
merged_data["MetabolicSyndromeStatus"] = merged_data.apply(metabolic_syndrome_status, axis=1)

# Calculate the distribution of metabolic syndrome status
metabolic_syndrome_distribution = calculate_distribution_by_subgroup(merged_data, "MetabolicSyndromeStatus", None, "Missing", "OCD231")

# Display the distributions as tables
print("Distribution of Industry (OCD231):\n", industry_count)
print("\nDistribution of Gender (RIAGENDR):\n", gender_distribution)
print("\nDistribution of Age (AgeGroup):\n", age_distribution)
print("\nDistribution of Race (RIDRETH3):\n", race_distribution)
print("\nDistribution of Family Monthly Poverty to Income Level (INDFMMPC):\n", poverty_distribution)
print("\nDistribution of Metabolic Syndrome Status:\n", metabolic_syndrome_distribution)


# Filter people with high triglyceride level or low HDL cholesterol
high_triglyceride_or_low_hdl = merged_data[(merged_data['LBXTR'] >= metabolic_syndrome_criteria["High triglyceride level"]) | ((merged_data['RIAGENDR'] == 1) & (merged_data['LBDHDD'] < metabolic_syndrome_criteria["Low HDL cholesterol"]["Male"])) | ((merged_data['RIAGENDR'] == 2) & (merged_data['LBDHDD'] < metabolic_syndrome_criteria["Low HDL cholesterol"]["Female"]))]

# Filter people with high blood pressure based on both SBP and DBP criteria
high_bp = merged_data[(merged_data['BPXSY1'] >= metabolic_syndrome_criteria["High blood pressure"]["SBP"]) | (merged_data['BPXDI1'] >= metabolic_syndrome_criteria["High blood pressure"]["DBP"])]

# Filter people with high fasting glucose
high_glucose = merged_data[merged_data['LBXGLU'] >= metabolic_syndrome_criteria["High fasting glucose"]]

# Filter people with abdominal obesity based on gender-specific criteria
abdominal_obesity = merged_data[((merged_data['RIAGENDR'] == 1) & (merged_data['BMXWAIST'] >= metabolic_syndrome_criteria["Abdominal obesity"]["Male"])) | ((merged_data['RIAGENDR'] == 2) & (merged_data['BMXWAIST'] >= metabolic_syndrome_criteria["Abdominal obesity"]["Female"]))]

# Define codebook for BPQ080, BPQ090D, BPQ100D, BPQ020, BPQ040A, BPQ050A, DIQ010, DIQ050, DIQ070, MCQ080, MCQ365a
qnr_coded_values = {
    1: "1_Yes",
    2: "2_No",
    3: "3_Borderline",
    7: "7_Refused",
    9: "9_Don't Know",
    "": "Missing"
}

def count_and_percentage(df, column_name):
    # Calculate the count of each category
    count_df = df[column_name].map(qnr_coded_values).value_counts().sort_index().reset_index()
    count_df.columns = [column_name, 'Count']

    # Calculate the total count
    total_count = len(df)

    # Calculate the percentage of each category
    count_df['Percentage'] = (count_df['Count'] / total_count * 100).round(1).astype(str) + '%'

    # Combine count and percentage into a new column
    count_df['Combined'] = count_df['Count'].astype(str) + ' (' + count_df['Percentage'] + ')'

    # Add the 'Total' row
    count_df = pd.DataFrame({column_name: ['Total'] + count_df[column_name].astype(str).tolist(), 
                             'Count': [total_count] + count_df['Combined'].tolist()})

    return count_df


# 1. Told to have high cholesterol level (BPQ080)
bpq080_count = count_and_percentage(high_triglyceride_or_low_hdl, 'BPQ080')

# 2. Told to take prescription for cholesterol (BPQ090D)
bpq090d_condition = high_triglyceride_or_low_hdl[high_triglyceride_or_low_hdl['BPQ080'] == 1]
bpq090d_count = count_and_percentage(bpq090d_condition, 'BPQ090D')

# 3. Now taking prescribed medicine (BPQ100D)
bpq100d_condition = high_triglyceride_or_low_hdl[(high_triglyceride_or_low_hdl['BPQ080'] == 1) & (high_triglyceride_or_low_hdl['BPQ090D'] == 1)]
bpq100d_count = count_and_percentage(bpq100d_condition, 'BPQ100D')

print("[Participants with high cholesterol level]")
print("\n1. Told to have high cholesterol level (BPQ080)")
print(bpq080_count)
print("\n2. Told to take prescription for cholesterol (BPQ090D)")
print(bpq090d_count)
print("\n3. Now taking prescribed medicine (BPQ100D)")
print(bpq100d_count)


# 1. Told to have high blood pressure (BPQ020)
bpq020_count = count_and_percentage(high_bp, 'BPQ020')

# 2. Told to take prescription for hypertension (BPQ040A)
bpq040a_condition = high_bp[high_bp['BPQ020'] == 1]
bpq040a_count = count_and_percentage(bpq040a_condition, 'BPQ040A')

# 3. Now taking prescribed medicine (BPQ050A)
bpq050a_condition = high_bp[(high_bp['BPQ020'] == 1) & (high_bp['BPQ040A'] == 1)]
bpq050a_count = count_and_percentage(bpq050a_condition, 'BPQ050A')

print("\n[Participants with high blood pressure]")
print("\n1. Told to have high blood pressure (BPQ020)")
print(bpq020_count)
print("\n2. Told to take prescription for hypertension (BPQ040A)")
print(bpq040a_count)
print("\n3. Now taking prescribed medicine (BPQ050A)")
print(bpq050a_count)


# 1. Told to have diabetes (DIQ010)
diq010_count = count_and_percentage(high_glucose, 'DIQ010')

# 2. Now taking insulin (DIQ050)
diq050_condition = high_glucose[high_glucose['DIQ010'] == 1]
diq050_count = count_and_percentage(diq050_condition, 'DIQ050')

# 3. Now taking diabetic pills (DIQ070)
diq070_condition = high_glucose[high_glucose['DIQ010'] == 1]
diq070_count = count_and_percentage(diq070_condition, 'DIQ070')

# 4. Now taking insulin/diabetic pills (DIQ050 + DIQ070)
diabetes_tx_condition = high_glucose[(high_glucose['DIQ010'] == 1) & ((high_glucose['DIQ050'] == 1) | (high_glucose['DIQ070'] == 1))]
diabetes_tx_condition['Diabetes Treatment'] = np.where((diabetes_tx_condition['DIQ050'] == 1) | (diabetes_tx_condition['DIQ070'] == 1), 'Yes', 'No')
diabetes_tx_count = count_and_percentage(diabetes_tx_condition, 'Diabetes Treatment')

print("\n[Participants with high fasting glucose]")
print("\n1. Told to have diabetes (DIQ010)")
print(diq010_count)
print("\n2. Now taking insulin (DIQ050)")
print(diq050_count)
print("\n3. Now taking diabetic pills (DIQ070)")
print(diq070_count)
print("\n4. Now taking insulin/diabetic pills (DIQ050 + DIQ070)")
print(diabetes_tx_count)


# 1. Told to be overweight (MCQ080)
mcq080_count = count_and_percentage(abdominal_obesity, 'MCQ080')

# 2. Told to lose weight (MCQ365a)
mcq365a_condition = abdominal_obesity[abdominal_obesity['MCQ080'] == 1]
mcq365a_count = count_and_percentage(mcq365a_condition, 'MCQ365A')

# 3. Now controlling weight (MCQ370a)
mcq370a_condition = abdominal_obesity[(abdominal_obesity['MCQ080'] == 1) & (abdominal_obesity['MCQ365A'] == 1)]
mcq370a_count = count_and_percentage(mcq370a_condition, 'MCQ370A')

# 4. Told to exercise (MCQ365b)
mcq365b_condition = abdominal_obesity[abdominal_obesity['MCQ080'] == 1]
mcq365b_count = count_and_percentage(mcq365b_condition, 'MCQ365B')

# 5. Now increasing exercise (MCQ370b)
mcq370b_condition = abdominal_obesity[(abdominal_obesity['MCQ080'] == 1) & (abdominal_obesity['MCQ365B'] == 1)]
mcq370b_count = count_and_percentage(mcq370b_condition, 'MCQ370B')

# 6. Told to reduce salt in diet (MCQ365c)
mcq365c_condition = abdominal_obesity[abdominal_obesity['MCQ080'] == 1]
mcq365c_count = count_and_percentage(mcq365c_condition, 'MCQ365C')

# 7. Now reducing salt in diet (MCQ370c)
mcq370c_condition = abdominal_obesity[(abdominal_obesity['MCQ080'] == 1) & (abdominal_obesity['MCQ365C'] == 1)]
mcq370c_count = count_and_percentage(mcq370c_condition, 'MCQ370C')

# 8. Told to reduce fat/calories (MCQ365d)
mcq365d_condition = abdominal_obesity[abdominal_obesity['MCQ080'] == 1]
mcq365d_count = count_and_percentage(mcq365d_condition, 'MCQ365D')

# 9. Now reducing fat in diet (MCQ370d)
mcq370d_condition = abdominal_obesity[(abdominal_obesity['MCQ080'] == 1) & (abdominal_obesity['MCQ365D'] == 1)]
mcq370d_count = count_and_percentage(mcq370d_condition, 'MCQ370D')

print("\n[Participants with abdoiminal obesity]")
print("\n1. Told to be overweight (MCQ080)")
print(mcq080_count)
print("\n2. Told to lose weight (MCQ365A)")
print(mcq365a_count)
print("\n3. Now controlling weight (MCQ370A)")
print(mcq370a_count)
print("\n4. Told to exercise (MCQ365B)")
print(mcq365b_count)
print("\n5. Now increasing exercise (MCQ370B)")
print(mcq370b_count)
print("\n6. Told to reduce salt in diet (MCQ365C)")
print(mcq365c_count)
print("\n7. Now reducing salt in diet (MCQ370C)")
print(mcq370c_count)
print("\n8. Told to reduce fat/calories (MCQ365D)")
print(mcq365d_count)
print("\n9. Now reducing fat in diet (MCQ370D)")
print(mcq370d_count)


# Write each distribution to a separate worksheet in an Excel file
output_file = 'distribution_tables.xlsx'

# Delete the existing file if it exists
if os.path.exists(output_file):
    os.remove(output_file)
    print(f"\nDeleted existing file: {output_file}")

# Write each distribution to a separate worksheet in an Excel file
with pd.ExcelWriter(output_file, mode='w') as writer:
    industry_count.to_excel(writer, sheet_name='Industry Distribution', index=True, header=True)
    gender_distribution.to_excel(writer, sheet_name='Gender Distribution', index=True, header=True)
    age_distribution.to_excel(writer, sheet_name='Age Distribution', index=True, header=True)
    race_distribution.to_excel(writer, sheet_name='Race Distribution', index=True, header=True)
    poverty_distribution.to_excel(writer, sheet_name='Poverty Distribution', index=True, header=True)
    metabolic_syndrome_distribution.to_excel(writer, sheet_name='MetS Distribution', index=True, header=True)
    bpq080_count.to_excel(writer, sheet_name='Told to have high Chol level', index=True, header=True)
    bpq090d_count.to_excel(writer, sheet_name='Told to take Rx for Chol', index=True, header=True)
    bpq100d_count.to_excel(writer, sheet_name='Now taking Rx for Chol', index=True, header=True)
    bpq020_count.to_excel(writer, sheet_name='Told to have HBP', index=True, header=True)
    bpq040a_count.to_excel(writer, sheet_name='Told to take Rx for HBP', index=True, header=True)
    bpq050a_count.to_excel(writer, sheet_name='Now taking Rx for HBP', index=True, header=True)
    diq010_count.to_excel(writer, sheet_name='Told to have diabetes', index=True, header=True)
    diq050_count.to_excel(writer, sheet_name='Now taking insulin', index=True, header=True)
    diq070_count.to_excel(writer, sheet_name='Now taking diabetic pills', index=True, header=True)
    mcq080_count.to_excel(writer, sheet_name='Told to be overweight', index=True, header=True)
    mcq365a_count.to_excel(writer, sheet_name='Told to lose weight', index=True, header=True)
    mcq370a_count.to_excel(writer, sheet_name='Now controlling weight', index=True, header=True)
    mcq365b_count.to_excel(writer, sheet_name='Told to exercise', index=True, header=True)
    mcq370b_count.to_excel(writer, sheet_name='Now increasing exercise', index=True, header=True)
    mcq365c_count.to_excel(writer, sheet_name='Told to reduce salt in diet', index=True, header=True)
    mcq370c_count.to_excel(writer, sheet_name='Now reducing salt in diet', index=True, header=True)
    mcq365d_count.to_excel(writer, sheet_name='Told to reduce fat in diet', index=True, header=True)
    mcq370d_count.to_excel(writer, sheet_name='Now reducing fat in diet', index=True, header=True)