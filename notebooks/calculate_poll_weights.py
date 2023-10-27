#%%
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from ElectionForecasting.src.root import ROOT_DIR
# Reading the data
poll_data = pd.read_csv(f'{ROOT_DIR}/data/dataland/dataland_polls_1984_2023.csv', index_col=0)
election_data = pd.read_csv(f'{ROOT_DIR}//data/dataland/dataland_election_results_1984_2023.csv', index_col=0)

# Columns for poll shares and election shares
poll_share_columns = ['cc_poll_share', 'dgm_poll_share', 'pdal_poll_share', 'ssp_poll_share', 'undecided_poll_share']
election_share_columns = ['cc_share', 'dgm_share', 'pdal_share', 'ssp_share']

# Box plots for poll shares and election shares
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(data=poll_data[poll_share_columns])
plt.title('Boxplot of Poll Shares')
plt.subplot(1, 2, 2)
sns.boxplot(data=election_data[election_share_columns])
plt.title('Boxplot of Election Shares')
plt.tight_layout()
plt.show()

# Label encode categorical variables in poll data
label_columns = ['pollster', 'geography', 'mode', 'population_surveyed', 'sponsor']
label_encoder = LabelEncoder()
for col in label_columns:
    poll_data[col] = label_encoder.fit_transform(poll_data[col].astype(str))

# Create interaction terms
interaction_terms = []
for poll_col in poll_share_columns:
    for meta_col in label_columns:
        interaction_term = f"{meta_col}_x_{poll_col}"
        interaction_terms.append(interaction_term)
        poll_data[interaction_term] = poll_data[poll_col] * poll_data[meta_col]

# Averaging poll data and election data by year for merging
avg_poll_data = poll_data.groupby('year')[poll_share_columns + interaction_terms].mean().reset_index()
avg_election_data = election_data.groupby('year')[election_share_columns].mean().reset_index()

# Merge on the 'year' column
merged_data = pd.merge(avg_poll_data, avg_election_data, on='year', how='inner')

# Prepare feature matrix X and target matrix Y
X = merged_data[interaction_terms]
Y = merged_data[election_share_columns]

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model
rf_model.fit(X_train, Y_train)

# Extract feature importances
feature_importances = rf_model.feature_importances_

# Map importances to the range [0.9, 1.1]
def map_to_range(value, src_range, dst_range):
    src_min, src_max = src_range
    dst_min, dst_max = dst_range
    return dst_min + ((value - src_min) / (src_max - src_min)) * (dst_max - dst_min)

mapped_importances = {key: map_to_range(val, (0, 1), (0.9, 1.1)) for key, val in zip(X.columns, feature_importances)}

# Normalize mapped importances so they sum to the number of poll_share_columns
def normalize_weights_to_sum(weights_dict, target_sum):
    normalized_weights_dict = {}
    for meta_col in label_columns:
        sum_weights = sum(weights_dict[f"{meta_col}_x_{poll_col}"] for poll_col in poll_share_columns)
        for poll_col in poll_share_columns:
            interaction_term = f"{meta_col}_x_{poll_col}"
            normalized_weights_dict[interaction_term] = (weights_dict[interaction_term] / sum_weights) * target_sum
    return normalized_weights_dict

normalized_importances = normalize_weights_to_sum(mapped_importances, len(poll_share_columns))

# Apply constrained and normalized weights to the original poll data
def apply_constrained_normalized_weights(row):
    adjusted_row = row.copy()
    for poll_col in poll_share_columns:
        for meta_col in label_columns:
            interaction_term = f"{meta_col}_x_{poll_col}"
            weight = normalized_importances.get(interaction_term, 1)
            adjusted_row[poll_col] *= weight
    return adjusted_row

adjusted_poll_data_constrained_normalized = poll_data.apply(apply_constrained_normalized_weights, axis=1)

# Average the constrained and normalized adjusted poll data by year for comparison
avg_adjusted_poll_data_constrained_normalized = adjusted_poll_data_constrained_normalized.groupby('year')[poll_share_columns].mean().reset_index()

# Merge the averaged constrained and normalized adjusted poll data and election data for common years
merged_adjusted_data_constrained_normalized = pd.merge(avg_adjusted_poll_data_constrained_normalized, avg_election_data, on='year', how='inner')

# Line plots for original, constrained and normalized adjusted poll, and election shares
plt.figure(figsize=(20, 16))
for i, (poll_col, election_col) in enumerate(zip(poll_share_columns, election_share_columns), 1):
    plt.subplot(2, 2, i)
    sns.lineplot(data=merged_data, x='year', y=poll_col, label='Original Poll Share', marker='o', color='blue')
    sns.lineplot(data=merged_adjusted_data_constrained_normalized, x='year', y=poll_col, label='Constrained & Normalized Adjusted Poll Share', marker='s', color='green')
    sns.lineplot(data=merged_adjusted_data_constrained_normalized, x='year', y=election_col, label='Election Share', marker='x', color='red')
    plt.title(f'Original vs Constrained & Normalized Adjusted Poll vs Election Shares for {poll_col.split("_")[0].upper()}')
    plt.legend()
plt.tight_layout()
plt.show()
#%%
normalized_importances
#%%
poll_data