#01 ####
### Packages
print("\n### 01 Packages\n")

# 02
import pandas as pd

# 04
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 10
from sklearn.model_selection import train_test_split

# 11
import time
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# 12
from mrmr import mrmr_classif



#02 ####
### Read data files 
print("### 02 Read Data Files\n")

radiological = pd.read_csv("./Data/radFeatures_UPENN.csv", header=0, index_col=0)
clinical = pd.read_csv("./Data/clinFeatures_UPENN.csv", header=0,  index_col=0)



#03 ####
### Basic data understanding
print("### 03 Basic Data Understanding\n")

## Clinical features
c_shape = clinical.shape

c_miss = clinical.isnull().sum()
c_unique = []
for i in range(0,clinical.shape[1]):
    num_unique = len(clinical.iloc[:, i].unique())
    c_unique.append(num_unique)

df_cmissing = pd.DataFrame(data={'Clinical Feature': c_miss.index, 'Sum missing': c_miss.values, 'Number of unique values': c_unique})

print(f"-- Clinical Shape: \n{c_shape}")
print()
print(f"-- Clinical missing: \n{df_cmissing}")
print()

"""
# Print unique values
print("-- Unique values per clinical feature")
for i in range(0,clinical.shape[1]):
    print(clinical.columns[i], clinical.iloc[:, i].unique(), sep="\n")
print()
"""

# Find clinical features with only 1 unique value
c_missing_cols = df_cmissing['Clinical Feature'][df_cmissing['Number of unique values'] == 1].to_list()


## Radiological features
r_shape = radiological.shape

r_miss_f = radiological.isnull().sum()

df_rmissing_f = pd.DataFrame(data={'Radiological': r_miss_f.index, 'Sum missing': r_miss_f.values})
percent = 0.1
rmissf1_sum = df_rmissing_f[df_rmissing_f['Sum missing'] == 0]['Sum missing'].count()
rmissf2_sum = df_rmissing_f[df_rmissing_f['Sum missing'] != 0]['Sum missing'].count()
rmissf3_sum = df_rmissing_f[df_rmissing_f['Sum missing'] > (r_shape[0] * percent)]['Sum missing'].count()

print(f"-- Radiological Shape: \n{r_shape}")
print()
print(f"-- Radiological missing per feature: \n{r_miss_f.sort_values()}")
print()
print(f"Max number missing (per radiological feature): {max(r_miss_f)}")
print()
print(f"Missing no data: {rmissf1_sum} ~ {(rmissf1_sum/r_shape[1])*100:.2f}% of radiological features.")
print(f"Missing data: {rmissf2_sum} ~ {(rmissf2_sum/r_shape[1])*100:.2f}% of radiological features")
print(f"Missing more than {percent*100}% of feature values: {rmissf3_sum} ~ {(rmissf3_sum/r_shape[1])*100:.2f}% of radiological features.")
print()

# Find radiological features with only 3 unique values
num_unique = 3
faulty_cols = []
faulty_col_index = []
idx = 0

for idx_num, col in enumerate(radiological.columns):
    num_values = radiological[col].unique().shape[0]
    
    tmp = []
    if num_values <= num_unique:
        tmp.append(col)
        tmp.append(idx_num)
        tmp.append(num_values)
        tmp.append(radiological[col].unique())
    
        faulty_cols.append(tmp)
        idx += 1

df_faulty_cols = pd.DataFrame(faulty_cols)
df_faulty_cols.columns = ['Col Names', 'Col Index', 'Number of Unique Values', 'Unique Values']
df_faulty_cols = df_faulty_cols.sort_values(by=['Number of Unique Values'])
# Saved df as .csv file to explore in Excel
# If num_unique > 3 the registered values get significantly less suspicious 
df_faulty_cols.to_csv("./Output/Data/02 03-Understanding/faulty_cols.csv", sep=',', index=False, encoding='utf-8')



#04 ####
### Data Visualisation
print("### 04 Data Visualisation\n")

## Missing Values Heatmap
image_type = pd.read_csv("./Output/Data/01 Understanding/feature_exploration_imaging.csv", header=0)
measurement_type = pd.read_csv("./Output/Data/01 Understanding/feature_exploration_measurement.csv", header=0)

# removes last char i.e. '_'
image_type['ImageType'] = image_type['ImageType'].map(lambda x: x[:-1])

# prepare heatmap data frame
heatmap_data = pd.DataFrame(index=image_type['ImageType'], columns=measurement_type['FeatureName'], dtype=int)

# fill heatmap data frame with missing values
for str_start in heatmap_data.index:
    for str_end in heatmap_data.columns:
        combined_str = f"{str_start}_{str_end}"
        matching_row = df_rmissing_f[df_rmissing_f['Radiological'].values == combined_str]
        
        heatmap_data.loc[str_start, str_end] = ((matching_row['Sum missing'].values[0])/611)*100

fig = plt.figure(figsize=(35,10))

ax = sns.heatmap(heatmap_data, 
                 cmap='YlGnBu', 
                 square=True, 
                 cbar=True,
                 cbar_kws={"pad": 0.01},
                 vmin=0, 
                 vmax=30)

# Colorbar modification
cbar = ax.collections[0].colorbar
cbar.set_ticks([0, 10, 20, 30])
cbar.set_ticklabels(['0%', '10%', '20%', '30%'])

# Set axis labels
plt.xticks(rotation=45, ha='right')
plt.xlabel('Radiomic Feature Types')
plt.ylabel('Image Types')

# Set titel
plt.title('Heatmap of all Radiological Features with Missing Values as Percentage of Observations')

# Layout
plt.subplots_adjust(top=0.7, bottom=0.6)
plt.tight_layout()

# Save plot
plt.savefig('./Output/Visualisations/02 04-Overview/Missing Data_Matrix.png', bbox_inches="tight")


## Histplot Label Distribution
clinical['IDH1'] = pd.Categorical(clinical['IDH1'], categories=['Wildtype', 'NOS/NEC', 'Mutated'], ordered=True)

fig = plt.figure(figsize=(7,10))

ax = sns.countplot(data=clinical,
                   x=clinical['IDH1'],
                   palette="mako_r")

# Define the scale of the y label 
pat_num = clinical['IDH1'].shape[0]
ax.set_ylim(0,pat_num+20)
ax.set_yticks(np.append(np.arange(0, pat_num, 100), pat_num))
ax.axhline(y=pat_num, color = "grey", linestyle = ':')

# Show count values
for bars in ax.containers:
    ax.bar_label(bars)

# Set the title
plt.title("Countplot for the label categories (in a total of 611 observations)")

# Layout
plt.tight_layout()

# Save plot
plt.savefig('./Output/Visualisations/02 04-Overview/Label_distribution.png', bbox_inches="tight")


## Histplot Age Distribution
fig = plt.figure(figsize=(7,10))

ax = sns.histplot(
    data=clinical,
    x=clinical['Age_at_scan_years'],
    hue=clinical['IDH1'],
    palette="mako_r",
    kde=True,
    multiple='stack',
    bins=25)  

# Define the scale of the y label 
ax.set_ylim(0,65)
ax.set_yticks(np.arange(0, 61, 10))
ax.axhline(y=pat_num, color = "grey", linestyle = ':')

"""
# Show count values
for bars in ax.containers:
    ax.bar_label(bars)
"""

# Set the title
plt.title("Countplot for the age distribution of patients")

# Layout
plt.tight_layout()

# Save plot
plt.savefig('./Output/Visualisations/02 04-Overview/Patient Age_distribution.png', bbox_inches="tight")



#05 ####
### Delete Feature Cols of Unnecessary Features
print("### 05 Delete Feature Cols of Unnecessary Features\n")

## Clinical features
# Delete empty columns/features clinical
clinical_revised = clinical.drop(columns=c_missing_cols)
print(f"-- Clinical_revised: \nColumns: {clinical_revised.columns.values}, \nShape: {clinical_revised.shape}")
print()

# Select relevant clinical features
clinical_label = pd.DataFrame(clinical_revised[['IDH1']])
clinical_features_relevant = pd.DataFrame(clinical_revised[['Gender', 'Age_at_scan_years']])
print(f"-- Clinical_revised: \nColumns: {clinical_features_relevant.columns.values},  \nShape: {clinical_features_relevant.shape}")
print()

## Radiological features
# Delete empty/suspicious columns/features radiological
radiological_revised = radiological.drop(columns=df_faulty_cols['Col Names'].to_list())
print(f"-- Clinical_revised: \nShape: {radiological_revised.shape}")
print()



#06 ####
### Data Types
print("### 06 Data Types\n")

## Clinical features
# Get the data types: clinical
c_datatypes = clinical_features_relevant.dtypes
print(f"-- Data types Clinical_relevant: \n{c_datatypes}")
print()

## Radiological features
# Get the data types: radiological
r_datatypes = radiological_revised.dtypes
print(f"-- Data types Radiological_revised: \n{r_datatypes}")
print()



#07 ####
### Encoding Categorical Data (only in clinical features necessary)
print("### 07 Encoding Categorical Data (only in clinical features necessary)\n")

## Clinical features
cat_cols = clinical_features_relevant.select_dtypes(include=['object', 'category']).columns.tolist()
clinical_relevant_encoded = pd.get_dummies(clinical_features_relevant, columns=cat_cols, drop_first=True)



#08 ####
### Merge Clinical Encoded Features and Radiological Features to New Dataframe
print("### 08 Merge Clinical Encoded Features and Radiological Features to New Dataframe\n")

# Idea join on Subject ID (Index)
df_all_features = clinical_relevant_encoded.join(radiological_revised)

print(f"-- Dataframe of all relevant Features: \nShape: {df_all_features.shape}")
print()



#09 ####
### Construct Features and Labels
print("### 09 Construct Features and Labels\n")

X = df_all_features
X.to_csv("./Euler folder/Data/X.csv", sep=',', index=False, encoding='utf-8')
# Label unencoded because of later preprocessing steps
y_unencoded = clinical_label



#10 ####
### Training and Testing Set
print("### 10 Training and Testing Set\n")

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train_ue, y_test_ue = train_test_split(X, y_unencoded, test_size=0.2, random_state=42)

# Save Data Splits
df_X_train = pd.DataFrame(X_train)
df_X_train.to_csv("./Output/Data/02 10-Splits/X_train.csv", sep=',', index=False, encoding='utf-8')
df_X_train.to_csv("./Euler folder/Data/X_train.csv", sep=',', index=False, encoding='utf-8')
df_X_test = pd.DataFrame(X_test)
df_X_test.to_csv("./Output/Data/02 10-Splits/X_test.csv", sep=',', index=False, encoding='utf-8')
df_X_test.to_csv("./Euler folder/Data/X_test.csv", sep=',', index=False, encoding='utf-8')


df_y_train_ue = pd.DataFrame(data=y_train_ue)
df_y_train_ue.to_csv("./Output/Data/02 10-Splits/y_train_unencoded.csv", sep=',', index=False, encoding='utf-8')
df_y_test_ue = pd.DataFrame(data=y_test_ue)
df_y_test_ue.to_csv("./Output/Data/02 10-Splits/y_test_unencoded.csv", sep=',', index=False, encoding='utf-8')



#11 ####
### Missing Value Imputation
print("### 11 Missing Value Imputation\n")

# Convert pandas DataFrame to numpy array
X_train, X_test = (
    np.array(X_train),
    np.array(X_test),
)

# Estimator
est = RandomForestRegressor(
        n_jobs=-1, # The number of jobs to run in parallel. 1 means using all processors. # 20 CPU cores on euler -> 496.10 minutes (~8.27h)
        random_state=42,
        n_estimators=50, # Half of default to reduce compution time.
    )

# Imputer
imp = IterativeImputer(
        estimator=est,
        random_state=42,
        max_iter=5, # Half of default to reduce compution time.
    )

"""
# Imputing process
t1_train = time.time()
X_train_imputed = imp.fit_transform(X_train)
t2_train = time.time()

t1_test = time.time()
X_test_imputed = imp.transform(X_test)
t2_test = time.time()

# Save Imputed Data
df_X_train_imputed = pd.DataFrame(X_train_imputed)
df_X_test_imputed = pd.DataFrame(X_test_imputed)

# Add names to imputed data
df_X_train_imputed.columns = X.columns
df_X_test_imputed.columns = X.columns

# Save Imputed data to .csv file
df_X_train_imputed.to_csv("./Output/Data/02 11-Imputations/X_train_imputed.csv", sep=',', index=False, encoding='utf-8')
df_X_test_imputed.to_csv("./Output/Data/02 11-Imputations/X_test_imputed.csv", sep=',', index=False, encoding='utf-8')

print(f"The training imputation process took {(t2_train - t1_train) / 60:.2f} minutes.")
print(f"The testing imputation process took {(t2_test - t1_test) / 60:.2f} minutes.")
print()
"""

# open imputations (without imputing)
df_X_train_imputed = pd.read_csv("./Output/Data/02 11-Imputations/X_train_imputed.csv", header=0)
df_X_test_imputed = pd.read_csv("./Output/Data/02 11-Imputations/X_test_imputed.csv", header=0)



#12 ####
### Feature Selection with mRMR
print("### 12 Feature Selection with mRMR\n")

## minimum Redundancy - Maximum Relevance (mRMR)

# Data used for feature selection
"""
- Imputed X_train data set    :  df_X_train_imputed
- Unencoded Y_train data set  :  df_y_train_ue
"""
df_y_train_ue = pd.read_csv("./Output/Data/02 10-Splits/y_train_unencoded.csv", header=0)


# Select top 300 features using mRMR
t1_mrmr = time.time()
selected_features = mrmr_classif(X=df_X_train_imputed, y=df_y_train_ue, K=300)
t2_mrmr = time.time()

print()
print(f"The mRMR feature selection took {(t2_mrmr - t1_mrmr) / 60:.2f} minutes.")
print()

# Construct X_train and X_test based on selected features
X_train_selected = pd.DataFrame(data=df_X_train_imputed.loc[:, selected_features])
X_test_selected = pd.DataFrame(data=df_X_test_imputed.loc[:, selected_features])

# Save selected features
X_train_selected.to_csv("./Output/Data/02 12-Feature Selection/X_train_selected300_mrmr.csv", sep=',', index=False, encoding='utf-8')
X_test_selected.to_csv("./Output/Data/02 12-Feature Selection/X_test_selected300_mrmr.csv", sep=',', index=False, encoding='utf-8')



#13 ####
### Final Data Sets after Preprocessing
print("### 13 Final Data Sets after Preprocessing\n")

# X_train (after feature selection, balanced)
X_train_selected.to_csv("./Output/Data/02 13-Preprocessed Data/X_train_final.csv", sep=',', index=False, encoding='utf-8')

# X_test  (after feature selection, unbalanced)
X_test_selected.to_csv("./Output/Data/02 13-Preprocessed Data/X_test_final.csv", sep=',', index=False, encoding='utf-8')

# y_train (unencoded, balanced)
df_y_train_ue.to_csv("./Output/Data/02 13-Preprocessed Data/y_train_final.csv", sep=',', index=False, encoding='utf-8')

# y_test  (unencoded, unbalanced)
df_y_test_ue.to_csv("./Output/Data/02 13-Preprocessed Data/y_test_final.csv", sep=',', index=False, encoding='utf-8')

# Print Final Data Set Information
print(f"Final X_train Data Set - Shape: {X_train_selected.shape}")
print(f"Final X_test Data Set  - Shape: {X_test_selected.shape}")
print(f"Final y_train Data Set - Shape: {df_y_train_ue.shape}")
print(f"Final y_test Data Set  - Shape: {df_y_test_ue.shape}")
print()


