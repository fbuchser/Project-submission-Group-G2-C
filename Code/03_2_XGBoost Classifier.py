# 01 ####
### Packages
print("\n### 01 Packages\n")

# 02
import pandas as pd

# 03
# 03.1
from sklearn.preprocessing import StandardScaler

# 03.2
from sklearn.preprocessing import LabelEncoder

# 03.3
from sklearn.model_selection import train_test_split

# 03.4
import numpy as np

# 04
from collections import Counter

# 05
from sklearn.utils.class_weight import compute_class_weight

# 06
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import xgboost as xgb
from sklearn.metrics import f1_score

# 08
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle

# 09
import shap



# 02 ####
### Read Feature and Label - Training and Test Sets
print("### 02 Read Feature and Label - Training and Testing Sets\n")

# X_train (after feature selection, balanced)
X_train_final = pd.read_csv("./Output/Data/02 13-Preprocessed Data/X_train_final.csv", header=0)
# X_test  (after feature selection, unbalanced)
X_test_final = pd.read_csv("./Output/Data/02 13-Preprocessed Data/X_test_final.csv", header=0)
# y_train (unencoded, balanced)
y_train_final = pd.read_csv("./Output/Data/02 13-Preprocessed Data/y_train_final.csv", header=0)
# y_test  (unencoded, unbalanced)
y_test_final = pd.read_csv("./Output/Data/02 13-Preprocessed Data/y_test_final.csv", header=0)



# 03 ####
### Model Specific Preprocessing
print("### 03 Model Specific Preprocessing\n")


# 03.1 ####
### Scaling
print("### - 03.1 Scaling\n")

# Numerical columns after feature selection
num_cols_train = X_train_final.select_dtypes(include=['int', 'float']).columns.tolist()
num_cols_test  = X_test_final.select_dtypes(include=['int', 'float']).columns.tolist()


## Sklearn StandardScaler
sc = StandardScaler()

X_train_final[num_cols_train] = sc.fit_transform(X_train_final[num_cols_train])
X_test_final[num_cols_test] = sc.transform(X_test_final[num_cols_test])


# 03.2 ####
### Label Encoding for XGBoost
print("### - 03.2 Label Encoding for XGBoost\n")

## Sklearn LabelEncoder
label_encoder = LabelEncoder()

# Fit LabelEncoder and transform string labels to integers
y_train_encoded = label_encoder.fit_transform(y_train_final['IDH1'])
y_test_encoded = label_encoder.transform(y_test_final['IDH1'])


# 03.3 ####
### Validation Set for Hyperparameter Tuning
print("### - 03.3 Validation Set for Hyperparameter Tuning\n")

# Split Traning Set into Train and Validation
X_train, X_val, y_train, y_val = train_test_split(X_train_final, y_train_encoded, test_size=0.2, stratify=y_train_encoded, random_state=42)


# 03.4 ####
### Prepare Data Sets for Model Implementation
print("### - 03.4 Prepare Data Sets for Model Implementation\n")

# Convert pandas DataFrame to numpy array
X_train, X_val, X_test, y_train, y_val, y_test = (
    np.array(X_train),
    np.array(X_val),
    np.array(X_test_final),
    np.array(y_train).ravel(),
    np.array(y_val).ravel(),
    np.array(y_test_encoded).ravel()
)



# 04 ####
### Final Data Overview before Model Implementation
print("### 04 Final Data Overview before Model Implementation\n")

# Print Final Data Set Information
print(f"X_train Data Set - Shape: {X_train.shape}")
print(f"X_val Data Set   - Shape: {X_val.shape}")
print(f"X_test Data Set  - Shape: {X_test.shape}")
print(f"y_train Data Set - Shape: {y_train.shape}")
print(f"y_val Data Set   - Shape: {y_val.shape}")
print(f"y_test Data Set  - Shape: {y_test.shape}")
print()
print(f"-- y_train:  {Counter(y_train)}")
print(f"-- y_val:    {Counter(y_val)}")
print(f"-- y_test:   {Counter(y_test)}")
print()



# 05 ####
### Compute Class Weights to Handle Label Imbalance
print("### 05 Compute Class Weights to Handle Label Imbalance\n")

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)

# New DataFrame
Classes_Weights = {
    "Unencoded Classes": np.unique(y_train_final),
    "Encoded Classes": np.unique(y_train),
    "Computed Class Weights": class_weights.round(4)
}

df_Classes_Weights = pd.DataFrame(Classes_Weights)

print(f"Table of all Classes and assigned Weights: \n{df_Classes_Weights}")

# Convert DataFrame to LaTeX table
latex_table_classes_weights = df_Classes_Weights.to_latex(index=False, formatters={'Computed Class Weights': (lambda x: "{:.4f}".format(x))})

# Save LaTeX table to a text file
with open("./Output/Visualisations/03_2 XGB/Classes and Weights - XGB.txt", "w+") as f:
    f.write(latex_table_classes_weights)

print("LaTeX table has been saved to 'Classes and Weights.txt'.")
print()

## Set sample weights for each observation based on class weights
sample_weights = np.zeros_like(y_train, dtype=float)
for i, value in enumerate(y_train):
    sample_weights[i] = class_weights[value]



# 06 ####
### Hyperparameter Tuning with Hyperopt
print("### 06 Hyperparameter Tuning with Hyperopt\n")

## Parameter ranges for hyperparameter tuning
space = {
    'max_depth': hp.quniform("max_depth", 3, 10, 1),
    'gamma': hp.uniform ('gamma', 1, 5),
    'reg_alpha' : hp.quniform('reg_alpha', 0, 5, 0.5),
    'reg_lambda' : hp.uniform('reg_lambda', 0, 2),
    'colsample_bytree' : hp.uniform('colsample_bytree', 0.5, 1),
    'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
    'n_estimators': hp.quniform('n_estimators', 100, 500, 10),
    'subsample': hp.uniform('subsample', 0.1, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
    'seed': 42
}


def objective(space):
    clf=xgb.XGBClassifier(
        n_estimators = int(space['n_estimators']),
        max_depth = int(space['max_depth']),
        gamma = float(space['gamma']),
        reg_alpha = float(space['reg_alpha']),
        reg_lambda = float(space['reg_lambda']),
        min_child_weight = int(space['min_child_weight']),
        colsample_bytree = float(space['colsample_bytree']),
        subsample = float(space['subsample']),
        learning_rate = float(space['learning_rate']),

        # Specific for multi-class classification
        objective='multi:softmax',
        num_class=len(np.unique(y_train)),
        early_stopping_rounds=10,
        eval_metric='merror',
        n_jobs=-1,
    )
    
    evaluation = [(X_val, y_val)]

    clf.fit(
        X_train,
        y_train,
        eval_set=evaluation,
        verbose=False,
        sample_weight=sample_weights
    )
    
    pred = clf.predict(X_val)
    
    score = f1_score(y_val, pred, average='macro')

    return {'loss': -score, 'status': STATUS_OK }


trials = Trials()

rstate = np.random.default_rng(42)

best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 500,
                        trials = trials,
                        rstate = rstate
                        )

print()
print(f"The best hyperparameters are : \n{best_hyperparams}")
print()



# 07 ####
### XGBoost Model with the best Hyperparameters
print("### 07 XGBoost Model with the best Hyperparameters\n")

# Initialize XGBoost classifier with the best hyperparameters
best_params = {
    'max_depth': int(best_hyperparams['max_depth']),
    'gamma': float(best_hyperparams['gamma']),
    'reg_alpha': float(best_hyperparams['reg_alpha']),
    'reg_lambda': float(best_hyperparams['reg_lambda']),
    'colsample_bytree': float(best_hyperparams['colsample_bytree']),
    'min_child_weight':int(best_hyperparams['min_child_weight']),
    'n_estimators': int(best_hyperparams['n_estimators']),
    'subsample': float(best_hyperparams['subsample']),
    'learning_rate': float(best_hyperparams['learning_rate']),
    'objective': 'multi:softmax',
    'num_class': len(np.unique(y_train)),
    'random_state': 42
    }

xgb_classifier = xgb.XGBClassifier(**best_params)

# Train the classifier on the entire training dataset
xgb_classifier.fit(X_train, y_train, sample_weight=sample_weights)

# Check overfitting by predicting the training dataset
y_pred_train = xgb_classifier.predict(X_train)

# Make predictions on the test dataset
y_pred = xgb_classifier.predict(X_test)



# 08 ####
### Performance Evaluation
print("### 08 Performance Evaluation\n")

# Save classification report
def save_classification_report(y_test, y_pred, relative_path, data_set, name):
    # Report as dictionary
    report = classification_report(
                y_test,
                y_pred,
                zero_division=0,
                output_dict=True)
    
    report.update({"accuracy": {"precision": None, "recall": None, "f1-score": report["accuracy"], "support": report['macro avg']['support']}})

    # Convert dictionary to DataFrame
    df_clsf_report = pd.DataFrame.from_dict(report, orient='index')
    df_clsf_report = df_clsf_report.round({'precision': 2, 'recall': 2, 'f1-score': 2, 'support': 0})

    # Save LaTeX table to a text file
    title_clsfr = f"Classification Report - {name} ({data_set})"
    df_clsf_report.to_csv(str('./Output/Data/03 Summary results/' + title_clsfr + '.csv'), sep=',', index=True, encoding='utf-8')
    
    print(f"The classification report has been saved to: '{str('./Output/Data/03 Summary results/' + title_clsfr + '.csv')}'.")
    print()

# Performance Evaluation Function
def performance_evaluation(
        # model_name is a str (e.g. "XGBoost Classifier")
        model_name,
        # data_set is a str (e.g. "Training Set", "Testing Set")
        #  -> use exactly these two string options
        #  -> if you dont use data_set = "Testing Set" then only the classification_report and confusion_matrix will be executed
        data_set,
        # relative_path is a str (e.g. ./Output/Visualisations/03_2 XGB/)
        #  -> this is the path where the visualisations will be saved without the file names
        relative_path,
        # Your true label values most likely called y_test
        y_test,
        # Your predicted model values
        y_pred,
        # Your testing features most likely called X_test
        X_test,
        # Your model (e.g. what ever you called your model)
        #  -> find in the code looking something like this: xgb_classifier = xgb.XGBClassifier(**best_params)
        #  -> xgb_classifier should be passed as model here
        model,
        ):

    ## Print Header
    print(f"-- Performance Evaluation for {model_name} - {data_set}")
    print()

    ## Classification report
    print(f"Classification Report:")
    clsf_report = classification_report(
                        y_test,
                        y_pred,
                        zero_division=0,
                        target_names=label_encoder.classes_,
                        )
    print(clsf_report)

    # Save classification report as DataFrame
    save_classification_report(y_test, y_pred, relative_path, data_set, name = model_name)

    ## Plot confusion matrix
    fig, ax = plt.subplots()
    title_cm = f"Confusion Matrix - {model_name} ({data_set})"
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        annot_kws={"size": 12},
        fmt='d',
        cmap='Blues',
        ax=ax,
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
        )
    ax.set_title(title_cm)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    fig.savefig(str(relative_path + title_cm + ".png"), bbox_inches="tight")

    if data_set == "Testing Set":
        ## ROC AUC
        y_pred_proba = model.predict_proba(X_test)
       
        ## Multiclass ROC curves

        # Binarize the output
        classes = np.unique(y_test)
        classes_unencoded = label_encoder.inverse_transform(classes)

        y_test_bin = label_binarize(y_test, classes=classes)
        n_classes = y_test_bin.shape[1]
        
        ## Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        # Compute macro-average ROC curve and ROC area
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr['macro-average'] = all_fpr
        tpr['macro-average'] = mean_tpr
        roc_auc['macro-average'] = auc(fpr['macro-average'], tpr['macro-average'])

        # Save macro-average ROC curve to csv for model comparison            
        macro_plot = pd.DataFrame(
            data={
                'Model': 'XGB',
                'FPR': fpr['macro-average'],
                'TPR': tpr['macro-average'],
                'ROC': roc_auc['macro-average']}
        )

        macro_plot.to_csv("./Output/Data/03 Summary Results/XGB ROC.csv", sep=',', index=False, encoding='utf-8')


        ## Plot all ROC curves
        file_name_roc = f"ROC Curves - {model_name} ({data_set})"
        title_roc = f"One-vs-Rest & Macro-averaged ROC Curves\n with {model_name}"

        fig, ax = plt.subplots(figsize=(7, 7))

        # Plot macro-average ROC curve
        plt.plot(
            fpr['macro-average'],
            tpr['macro-average'],
            label=f"macro-average ROC curve (area = {roc_auc['macro-average']:.2f})",
            color='forestgreen',
            linestyle='-.',
            lw=2
            )

        # Plot ROC curve for each class
        colors = cycle(['mediumorchid', 'lightcoral', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=2,
                label=f"ROC curve of class {classes_unencoded[i]} (area = {roc_auc[i]:.2f})"
                )
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim([0.0, 1.05])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title_roc)
        ax.legend(loc="lower right")
        fig.savefig(str(relative_path + file_name_roc + ".png"), bbox_inches="tight")

        ## Print ROC AUC Scores
        df_roc = pd.DataFrame()
        print(f"macro-average ROC AUC: {roc_auc['macro-average']:.2f}")
        df_roc['macro-average ROC AUC'] = [round(roc_auc['macro-average'], 2)]
        for i in range(n_classes):
            print(f"ROC AUC of class {classes_unencoded[i]}: {roc_auc[i]:.2f}")
            df_roc[f"ROC AUC of class {classes_unencoded[i]}"] = [round(roc_auc[i], 2)]
        df_roc = df_roc.T
        df_roc.columns = ['XGBoost']
        # Change model name to correct model
        df_roc.to_csv(f"./Output/Data/03 Summary results/ROC AUC Summary - {model_name}.csv", sep=',', index=True, encoding='utf-8')


# Performance Evaluation for Training Set
performance_evaluation(
    # model_name
    "XGB",
    # data_set
    "Training Set",
    # relative_path
    "./Output/Visualisations/03_2 XGB/",
    # y_test
    y_train,
    # y_pred 
    y_pred_train,
    # X_test
    X_train,
    # model
    xgb_classifier
    )

# Performance Evaluation for Testing Set
performance_evaluation(
    # model_name
    "XGB",
    # data_set
    "Testing Set",
    # relative_path
    "./Output/Visualisations/03_2 XGB/",
    # y_test
    y_test,
    # y_pred 
    y_pred,
    # X_test
    X_test,
    # model
    xgb_classifier
    )

print()

# 09 ####
### Feature Importance
print("### 09 Feature Importance\n")

explainer = shap.TreeExplainer(xgb_classifier)
shap_values = explainer.shap_values(X_test_final)

plt.figure(figsize=(21, 6))

shap.summary_plot(
    shap_values,
    feature_names=X_test_final.columns,
    class_names=label_encoder.classes_,
    plot_type='bar',
    plot_size=(20, 5),
    max_display=10, 
    show=False
    )

# Customize colors
colors = ['mediumturquoise', 'paleturquoise', 'teal']
for i, bar in enumerate(plt.gca().patches):
    bar.set_color(colors[i % len(colors)])

# Fromat Legend
plt.legend(title="Classes",fontsize=18, title_fontsize=20)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.title("Feature Importance (SHAP Barplot) - XGBoost", fontsize=24)
plt.ylabel("Feature Names", fontsize=18)
plt.xlabel("SHAP Value (average impact on model output)", fontsize=18)

plt.savefig("./Output/Visualisations/03_2 XGB/SHAP barplot - XGB.png", bbox_inches="tight")