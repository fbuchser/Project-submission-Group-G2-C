# 01 ####
### Packages
print("### 01 Import all Packages\n")

import pandas as pd
import numpy as np
import time

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

from imblearn.ensemble import BalancedRandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
import shap
import matplotlib.patches as mpatches


# 02 ####
### Load data
print("### 02 Load data\n")
X_train = pd.read_csv("./Output/Data/02 13-Preprocessed Data/X_train_final.csv")
y_train = pd.read_csv("./Output/Data/02 13-Preprocessed Data/y_train_final.csv")
X_test = pd.read_csv("./Output/Data/02 13-Preprocessed Data/X_test_final.csv")
y_test = pd.read_csv("./Output/Data/02 13-Preprocessed Data/y_test_final.csv")
y_train = np.ravel(y_train)


# 03 ####
###  Custom scoring function that looks at f1, precision, recall and accuracy
print("### 03 Define a custom scoring function for the hyperparamter tuning\n")
def define_scorer(y_true, y_pred):
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return (f1_macro + precision_macro + recall_macro + accuracy) / 4
custom_scorer = make_scorer(define_scorer, greater_is_better=True)


# 04 ####
### Function to test model performance on training set
print("### 04 Define a function to output the training set performance\n")
def performance(model, model_name ="None", X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test):
    print(f"\033[4mModel: {model_name}\033[0m")

    # Performance for Training Data
    print(f"\033[1mPerformance for Training Data:\033[0m")
    y_pred = model.predict(X_train)
    conf_mat = confusion_matrix(y_train, y_pred)
    print("Confusion Matrix:\n", conf_mat)
    print("Classification report:\n", classification_report(y_train, y_pred, zero_division=0))


# 05 ###
### Balancing dataset with SMOTE and SMOTETomek
print("### 05 Balance dataset with SMOTE and SMOTETomek\n")

# Balance dataset with SMOTETomek
smt = SMOTE(random_state=42)
X_train_SMOTE, y_train_SMOTE = smt.fit_resample(X_train, y_train)

# Balance dataset with SMOTETomek
smto = SMOTETomek(random_state=42)
X_train_SMOTETomek, y_train_SMOTETomek = smto.fit_resample(X_train, y_train)


# 06 ###
### Test the model for SMOTE and SMOTETomek with default parameters
print("### 06 Test the model for SMOTE and SMOTETomek with default parameters\n")
base_model_SMOTE = RandomForestClassifier(random_state = 42)
base_model_SMOTE.fit(X_train_SMOTE, y_train_SMOTE)
performance(base_model_SMOTE, model_name="Base Model SMOTE (Default Random Forest Classifier)", X_train = X_train_SMOTE, y_train = y_train_SMOTE, X_test = X_test)

base_model_SMOTETomek = RandomForestClassifier(random_state = 42)
base_model_SMOTETomek.fit(X_train_SMOTETomek, y_train_SMOTETomek)
performance(base_model_SMOTETomek, model_name="Base Model SMOTETomek (Default Random Forest Classifier)", X_train = X_train_SMOTETomek, y_train = y_train_SMOTETomek,  X_test = X_test)

print("The model predictions for the entire training set are correct, indicating a major overfitting problem with SMOTE and SMOTE-Tomek.\n")


# 07 ###
### Try Hyperparametertuning to reduce overfitting for SMOTETomek
print("### 07 Try Hyperparameter tuning to reduce overfitting for SMOTETomek\n")

# Fit the randomized search on Hyperparameters for SMOTETomek
random_grid = {'n_estimators': [100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000],
                'criterion': ['gini', 'entropy'],
                'max_depth':[3,5,7,10,15,20], # shouldn't be too high otherwise it overfitts (https://www.analyticsvidhya.com/blog/2020/03/beginners-guide-random-forest-hyperparameter-tuning/)
                'min_samples_split': [2, 5, 10], # To prevent overfitting higher is better
                'min_samples_leaf': [1, 2, 4, 10],
                'max_leaf_nodes': [10,15,20,25,30,35,40],
                'max_samples': [0.2,0.4,0.6,0.8,None], 
                'max_features': ['sqrt', 'log2'],
                'bootstrap': [True]}

t1 = time.time()
rf = RandomForestClassifier(random_state = 42)
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, scoring=custom_scorer, n_iter = 100, cv = 3, verbose=3, random_state=42, n_jobs = -1)
rf_random.fit(X_train_SMOTETomek, y_train_SMOTETomek)
best_param = rf_random.best_params_
print()
print(f"Best Hyperparameters: {best_param}\n")
performance(model=rf_random.best_estimator_, model_name='SMOTETomek - randomsearchCV Hyperparameter tuning', X_train=X_train_SMOTETomek, y_train=y_train_SMOTETomek, X_test=X_test)
t2 = time.time()
print(f"The random search for the Hyperparameters took {(t2-t1)/60:.2f} min.\n")

# Evaluation:
print("There is a major overfitting problem with SMOTE and SMOTE-Tomek, which cannot be resolved solely through hyperparameter tuning.\n")


# 08 ####
### Handle Data Imbalance by using the Balanced Random Forest Classifier of imblearn
print("A different aproach to handle data imbalance is to use the Balanced Random Forest Classifier of imblearn.\n")
print("### 08 Test the model with the Balanced Random Forest Classifier and default parameters\n")

# Model with default Hyperparameters (older than version 0.13)
default_hyperparameters = {'bootstrap': True, 'replacement': False, 'sampling_strategy': 'auto'}
clf_old = BalancedRandomForestClassifier(**default_hyperparameters, random_state=42)
clf_old.fit(X_train, y_train)
performance(model = clf_old, model_name = "Base with Balanced Random Forest Classifier and old default parameters")

# Model with default Hyperparameters (version 0.13)
default_hyperparameters = {'bootstrap': False, 'replacement': True, 'sampling_strategy': 'all'}
clf_new = BalancedRandomForestClassifier(**default_hyperparameters, random_state=42)
clf_new.fit(X_train, y_train)
performance(model = clf_new, model_name = "Base with Balanced Random Forest Classifier and new default parameters")

print("There is less overfitting for the Balanced Random Forest Classifier.\n")


# 09 ####
### Hyperparametertuning for the Balanced Random Forest Classifier
print("### 09 Hyperparameter tuning with a randomized search for the Balanced Random Forest Classifier\n")

# Fit the randomized search on Hyperparameters for the Balanced Random Forest Classifier
random_grid = {'n_estimators': [100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000],
                'criterion': ['gini', 'entropy'],
                'max_depth':[3,5,7,10,15,20], # shouldn't be too high otherwise it overfitts (https://www.analyticsvidhya.com/blog/2020/03/beginners-guide-random-forest-hyperparameter-tuning/)
                'min_samples_split': [2, 5, 10], # To prevent overfitting higher is better
                'min_samples_leaf': [1, 2, 4, 10],
                'max_leaf_nodes': [10,15,20,25,30,35,40],
                'max_samples': [0.2,0.4,0.6,0.8,None], 
                'max_features': ['sqrt', 'log2'],
                'sampling_strategy': ['all','not minority','not majority','majority'], 
                'class_weight': ["balanced", "balanced_subsample", None],
                'bootstrap': [True,False], 
                'replacement': [True, False]}

t1 = time.time()
brf = BalancedRandomForestClassifier(random_state = 42)
brf_random = RandomizedSearchCV(estimator = brf, param_distributions = random_grid, scoring=custom_scorer, n_iter = 100, cv = 3, verbose=3, random_state=42, n_jobs = -1)
brf_random.fit(X_train, y_train)
best_param = brf_random.best_params_
print()
print(f"Best Hyperparameters: {best_param}\n")
performance(model=brf_random.best_estimator_, model_name='Balanced Random Forest Classifier - randomsearchCV Hyperparameter tuning')
t2 = time.time()
print(f"The random search for the Hyperparameters took {(t2-t1)/60:.2f} min.\n")


# 10 ####
### Performance Evaluation
print("### 10 Performance Evaluation\n")

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

def performance_evaluation(
        # model_name is a str (e.g. "XGBoost Classifier")
        model_name,
        # data_set is a str (e.g. "Training Set", "Testing Set")
        #  -> use exactly these two string options
        #  -> if you dont use data_set = "Testing Set" then only the classification_report and confusion_matrix will be executed
        data_set,
        # relative_path is a str (e.g. ./Output/Visualisations/03_2 08-Performance Evaluation/)
        #  -> this is the path where the visualisations will be saved without the file names
        relative_path,
        # Your true label values most likely called y_test
        y_test,
        # Your predicted model values
        # y_pred,
        # Your testing features most likely called X_test
        X_test,
        # Your model (e.g. what ever you called your model)
        #  -> find in the code looking something like this: xgb_classifier = xgb.XGBClassifier(**best_params)
        #  -> xgb_classifier should be passed as model here
        model,
        ):
    
    y_pred = model.predict(X_test)

    ## Print Header
    print(f"-- Performance Evaluation for {model_name} - {data_set}")
    print()

    ## Classification report
    print(f"Classification Report:")
    clsf_report = classification_report(
                        y_test,
                        y_pred,
                        zero_division=0,
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
        xticklabels=["Mutatated", "NOS/NEC", "Wildtype"],
        yticklabels=["Mutatated", "NOS/NEC", "Wildtype"]
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
        classes_unencoded = classes

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
                'Model': 'BFR',
                'FPR': fpr['macro-average'],
                'TPR': tpr['macro-average'],
                'ROC': roc_auc['macro-average']}
        )

        macro_plot.to_csv("./Output/Data/03 Summary Results/BFR ROC.csv", sep=',', index=False, encoding='utf-8')

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
        df_roc.columns = ['BRF']
        # Change model name to correct model
        df_roc.to_csv(str('./Output/Data/03 Summary results/ROC AUC Summary - BRF.csv'), sep=',', index=True, encoding='utf-8')

brfc = brf_random.best_estimator_
performance_evaluation(model_name = "BRF", data_set = "Testing Set", relative_path = "./Output/Visualisations/03_1 Random Forest Classifier/", y_test = y_test, X_test = X_test, model = brfc)
print()
performance_evaluation(model_name = "BRF", data_set = "Training Set", relative_path = "./Output/Visualisations/03_1 Random Forest Classifier/", y_test = y_train, X_test = X_train, model = brfc)


# 11 ####
### Feature Importance
print("### 11 Plot the Feature Importance\n")

explainer = shap.TreeExplainer(brfc)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(21, 6))

shap.summary_plot(
    shap_values,
    feature_names=X_test.columns,
    class_names=brfc.classes_,
    plot_type='bar',
    plot_size=(20, 5),
    max_display=10,  
    show=False
    )

# Customize colors
colors = ['teal', 'mediumturquoise', 'paleturquoise']
for i, bar in enumerate(plt.gca().patches):
    bar.set_color(colors[i % len(colors)])

# Add legend for classes
patches = [mpatches.Patch(color=colors[i], label=brfc.classes_[i]) for i in range(len(brfc.classes_))]
plt.legend(handles=patches, title="Classes",fontsize=18, title_fontsize=20)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.title("Feature Importance (SHAP Barplot) - Balanced Random Forest", fontsize=24)
plt.ylabel("Feature Names", fontsize=18)
plt.xlabel("SHAP Value (average impact on model output)", fontsize=18)

plt.savefig("./Output/Visualisations/03_1 Random Forest Classifier/SHAP barplot - BRF.png", bbox_inches="tight")
