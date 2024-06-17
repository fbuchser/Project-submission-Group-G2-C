# 01 Load Packages

from itertools import cycle
from matplotlib.pylab import randint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.calibration import label_binarize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures,StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
import matplotlib.patches as mpatches


# 02 Load data 
X_train = pd.read_csv("./Output/Data/02 13-Preprocessed Data/X_train_final.csv", header=0)
X_test = pd.read_csv("./Output/Data/02 13-Preprocessed Data/X_test_final.csv", header=0)
y_train = pd.read_csv("./Output/Data/02 13-Preprocessed Data/y_train_final.csv", header=0)
y_test = pd.read_csv("./Output/Data/02 13-Preprocessed Data/y_test_final.csv", header=0)


# 03 Preprocessing for Logistic Regression

# 03.1 Convert y-values to 1D arrays
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# 03.2 Scaling the training and test datasets
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 03.3 Label encoding for Logistic regression
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


# 04 Define the parameter grid

# Define the parameter grid
#grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"], 'solver': ['liblinear'],'class_weight': ['balanced']}
#grid = {"C": np.logspace(-4, 4, 10), "penalty": ["l1", "l2"], 'solver': ['liblinear'],'class_weight': ['balanced']}
#grid = {"C": np.logspace(-5, 5, 11), "penalty": ["l1", "l2"], 'solver': ['liblinear'],'class_weight': ['balanced']}
#grid = {"C": np.logspace(-6, 6, 13), "penalty": ["l1", "l2"], 'solver': ['liblinear'],'class_weight': ['balanced']}

grid = {
    "C": np.logspace(-4, 4, 10),  # Adjust the number of values as needed, "multinominal LR" can only be performed with penalty l2
    "penalty": ["l2"],
    "solver": ["lbfgs"],
    "class_weight": ["balanced"],
    'max_iter': [1000, 1500, 2000]}
logreg = LogisticRegression(multi_class='multinomial', random_state=42)


# 05 Perform GridSearchCV

logreg_cv = GridSearchCV(logreg, grid, cv=10)
logreg_cv.fit(X_train_scaled, y_train_encoded)
print()
print("Tuned hyperparameters (best parameters):", logreg_cv.best_params_)
print("Best accuracy:", logreg_cv.best_score_)
print()


# 06 Train final model with best parameters

logreg_final = LogisticRegression(**logreg_cv.best_params_)
logreg_final.fit(X_train_scaled, y_train_encoded)


# 07 Extract the feature importances

feature_importances = abs(logreg_final.coef_[0])
feature_names = X_train.columns
features_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
features_df = features_df.sort_values(by='Importance', ascending=False) # Sort features by importance
print("Top 10 important features:")
top_features = features_df.head(10)
print(top_features)
print()

# 07.1 Plot feature importance
plt.figure(figsize=(20, 5))
plt.barh(top_features['Feature'], top_features['Importance'], color='mediumturquoise')
plt.xlabel('Importance (logistic regression coefficients)', fontsize=18) 
plt.ylabel("Feature Names", fontsize=18)
plt.title('Feature Importance - Logistic Regression', fontsize=24)  
plt.gca().invert_yaxis()  
plt.xticks(fontsize=16)
plt.yticks(fontsize=16) 
plt.tight_layout()  
plt.savefig("./Output/Visualisations/03_4 Logistic Regression/Importance barplot - LR.png", bbox_inches="tight")


# 08 Make predictions on the test data

y_pred = logreg_final.predict(X_test_scaled)


# 09 Create a confusion matrix

labels = label_encoder.classes_ # Adjust these to the actual encoded labels if needed
conf_matrix = confusion_matrix(y_test_encoded, y_pred)
print('Confusion Matrix:')
print(conf_matrix)


# 10 Create classification report

def report_to_latex(y_test, y_pred, relative_path, data_set, name):
    # Report as dictionary
    report = classification_report(
                y_test,
                y_pred,
                zero_division=0,
                target_names=label_encoder.classes_,
                output_dict=True)
    
    report.update({"accuracy": {"precision": None, "recall": None, "f1-score": report["accuracy"], "support": report['macro avg']['support']}})

    # Convert dictionary to DataFrame
    df_clsf_report = pd.DataFrame.from_dict(report, orient='index')
    df_clsf_report = df_clsf_report.round({'precision': 2, 'recall': 2, 'f1-score': 2, 'support': 0})

    title_clsfr = f"Classification Report - {name} ({data_set})"
    df_clsf_report.to_csv(str('./Output/Data/03 Summary results/' + title_clsfr + '.csv'), sep=',', index=True, encoding='utf-8')


# 11 Function to evaluate the performance 

def performance_evaluation(
        model_name,
        data_set,
        relative_path,
        y_test,
        y_pred,
        X_test,
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

    ## Save classification report to LaTeX table
    report_to_latex(y_test, y_pred, relative_path, data_set, model_name)

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
                'Model': 'LogReg',
                'FPR': fpr['macro-average'],
                'TPR': tpr['macro-average'],
                'ROC': roc_auc['macro-average']}
        )

        macro_plot.to_csv("./Output/Data/03 Summary Results/LogReg ROC.csv", sep=',', index=False, encoding='utf-8')

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
        df_roc.columns = ['LogReg']
        # Change model name to correct model
        df_roc.to_csv(str('./Output/Data/03 Summary results/ROC AUC Summary - LogReg.csv'), sep=',', index=True, encoding='utf-8')

print()


# 12 Call function performance_evaluation

performance_evaluation(model_name = "LogReg", data_set = "Testing Set", relative_path = "./Output/Visualisations/03_4 Logistic Regression/", y_test = y_test_encoded, y_pred= y_pred, X_test = X_test_scaled, model = logreg_final)
print()

y_pred_train = logreg_final.predict(X_train_scaled)
performance_evaluation(model_name = "LogReg", data_set = "Training Set", relative_path = "./Output/Visualisations/03_4 Logistic Regression/", y_test = y_train_encoded, y_pred= y_pred_train, X_test = X_train_scaled, model = logreg_final)
