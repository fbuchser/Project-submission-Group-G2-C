import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from imblearn.combine import SMOTETomek

#imports for performance evaluation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle

#Custom scorer for Tuning with RandomizedSearchCV
def define_scorer(y_true, y_pred):
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    precision_mutated = precision_score(y_true, y_pred, labels=['Mutated'], average='macro', zero_division=0)
    recall_mutated = recall_score(y_true, y_pred, labels=['Mutated'], average='macro', zero_division=0)
    f1_mutated = f1_score(y_true, y_pred, labels=['Mutated'], average='macro', zero_division=0)
    return (precision_macro + recall_macro + f1_macro + precision_mutated + recall_mutated + f1_mutated)/6
custom_scorer = make_scorer(define_scorer, greater_is_better=True)


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


#Function for performance evaluation
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
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save classification report as DataFrame
    save_classification_report(y_test, y_pred, relative_path, data_set, name = model_name)

    ## Plot confusion matrix
    fig, ax = plt.subplots()
    title_cm = f"Confusion Matrix - {model_name} ({data_set})"
    label_list = ['Mutated', 'NOS/NEC', 'Wildtype']
    cm = confusion_matrix(y_test, y_pred, labels=label_list)
    sns.heatmap(
        cm,
        annot=True,
        annot_kws={"size": 12},
        fmt='d',
        cmap='Blues',
        ax=ax,
        xticklabels=label_list,
        yticklabels=label_list
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
                'Model': 'SVM',
                'FPR': fpr['macro-average'],
                'TPR': tpr['macro-average'],
                'ROC': roc_auc['macro-average']}
        )

        macro_plot.to_csv("./Output/Data/03 Summary Results/SVM ROC.csv", sep=',', index=False, encoding='utf-8')

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
                label=f"ROC curve of class {classes[i]} (area = {roc_auc[i]:.2f})"
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
            print(f"ROC AUC of class {classes[i]}: {roc_auc[i]:.2f}")
            df_roc[f"ROC AUC of class {classes[i]}"] = [round(roc_auc[i], 2)]
        df_roc = df_roc.T
        df_roc.columns = ['SVM']
        # Change model name to correct model
        df_roc.to_csv(str('./Output/Data/03 Summary results/ROC AUC Summary - SVM.csv'), sep=',', index=True, encoding='utf-8')


print('################ SVM ####################')

X_train = pd.read_csv("./Output/Data/02 13-Preprocessed Data/X_train_final.csv")
X_test = pd.read_csv("./Output/Data/02 13-Preprocessed Data/X_test_final.csv")
y_train = pd.read_csv("./Output/Data/02 13-Preprocessed Data/y_train_final.csv")
y_test = pd.read_csv("./Output/Data/02 13-Preprocessed Data/y_test_final.csv")

print('Scaling')
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train = y_train.to_numpy()

#Tuning with the original imbalanced data (class_weight='balanced') and with the custom scorer, 3 different kernels
print('sigmoid')
param_grid1 = {'C': [0.1, 1, 10],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['sigmoid']}
grid1 = RandomizedSearchCV(svm.SVC(probability=True, class_weight='balanced', decision_function_shape='ovr'), param_grid1, scoring=custom_scorer, n_iter=15, refit = True, verbose = 3, random_state=45, n_jobs=-1)
grid1.fit(X_train_scaled, y_train.ravel())
pred1 = grid1.predict(X_train_scaled)

print('poly')
param_grid2 = {'C': [0.1, 1, 10],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['poly']}
grid2 = RandomizedSearchCV(svm.SVC(probability=True, class_weight='balanced', decision_function_shape='ovr'), param_grid2, scoring=custom_scorer, n_iter=15, refit = True, verbose = 3, random_state=45, n_jobs=-1)
grid2.fit(X_train_scaled, y_train.ravel())
pred2 = grid2.predict(X_train_scaled)

print('linear')
param_grid3 = {'C': [0.1, 1, 10],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['linear']}
grid3 = RandomizedSearchCV(svm.SVC(probability=True, class_weight='balanced', decision_function_shape='ovr'), param_grid3, scoring=custom_scorer, n_iter=15, refit = True, verbose = 3, random_state=45, n_jobs=-1)
grid3.fit(X_train_scaled, y_train.ravel())
pred3 = grid3.predict(X_train_scaled)

print('Classification reports training set')
print("Classification report sigmoid")
print(classification_report(y_train, pred1, zero_division=0))

print("Classification report poly")
print(classification_report(y_train, pred2, zero_division=0))

print("Classification report linear")
print(classification_report(y_train, pred3, zero_division=0))

print('Model 1 with sigmoid kernel seems to have the lowest risk of overfitting when looking at the training set')

#Sigmoid kernel, tuning with default scorer instead of custom scorer
print('Tuning with default scorer instead of custom scorer')
param_grid4 = {'C': [0.1, 1, 10],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['sigmoid']}
grid4 = RandomizedSearchCV(svm.SVC(probability=True, class_weight='balanced', decision_function_shape='ovr'), param_grid4, n_iter=15, refit = True, verbose = 3, random_state=45, n_jobs=-1)
grid4.fit(X_train_scaled, y_train.ravel())
pred4 = grid4.predict(X_train_scaled)

print("Classification report Training set with default scorer -> lower recall for classes mutated and NOS/NEC compared to the custom scorer")
print(classification_report(y_train, pred4, zero_division=0))


#Sigmoid kernel, custom scorer, SMOTETomek oversampling before tuning (balanced data instead of the original imbalanced data)
print('Oversampling')
smt = SMOTETomek(random_state=42)
X_train_balanced, y_train_balanced = smt.fit_resample(X_train, y_train)

print('Scaling')
scaler2 = StandardScaler()
X_train_scaled2 = scaler2.fit_transform(X_train_balanced)
X_test_scaled2 = scaler2.transform(X_test)


#Tuning with SMOTE oversampled data
print('Tuning with balanced data instead of the original imbalanced data')
param_grid5 = {'C': [0.1, 1, 10],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['sigmoid']}
grid5 = RandomizedSearchCV(svm.SVC(probability=True, decision_function_shape='ovr'), param_grid5, scoring=custom_scorer, n_iter=15, refit = True, verbose = 3, random_state=45, n_jobs=-1)
grid5.fit(X_train_scaled2, y_train_balanced.ravel())
pred5 = grid5.predict(X_train_scaled2)

print(f"Classification Report Training Set (SMOTE oversampling instead of using the original imbalanced data) -> very high performance values on training set, more risk of overfitting")
print(classification_report(y_train_balanced, pred5, zero_division=0))

print("Model 1 that was trained on imbalanced data (class_weight='balanced') with sigmoid kernel and custom scorer seems to be best (lowest risk of overfitting, high recall for the minority classes mutated and NOS/NEC)")
print('Best model:')
print(grid1.best_estimator_)
#performance on training set
performance_evaluation(
    # model_name
    "SVM",
    # data_set
    "Training Set",
    # relative_path
    "./Output/Visualisations/03_3 Support Vector Machine/",
    # y_test
    y_train,
    # y_pred 
    pred1,
    # X_test
    X_train_scaled,
    # model
    grid1
    )
#Performance on testing set:
pred1_test = grid1.predict(X_test_scaled)
performance_evaluation(
    # model_name
    "SVM",
    # data_set
    "Testing Set",
    # relative_path
    "./Output/Visualisations/03_3 Support Vector Machine/",
    # y_test
    y_test,
    # y_pred 
    pred1_test,
    # X_test
    X_test_scaled,
    # model
    grid1
    )
