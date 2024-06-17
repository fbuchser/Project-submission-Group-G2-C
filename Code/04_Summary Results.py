# Combine the results of the different models

### 01
# Packages
import pandas as pd
import matplotlib.pyplot as plt



### 02
# Table with all the metrics

# Load the classification report of all models
BRF_test = pd.read_csv("./Output/Data/03 Summary Results/Classification Report - BRF (Testing Set).csv", index_col=0)
XGBoost_test = pd.read_csv("./Output/Data/03 Summary Results/Classification Report - XGB (Testing Set).csv", index_col=0)
SVM_test = pd.read_csv("./Output/Data/03 Summary Results/Classification Report - SVM (Testing Set).csv", index_col=0)
LogReg_test = pd.read_csv("./Output/Data/03 Summary Results/Classification Report - LogReg (Testing Set).csv", index_col=0)

# Drop additional support columns
support_column = BRF_test['support'].astype(int)
BRF_test = BRF_test.drop(columns=['support'])
SVM_test = SVM_test.drop(columns=['support'])
LogReg_test = LogReg_test.drop(columns=['support'])
XGBoost_test = XGBoost_test.drop(columns=['support'])

# Merge them to one table & save this table
merged_cr = pd.concat([BRF_test, XGBoost_test, SVM_test, LogReg_test], axis=1, keys=['BRF', 'XGBoost', 'SVM', 'LogReg'])
merged_cr['Support'] = support_column

# Load the ROC AUC summary of all models
BRF_ROC = pd.read_csv("./Output/Data/03 Summary Results/ROC AUC Summary - BRF.csv", index_col=0)
XGBoost_ROC = pd.read_csv("./Output/Data/03 Summary Results/ROC AUC Summary - XGB.csv", index_col=0)
SVM_ROC = pd.read_csv("./Output/Data/03 Summary Results/ROC AUC Summary - SVM.csv", index_col=0)
LogReg_ROC = pd.read_csv("./Output/Data/03 Summary Results/ROC AUC Summary - LogReg.csv", index_col=0)

# Merge them to one table & save this table
merged_ROC = pd.concat([BRF_ROC, XGBoost_ROC, SVM_ROC, LogReg_ROC], axis=1)
merged_ROC.columns = ['BRF', 'XGBoost', 'SVM', 'LogReg']



### 03 
# Save all performance metrics in a latex table
def dataframe_to_latex(df, name):
    latex_table_clsfr = df.to_latex(
                            index=True,
                            multirow=True,
                            na_rep='',
                            header=True,
                            float_format="%.2f"
                            )
    
    with open('./Output/Visualisations/04 Summary results/' + name + '.txt', "w+") as f:
        f.write(latex_table_clsfr)
dataframe_to_latex(merged_cr, 'summary classification report_latex')
dataframe_to_latex(merged_ROC, 'summary ROC AUC_latex')



### 04
# Combined ROC curves
BFR_data = pd.read_csv("./Output/Data/03 Summary Results/BFR ROC.csv", header=0)
XGB_data = pd.read_csv("./Output/Data/03 Summary Results/XGB ROC.csv", header=0)
SVM_data = pd.read_csv("./Output/Data/03 Summary Results/SVM ROC.csv", header=0)
LogReg_data = pd.read_csv("./Output/Data/03 Summary Results/LogReg ROC.csv", header=0)


BFR_ROC = BFR_data['ROC'].unique()[0]
XGB_ROC = XGB_data['ROC'].unique()[0]
SVM_ROC = SVM_data['ROC'].unique()[0]
LogReg_ROC = LogReg_data['ROC'].unique()[0]

# Plot
fig, ax = plt.subplots(figsize=(7, 7))

plt.plot(
    BFR_data['FPR'],
    BFR_data['TPR'],
    label=f"BFR  (AUC = {BFR_ROC:.2f})",
    color='lightsalmon',
    linestyle='-',
    lw=2
    )

plt.plot(
    XGB_data['FPR'],
    XGB_data['TPR'],
    label=f"XGB  (AUC = {XGB_ROC:.2f})",
    color='yellowgreen',
    linestyle='-',
    lw=2
    )

plt.plot(
    SVM_data['FPR'],
    SVM_data['TPR'],
    label=f"SVM  (AUC = {SVM_ROC:.2f})",
    color='cornflowerblue',
    linestyle='-',
    lw=2
    )

plt.plot(
    LogReg_data['FPR'],
    LogReg_data['TPR'],
    label=f"LogReg  (AUC = {LogReg_ROC:.2f})",
    color='mediumorchid',
    linestyle='-',
    lw=2
    )

ax.plot([0, 1], [0, 1], 'k--', lw=2)
ax.set_xlim([0.0, 1.05])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Macro-average ROC curves of all Models')
ax.legend(loc="lower right")

plt.savefig('./Output/Visualisations/04 Summary results/Combined macro-average ROC curves.png', bbox_inches='tight')