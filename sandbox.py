import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, recall_score, precision_recall_fscore_support
from amrlib.evaluate.smatch_enhanced import compute_smatch, get_entries
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = "/home/nafkhanzam/kode/nafkhanzam/thesis/ds/wrete"
SPLIT = "test"
AVG = "macro"

train_macro = 0.47058823529411764
dev_weighted = 0.4835164835164835

THRES = train_macro

print("SPLIT =", SPLIT)

entriesA = get_entries(f"{ROOT}/{SPLIT}-A.amr")
entriesB = get_entries(f"{ROOT}/{SPLIT}-B.amr")

assert len(entriesA) == len(entriesB)

results = []

for i in range(len(entriesA)):
    _, _, f1 = compute_smatch([entriesA[i]], [entriesB[i]])
    results.append(f1)

LABELS = ["NotEntail", "Entail_or_Paraphrase"]

df = pd.read_csv(f"{ROOT}/{SPLIT}.csv")
labels = df["label"].to_list()

#~ Prediction
pred = np.where(np.array(results) > THRES, LABELS[1], LABELS[0])

report = classification_report(labels, pred, digits=4)
print(report)

print(confusion_matrix(labels, pred, labels=LABELS))

#~ Find ROC
fpr, tpr, thresholds = roc_curve(labels, results, pos_label=LABELS[1])
scores = []
for thres in thresholds:
    y_pred = np.where(results > thres, LABELS[1], LABELS[0])
    f1 = f1_score(labels, y_pred, labels=LABELS, average=AVG)
    scores.append(f1)

scores = pd.concat([pd.Series(thresholds), pd.Series(fpr), pd.Series(tpr), pd.Series(scores)],
                        axis = 1)
scores.columns = ['Thresholds', 'FPR', 'TPR', 'Score']
scores.sort_values(by ='Score', ascending = False, inplace = True)
scores.reset_index(drop = True, inplace = True)
print(scores['Thresholds'].to_list()[0], scores['Score'].to_list()[0])
print(scores)
