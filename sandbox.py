import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from amrlib.evaluate.smatch_enhanced import compute_smatch, get_entries
from sklearn.metrics import roc_curve, roc_auc_score

entries = get_entries(f"/home/nafkhanzam/kode/nafkhanzam/thesis/AMRBART-v3/outputs/infer-wrete/val_outputs/test_generated_predictions_0.txt")

ROWS = 50

results = []

for i in range(ROWS):
    _, _, f1 = compute_smatch([entries[i]], [entries[i+ROWS]])
    results.append(f1)

LABELS = ["NotEntail", "Entail_or_Paraphrase"]
# max weighted f1: 0.476190
# max acc: 0.463415
# max macro f1: 0.464286
pred = [LABELS[1] if x > 0.393443 else LABELS[0] for x in results]

df = pd.read_csv(f"/home/nafkhanzam/kode/nafkhanzam/thesis/AMRBART-v3/datasets/wrete/WReTE-dev.csv")
label = df["label"].to_list()

report = classification_report(label, pred, digits=3)
print(report)

print(confusion_matrix(label, pred, labels=LABELS))

#~ Find ROC
# label_num = [1 if x == LABELS[1] else 0 for x in label]

# fpr,tpr,thresholds = roc_curve(label_num, results)
# scores = []
# for thres in thresholds:
#     y_pred = np.where(results > thres, 1, 0)
#     scores.append(f1_score(label_num, y_pred, average="weighted"))

# scores = pd.concat([pd.Series(thresholds), pd.Series(fpr), pd.Series(tpr), pd.Series(scores)],
#                         axis = 1)
# scores.columns = ['Thresholds', 'FPR', 'TPR', 'Score']
# scores.sort_values(by ='Score', ascending = False, inplace = True)
# scores.reset_index(drop = True,inplace = True)
# print(scores)
