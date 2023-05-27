import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from amrlib.evaluate.smatch_enhanced import compute_smatch, get_entries
from sklearn.metrics import roc_curve, roc_auc_score

#~ Penelitian Putra (2022)
tn, fp, fn, tp = 9, 10, 3, 28
# tn, fp, fn, tp = 10, 9, 2, 29
precision = tp / (tp+fp)
recall = tp / (tp+fn)
f1 = 2*precision*recall/(precision+recall)
accuracy = (tp+tn) / (tn+fp+fn+tp)

print(accuracy, precision, recall, f1)

# tn, tp = tp, tn
# fn, fp = fp, fn
# precision = tp / (tp+fp)
# recall = tp / (tp+fn)
# f1 = 2*precision*recall/(precision+recall)
# accuracy = (tp+tn) / (tn+fp+fn+tp)

# print(accuracy, precision, recall, f1)

# 0.74 0.7368421052631579 0.9032258064516129 0.8115942028985507
# 0.74 0.75 0.47368421052631576 0.5806451612903226

# >>> (0.8115942028985507+0.5806451612903226)/2
# 0.6961196820944366 # macro f1
# >>> (0.8115942028985507*31+0.5806451612903226*19)/50
# 0.723833567087424 # weighted f1
