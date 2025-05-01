import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



data = pd.read_csv("/content/test_semevalA2_280.csv")

# Replace 'A', 'B', 'C' with stance labels
label_mapping = {'A': 'AGAINST', 'B': 'FAVOR', 'C': 'NONE'}
data['Final Judgement'] = data['Final Judgement'].map(label_mapping)

# True labels and predictions
y_true = data['Stance']
y_pred = data['Final Judgement']

print(data.columns)
print(data["Target"].value_counts())

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, zero_division=0))


# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

# Precision (Macro, Micro)
precision_macro = precision_score(y_true, y_pred, average='macro')
precision_micro = precision_score(y_true, y_pred, average='micro')
print(f"Precision (Macro): {precision_macro:.4f}")
print(f"Precision (Micro): {precision_micro:.4f}")

# Recall (Macro, Micro)
recall_macro = recall_score(y_true, y_pred, zero_division=0, average='macro')
recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
print(f"Recall (Macro): {recall_macro:.4f}")
print(f"Recall (Micro): {recall_micro:.4f}")

# F1-Score (Macro, Micro)
f1_macro = f1_score(y_true, y_pred, average='macro')
f1_micro = f1_score(y_true, y_pred, average='micro')
print(f"F1 Score (Macro): {f1_macro:.4f}")
print(f"F1 Score (Micro): {f1_micro:.4f}")


# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=["FAVOR", "AGAINST", "NONE"])
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["FAVOR", "AGAINST", "NONE"],
            yticklabels=["FAVOR", "AGAINST", "NONE"])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()