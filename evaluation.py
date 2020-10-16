import numpy as np
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix, auc, roc_curve, precision_recall_curve

# predict probabilities for test set
# yhat_probs = model.predict(X_test, batch_size=1024)

# predict crisp classes for test set
# yhat_classes = np.argmax(yhat_probs, axis=1)
# yhat_probs = yhat_probs[:,1]

def eval(y_classes, yhat_classes, yhat_probs):
    # y_classes = np.argmax(y_test, axis=1)
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_classes, yhat_classes)
    # print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    preci = precision_score(y_classes, yhat_classes)
    # print('Precision: %f' % preci)
    # recall: tp / (tp + fn)
    recal = recall_score(y_classes, yhat_classes)
    # print('Recall: %f' % recal)
    # f1: 2 tp / (2 tp + fp + fn)
    mcc = matthews_corrcoef(y_classes,yhat_classes)
    print('mcc: %f' % mcc)
    f1 = f1_score(y_classes, yhat_classes)
    print('F1 score: %f' % f1)
    # kappa
    kappa = cohen_kappa_score(y_classes, yhat_classes)
    print('Cohens kappa: %f' % kappa)
    # ROC AUC
    fpr, tpr, thresholds = roc_curve(y_classes, yhat_probs, pos_label=1)
    roc_auc = auc(fpr, tpr)
    # print(thresholds)
    print("ROC AUC: ", roc_auc)
    # PR AUC
    precision, recall, thresholds = precision_recall_curve(y_classes, yhat_probs, pos_label=1)
    pr_auc = auc(recall, precision)
    # print(thresholds)
    print("PR AUC: ", pr_auc)
    # confusion matrix
    matrix = confusion_matrix(y_classes, yhat_classes)
    print(matrix)

    return accuracy, preci, recal, mcc, f1, kappa,roc_auc, pr_auc, matrix