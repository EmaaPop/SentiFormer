import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
__all__ = ['MetricsTop']


class MetricsTop():
    def __init__(self):
        self.metrics_dict = {
            'FI': self.__eval_FI_regression,
            'Twitter_LDL': self.__eval_Twitter_LDL_regression,
            'Artphoto': self.__eval_Artphoto_regression
        }

    def __multiclass_acc(self, y_pred, y_true):
        return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))

    def __eval_Twitter_LDL_regression(self, y_pred, y_true, exclude_zero=False):
        test_preds = y_pred.cpu().detach()
        test_truth = y_true.cpu().detach()
        y_pred_labels = torch.argmax(test_preds, dim=1)  
        y_true_labels = torch.argmax(test_truth, dim=1)  

        y_pred_labels = y_pred_labels.numpy()
        y_true_labels = y_true_labels.numpy()

        accuracy = accuracy_score(y_true_labels, y_pred_labels)
        precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
        recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
        f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')

        roc_auc = roc_auc_score(test_truth.numpy(), test_preds.numpy(), multi_class='ovr')
        eval_results = {
            'Accuracy':accuracy,
            'Precision':precision,
            'Recall':recall,
            'F1-score':f1,
            'AUC-ROC':roc_auc
        }
        return eval_results


    def __eval_FI_regression(self, y_pred, y_true):
        return self.__eval_Twitter_LDL_regression(y_pred, y_true)

    def __eval_Artphoto_regression(self, y_pred, y_true):
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()
        test_preds = np.clip(test_preds, a_min=-1., a_max=1.)
        test_truth = np.clip(test_truth, a_min=-1., a_max=1.)

        # two classes{[-1.0, 0.0], (0.0, 1.0]}
        ms_2 = [-1.01, 0.0, 1.01]
        test_preds_a2 = test_preds.copy()
        test_truth_a2 = test_truth.copy()
        for i in range(2):
            test_preds_a2[np.logical_and(test_preds > ms_2[i], test_preds <= ms_2[i+1])] = i
        for i in range(2):
            test_truth_a2[np.logical_and(test_truth > ms_2[i], test_truth <= ms_2[i+1])] = i

        # three classes{[-1.0, -0.1], (-0.1, 0.1], (0.1, 1.0]}
        ms_3 = [-1.01, -0.1, 0.1, 1.01]
        test_preds_a3 = test_preds.copy()
        test_truth_a3 = test_truth.copy()
        for i in range(3):
            test_preds_a3[np.logical_and(test_preds > ms_3[i], test_preds <= ms_3[i+1])] = i
        for i in range(3):
            test_truth_a3[np.logical_and(test_truth > ms_3[i], test_truth <= ms_3[i+1])] = i
        
        # five classes{[-1.0, -0.7], (-0.7, -0.1], (-0.1, 0.1], (0.1, 0.7], (0.7, 1.0]}
        ms_5 = [-1.01, -0.7, -0.1, 0.1, 0.7, 1.01]
        test_preds_a5 = test_preds.copy()
        test_truth_a5 = test_truth.copy()
        for i in range(5):
            test_preds_a5[np.logical_and(test_preds > ms_5[i], test_preds <= ms_5[i+1])] = i
        for i in range(5):
            test_truth_a5[np.logical_and(test_truth > ms_5[i], test_truth <= ms_5[i+1])] = i
 
        mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a2 = self.__multiclass_acc(test_preds_a2, test_truth_a2)
        mult_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        f_score = f1_score(test_preds_a2, test_truth_a2, average='weighted')

        eval_results = {
            "Mult_acc_2": mult_a2,
            "Mult_acc_3": mult_a3,
            "Mult_acc_5": mult_a5,
            "F1_score": f_score,
            "MAE": mae,
            "Corr": corr, # Correlation Coefficient
        }
        return eval_results
    
    def getMetics(self, datasetName):
        return self.metrics_dict[datasetName.upper()]
