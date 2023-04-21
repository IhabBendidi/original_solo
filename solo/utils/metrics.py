# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Dict, List, Sequence
from numpy import average
from sklearn.metrics import confusion_matrix
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix



from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *

#number_of_classes = 100#0
def eval_step(engine, batch):
    return batch

def accuracy_at_k(
    outputs: torch.Tensor, targets: torch.Tensor, top_k: Sequence[int] = (1, 5),number_of_classes : int = 10
) -> Sequence[int]:
    """Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        outputs (torch.Tensor): output of a classifier (logits or probabilities).
        targets (torch.Tensor): ground truth labels.
        top_k (Sequence[int], optional): sequence of top k values to compute the accuracy over.
            Defaults to (1, 5).

    Returns:
        Sequence[int]:  accuracies at the desired k.
    """
    #print(top_k) # (1, 5)
    default_evaluator = Engine(eval_step)



    

    with torch.no_grad():
        maxk = max(top_k)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        _,pred_copy  = outputs.topk(number_of_classes, 1, True, True)
        #pred_copy = pred_copy.t()
        pred = pred.t()
        top_predictions = pred[0]

        one_hot_top_preds = F.one_hot(top_predictions, num_classes=number_of_classes)

        assert top_predictions.get_device() == targets.get_device()
        device = top_predictions.get_device()


        metric_precision = ConfusionMatrix(num_classes=number_of_classes,average=None,device=device)#, output_transform=binary_one_hot_output_transform)
        metric_precision.attach(default_evaluator, 'cm_precision')

        #per_class_accuracy = confusion_matrix(targets.cpu(),top_predictions.cpu(), normalize="true").diagonal()

        #per_class_accuracy = [torch.Tensor(x.astype('float')).to(device) for x in per_class_accuracy]
        #per_class_accuracy = torch.Tensor(np.array(per_class_accuracy)).to(device)
        #print(top_predictions.shape)
        #print(pred_copy.shape)

        # make dict containing occurences numbers for each value in targets
        occurences = {}
        #print(targets)
        for i in range(len(targets)):
            if targets[i] in occurences:
                occurences[targets[i].cpu().item()] += 1
            else:
                occurences[targets[i].cpu().item()] = 1

        # make a list with values ranging from 0 to 999
        ref_values = [x for x in range(0,number_of_classes)]
        #print("occurences",occurences)

        
        # merge ref values with keys of occurences while replacing nan values with 0
        ref_occurences = [occurences.get(x,0) for x in ref_values]

        



        



        state = default_evaluator.run([[one_hot_top_preds, targets]])
        matrix = state.metrics['cm_precision']
        #print("matrix",matrix)
        #exit()
        precision = []
        recall = []
        pred_count = []
        # divide each value in the diagonal of matrix on the sum of all values in the column
        for i in range(number_of_classes):
            pred_count.append(matrix[i][i])
            recall.append(matrix[:,i].sum())
            precision.append(matrix[i,:].sum())
            """
            if matrix[i,:].sum() > 0.0 and matrix[:,i].sum() > 0.0 :
                precision.append(matrix[i][i] / matrix[i,:].sum())
                recall.append(matrix[i][i] / matrix[:,i].sum())
            elif matrix[i,:].sum() > 0.0 and matrix[:,i].sum() == 0.0:
                precision.append(matrix[i][i] / matrix[i,:].sum())
                recall.append(0.0)
            elif matrix[i,:].sum() == 0.0 and matrix[:,i].sum() > 0.0:
                precision.append(0.0)
                recall.append(matrix[i][i] / matrix[:,i].sum())
            else :
                precision.append(100.0)
                recall.append(100.0)
            """

        precision = torch.Tensor(precision).to(device)
        recall = torch.Tensor(recall).to(device)
        pred_count = torch.Tensor(pred_count).to(device)

        f_measure = [pred_count,precision,recall]
        
        """        

        for i in range(number_of_classes):
            if matrix[i,:].sum() > 0.0 :
                precision.append(matrix[i][i] / matrix[i,:].sum())
            else:
                precision.append(10.0)
            if matrix[:,i].sum() > 0.0 :
                recall.append(matrix[i][i] / matrix[:,i].sum())
            else:
                recall.append(10.0)
        """
        # precision to torch tensor on device
        
        #recall = torch.nan_to_num(recall, nan=1000.0)
        #precision = torch.nan_to_num(precision, nan=1000.0)
        #f_measure = (2 * precision * recall) / (precision + recall)
        #f_measure = torch.nan_to_num(f_measure, nan=0.0)
        #f_measure = precision
        del metric_precision
        del default_evaluator
        """
        print("ignite_f1",f_measure*100.0)
        print("ignite_precision",precision*100.0)
        print("ignite_recall",recall*100.0)
        print("sklearn_precision",per_class_accuracy*100.0)
        exit()
        """

        per_class_accuracy = confusion_matrix(targets.cpu(),top_predictions.cpu(), normalize="true").diagonal()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        

        res = [f_measure]
        res.append(ref_occurences)
        
        for k in top_k:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def weighted_mean(outputs: List[Dict], key: str, batch_size_key: str) -> float:
    """Computes the mean of the values of a key weighted by the batch size.
    Args:
        outputs (List[Dict]): list of dicts containing the outputs of a validation step.
        key (str): key of the metric of interest.
        batch_size_key (str): key of batch size values.
    Returns:
        float: weighted mean of the values of a key
    """

    value = 0
    n = 0
    for out in outputs:
        value += out[batch_size_key] * out[key]
        n += out[batch_size_key]
    value = value / n
    return value.squeeze(0)

def per_class_weighted_mean(outputs: List[Dict], key: str, batch_size_key: str,class_occurences_key : str) -> float:
    """Computes the mean of the values of a key weighted by the batch size.
    Args:
        outputs (List[Dict]): list of dicts containing the outputs of a validation step.
        key (str): key of the metric of interest.
        batch_size_key (str): key of batch size values.
    Returns:
        float: weighted mean of the values of a key
    """
    value = []
    #batch_sizes = []
    recalls = []
    precisions = []
    #n = len(outputs) # number of batches
    #print(outputs) 
    # [{'batch_size': 256, 'val_loss': tensor(7.3892, device='cuda:0'), 
    #'val_acc1': tensor([0.], device='cuda:0'), 'val_acc5': tensor([0.], device='cuda:0'), 
    #'val_class_acc': tensor([0.0000e+00, 1.7794e+01, ..., 1.0000e+05], device='cuda:0')}, 
    #    {'batch_size': 256, 'val_loss': tensor(6.8371, device='cuda:0'), 'val_acc1': tensor([0.], device='cuda:0'),
    #     'val_acc5': tensor([0.], device='cuda:0'), 'val_class_acc': tensor([100000.,  ...,100000.],
    #   device='cuda:0')}]



    

    classes_results = []
    #n = []
    for out in outputs:
        pred_count,precision,recall = out[key]
        precisions.append(precision)
        recalls.append(recall) #(nb of batches, nb of classes)
        value.append(pred_count)
        #batch_sizes.append(out[batch_size_key])
        #n.append(out[class_occurences_key])#(nb of batches, nb of classes)

    #list of tensors to tensor
    value = torch.stack(value)
    recalls = torch.stack(recalls)
    precisions = torch.stack(precisions)
    #print(n)
    #n = torch.Tensor(n)
    """
      
    accs = []
    # sum all values in all axises in value 
    for i in range(value.shape[0]):
        m = value[i].sum()
        l = precisions[i].sum()
        acc = m/l
        accs.append(acc)
    # get average of accs
    accs = torch.stack(accs)
    accs = accs.mean()
    print("mean_accs",accs)
    """

 
    
    

    

    


    value = torch.swapaxes(value,0,1) #(nb of classes, nb of batches)
    recalls = torch.swapaxes(recalls,0,1)
    precisions = torch.swapaxes(precisions,0,1)
    for i in range(len(value))  : 
        # do sum of values in v 

        # catch division by zero exception
        try:
            precision = value[i].sum() / precisions[i].sum()
            recall = value[i].sum() / recalls[i].sum()
            f_measure = (2 * precision * recall) / (precision + recall)
            #f_measure = precision
            f_measure = torch.nan_to_num(f_measure, nan=0.0)
            classes_results.append(f_measure*100.0)
        except ZeroDivisionError:
            classes_results.append(0.0)
    """
        

    batch_count = [0 for x in value[0]]
    # i is the variable denoting what class we are on, and j is denoting what batch we are on
    for i in range(len(value[0])):
        sum_value = 0
        count_all_class_elms = 0
        for j in range(len(value)):
            #if value[j][i] <= 100.0:
            #n[i] += batch_sizes[j]
            batch_count[i] += 1
            sum_value += value[j][i] * n[j][i]
            count_all_class_elms += n[j][i]
        if count_all_class_elms > 0:
            classes_results.append(sum_value/count_all_class_elms)
        else :
            classes_results.append(0.0)
    """
    return classes_results



def get_confusion_matrix(outs):
    all_targets = []
    all_preds = []

    for batch in outs:
        targets = batch['val_targets'].cpu().numpy()
        logits = batch['val_logits'].cpu().numpy()
        preds = np.argmax(logits, axis=1)

        all_targets.extend(targets)
        all_preds.extend(preds)

    cm = confusion_matrix(all_targets, all_preds)
    n_classes = len(np.unique(all_targets))

    return cm  ,n_classes  

