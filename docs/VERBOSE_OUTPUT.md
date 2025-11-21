# Verbose Output Guide

All classifier training functions now include detailed progress information. Here's what you'll see for each classifier:

## Linear Classifier (Logistic Regression)

**Output includes:**
```
Training linear classifier (Logistic Regression)...
Training set size: XXXXX samples, XXXX features
Starting GridSearchCV with 6 C values and 5-fold CV...
Total fits: 30
Fitting 5 folds for each of 6 candidates, totalling 30 fits
[CV 1/5] END ........................C=0.001;, score=0.XXXX total time=   X.Xs
[CV 2/5] END ........................C=0.001;, score=0.XXXX total time=   X.Xs
...
[CV 5/5] END ............................C=100;, score=0.XXXX total time=   X.Xs

Best parameters: {'C': 1.0}
Validation accuracy: 0.XXXX
Validation F1 score: 0.XXXX
Validation MCC: 0.XXXX
Linear classifier training time: XXXX.XX seconds
```

**Progress indicators:**
- Shows total number of fits upfront (30 = 6 C values × 5 folds)
- Each cross-validation fold prints its score and time
- Total of 30 progress lines during training
- Summary at the end with best parameters and validation metrics

---

## SVM Classifier

### Simple Mode (Default - Minimal Memory)

**Output includes:**
```
Training SVM classifier with linear kernel...
Training set size: XXXXX samples, XXXX features
Running in simple mode (no hyperparameter search to save memory)...
Using fixed C=1.0
[LibSVM progress output during training...]
Validation accuracy: 0.XXXX
Validation F1 score: 0.XXXX
Validation MCC: 0.XXXX
SVM training time: XXXX.XX seconds
```

**Progress indicators:**
- No GridSearchCV (single training run)
- LibSVM shows internal progress
- Much faster and lower memory usage
- Good for testing or when memory is limited

### Full Mode (Hyperparameter Search)

**Output includes:**
```
Training SVM classifier with linear kernel...
Training set size: XXXXX samples, XXXX features
Starting GridSearchCV with 4 C values and 3-fold CV...
Total fits: 12
Fitting 3 folds for each of 4 candidates, totalling 12 fits
[CV 1/3] END .........................C=0.01;, score=0.XXXX total time=  XX.Xs
[CV 2/3] END .........................C=0.01;, score=0.XXXX total time=  XX.Xs
...
[CV 3/3] END ............................C=10;, score=0.XXXX total time=  XX.Xs

Best parameters: {'C': 1.0}
Validation accuracy: 0.XXXX
Validation F1 score: 0.XXXX
Validation MCC: 0.XXXX
SVM training time: XXXX.XX seconds
```

**Progress indicators:**
- Shows total number of fits upfront (12 = 4 C values × 3 folds)
- Each cross-validation fold prints its score and time
- Total of 12 progress lines during training
- Summary at the end with best parameters and validation metrics

**Note:**
- Full mode is MUCH slower than Linear classifier (~45-50s per fold vs ~1s per fold)
- Requires significantly more memory (256GB+)
- Remove `--svm_simple_mode` flag to enable

---

## Neural Network Classifier

**Output includes:**
```
Training 3-layer neural network classifier...
Training set size: XXXXX samples, XXXX features
Device: cuda
Performing hyperparameter search with 3 configurations...

============================================================
Configuration 1/3:
  Architecture: 256 -> 128 -> 1
  Dropout: 0.3
  Learning rate: 0.001
  Epochs: 100
============================================================
Epoch [20/100], Loss: 0.XXXX
Epoch [40/100], Loss: 0.XXXX
Epoch [60/100], Loss: 0.XXXX
Epoch [80/100], Loss: 0.XXXX
Epoch [100/100], Loss: 0.XXXX
Validation accuracy: 0.XXXX
Validation F1 score: 0.XXXX
Validation MCC: 0.XXXX
New best model found with F1: 0.XXXX

============================================================
Configuration 2/3:
  Architecture: 512 -> 256 -> 1
  Dropout: 0.3
  Learning rate: 0.001
  Epochs: 100
============================================================
Epoch [20/100], Loss: 0.XXXX
Epoch [40/100], Loss: 0.XXXX
Epoch [60/100], Loss: 0.XXXX
Epoch [80/100], Loss: 0.XXXX
Epoch [100/100], Loss: 0.XXXX
Validation accuracy: 0.XXXX
Validation F1 score: 0.XXXX
Validation MCC: 0.XXXX
New best model found with F1: 0.XXXX

============================================================
Configuration 3/3:
  Architecture: 256 -> 128 -> 1
  Dropout: 0.5
  Learning rate: 0.0001
  Epochs: 150
============================================================
Epoch [20/150], Loss: 0.XXXX
Epoch [40/150], Loss: 0.XXXX
Epoch [60/150], Loss: 0.XXXX
Epoch [80/150], Loss: 0.XXXX
Epoch [100/150], Loss: 0.XXXX
Epoch [120/150], Loss: 0.XXXX
Epoch [140/150], Loss: 0.XXXX
Epoch [150/150], Loss: 0.XXXX
Validation accuracy: 0.XXXX
Validation F1 score: 0.XXXX
Validation MCC: 0.XXXX

============================================================
Best validation F1 score: 0.XXXX
============================================================
Neural Network training time: XXX.XX seconds
```

**Progress indicators:**
- Shows configuration details before each training run
- Loss printed every 20 epochs
- Validation metrics after each configuration
- Indicates when a new best model is found
- Final summary showing best F1 score across all configurations

---

## Summary

| Classifier    | Total Progress Lines | Updates Frequency | Approx Time  | Memory  |
|---------------|---------------------|-------------------|--------------|---------|
| Linear        | 30 (6 × 5 CV)       | Every ~1s         | ~30 min      | 32GB    |
| SVM (Simple)  | Continuous          | LibSVM output     | ~20-30 min   | 32GB    |
| SVM (Full)    | 12 (4 × 3 CV)       | Every ~45s        | 2-12 hours   | 256GB+  |
| NN            | ~15-20 per config   | Every 20 epochs   | ~5-10 min    | 32GB    |

All classifiers now provide:
- Dataset size information at start
- Progress during training
- Best parameters/configuration
- Validation metrics
- Total training time
