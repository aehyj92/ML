
import pandas as pd
import argparse
import json
import numpy as np
import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sb
from tqdm import tqdm, trange
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Standardize the data
from sklearn.preprocessing import StandardScaler
# Dataset
from sklearn import datasets
# Model and performance evaluation
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support as score
# Hyperparameter tuning
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from hyperopt import tpe, STATUS_OK, Trials, hp, fmin, STATUS_OK, space_eval

# preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# model selection
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_validate, cross_val_score
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
import optuna

# models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
import warnings
import path
import time
warnings.filterwarnings('ignore')
from sklearn.ensemble import GradientBoostingClassifier
import time
import warnings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score # k-fold CV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# model selection
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_validate, cross_val_score
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
import optuna

from imblearn.over_sampling import SMOTE,ADASYN
from imblearn.over_sampling import BorderlineSMOTE

#결과 정리 함수
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time
import shap
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime
import matplotlib
import plotly.io as pio
import plotly.express as px

from numpy import sqrt
from numpy import argmax
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
import shap
import joblib
import argparse
import pandas as pd
import plotly.io as pio




def f_score(y_test, preds, beta=1):
    tp = np.sum((preds == 1) & (y_test == 1))
    fp = np.sum((preds == 1) & (y_test == 0))
    fn = np.sum((preds == 0) & (y_test == 1))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    fscore = (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall + 1e-8)
    
    return fscore




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate a binary classification model.')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the CSV file containing the data.')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to the output JSON file containing the results.')
    parser.add_argument('--model', type=str, required=True, choices=['xgb', 'rf','lgbm','gbm'],
                        help='Type of model to use: xgb (XGBoost) or rf (Random Forest).')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of cross-validation folds to use.')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state to use for reproducibility.')
    parser.add_argument('--num_iter', type=int, default=50,
                        help='Number of trees (iterations) to use for the model.')
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='Number of trees (estimators) to use for the model.')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate to use for the XGBoost model.')
    parser.add_argument('--max_depth', type=int, default=50,
                        help='Maximum depth of each tree in the model.')
#     parser.add_argument('--subsample', type=float, default=0.8,
#                         help='Subsample ratio of the training instances.')
#     parser.add_argument('--colsample_bytree', type=float, default=0.8,
#                         help='Subsample ratio of columns when constructing each tree.')
#     parser.add_argument('--min_child_weight', type=int, default=1,
#                         help='Minimum sum of instance weight (hessian) needed in a child.')
#     parser.add_argument('--gamma', type=float, default=0,
#                         help='Minimum loss reduction required to make a further partition on a leaf node of the tree.')
#     parser.add_argument('--scale_pos_weight', type=float, default=1,
#                         help='Ratio of the number of negative class to the positive class.')
    parser.add_argument('--eval_metric', type=str, default='log_loss',
                        help='Evaluation metric to use for XGBoost.')
    parser.add_argument('--early_stopping_rounds', type=int, default=10,
                        help='Number of early stopping rounds to use for XGBoost.')
    parser.add_argument('--verbose', type=bool, default=False,
                        help='Verbosity of the XGBoost output.')
    args = parser.parse_args()

    # Load the data

    data = pd.read_csv(args.data_path)
    
#     features=data[data.group_2=='Discovery'].iloc[:,3:].astype('float')
#     X_test=data[data.group_2=='Validation'].iloc[:,3:].astype('float')
#     label=data[data.group_2=='Discovery']['group_1'].astype('float')
#     y_test=data[data.group_2=='Validation']['group_1'].astype('float') 
    
    features = data.iloc[:,4:].astype('float')
    label = data.iloc[:,1].astype('float')
    
#     # Extract features and labels
#     features = data.drop('label', axis=1).values
#     label = data['label'].values

#     # Standardize the features
#     scaler = StandardScaler()
#     features = scaler.fit_transform(features)

    # Define the cross-validation folds
    timestamp = int(time.time())
    LABELS=['Control','Cancer']
    kf = RepeatedStratifiedKFold(n_splits=args.n_folds, random_state=args.random_state)  
    aucs = []
    accuracies = []
    recalls = []
    precisions = []
    f1_scores=[]
    n_iter = 0

    for train_idx, test_idx in kf.split(features, label):
#         X_train, X_test = features[train_idx], features[test_idx]
#         y_train, y_test = label[train_idx], label[test_idx]
        n_iter += 1
        print(f'--------------------{n_iter}번째 KFold-------------------')
        print(f'train_idx_len : {len(train_idx)} / test_idx_len : {len(test_idx)}')

#         label_train = label.iloc[train_idx]
#         label_test = label.iloc[test_idx]
        X_train, X_test = features.iloc[train_idx, :], features.iloc[test_idx, :]
        y_train, y_test = label.iloc[train_idx], label.iloc[test_idx]
        
        # Oversample the minority class using BorderlineSMOTE
        sm = BorderlineSMOTE(random_state=args.random_state)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        eval_set = [(X_train, y_train), (X_test, y_test)]

        # Train the model
        if args.model == 'xgb':
            model_name='XGBOOST'
            model = xgb.XGBClassifier(n_estimators=args.n_estimators,
                                      learning_rate=args.learning_rate,
                                      max_depth=args.max_depth,
#                                       subsample=args.subsample,
#                                       colsample_bytree=args.colsample_bytree,
#                                       min_child_weight=args.min_child_weight,
#                                       gamma=args.gamma,
#                                       scale_pos_weight=args.scale_pos_weight,
                                      eval_metric=args.eval_metric,
                                      early_stopping_rounds=args.early_stopping_rounds,

                                      random_state=args.random_state)
            model.fit(X_train, y_train, eval_set=eval_set,verbose=args.verbose)
        elif args.model == 'lgbm':
            model_name='LightGBM'
            model = LGBMClassifier(num_iterations=args.num_iter,
                                   n_estimators=1000,
                                   boosting_type='goss',
                                   max_depth=100,
                                   num_leaves=30,
                                   learning_rate=args.learning_rate,
                                   random_state=45,
                                   n_jobs=-1,
                                   boost_from_average=True)
            model.fit(X_train, y_train, eval_set=eval_set,verbose=args.verbose)
        elif args.model == 'rf':
            model_name='RandomForest'
            model = RandomForestClassifier(n_estimators=args.n_estimators,
                                            max_depth=args.max_depth,
                                            min_samples_split=2,
                                            min_samples_leaf=1,
                                            random_state=args.random_state)
            model.fit(X_train, y_train)
            
        elif args.model == 'gbm':
            model_name='GBM'
            model = GradientBoostingClassifier(max_depth=100
                    ,learning_rate=0.13001
                    ,random_state=45)       
            model.fit(X_train, y_train)

        else:
            raise ValueError('Invalid model type specified.')


        # Evaluate the model
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        #Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        J = tpr - fpr
        ix = argmax(J)
        best_thresh = thresholds[ix]
        sens, spec = tpr[ix], 1-fpr[ix]        
        auc = roc_auc_score(y_test, y_prob)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1_score=f_score(y_test, y_pred,beta=1)
        aucs.append(auc)
        accuracies.append(accuracy)
        recalls.append(recall)
        precisions.append(precision)
        f1_scores.append(f1_score)
 
        
        # plot the roc curve for the model
        plt.plot([0,1], [0,1], linestyle='--', markersize=0.01, color='black')
        plt.plot(fpr, tpr, marker='.', color='black', markersize=0.05, label=f"{model_name} AUC = %.4f" % roc_auc_score(y_test, y_prob))
        plt.scatter(fpr[ix], tpr[ix], marker='+', s=100, color='r', 
                label='Best threshold = %.4f, \nSensitivity = %.4f, \nSpecificity = %.4f' % (best_thresh, sens, spec))
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc=4)
        # Make folder
        os.makedirs(f'{n_iter}_fold', exist_ok=True)

        # Save image
        plt.savefig(f'{n_iter}_fold/roc_curve_{timestamp}.png')
        # show the plot
#         plt.show()        
        plt.clf() # clear the plot
    
        #Plot confusion matrix
        preds_1 = np.where(y_prob >= best_thresh, 1, 0)
        plt.figure(figsize=(4,3))
        sns.heatmap(confusion_matrix(y_test, preds_1),xticklabels=LABELS, yticklabels=LABELS, annot=True,fmt='d', cmap='Blues')
        plt.title('Confusion matrix', fontsize=13)
        plt.xlabel('Predict', fontsize=11)
        plt.ylabel('Label', fontsize=11)
        plt.tight_layout() # adjust the layout
        plt.savefig(f'{n_iter}_fold/confusion_matrix_{timestamp}.png')
#         plt.show() 
        plt.clf() # clear the plot


        # Plot top 20 feature importances as a horizontal bar chart
        pi = pd.DataFrame(model.feature_importances_, index=X_train.columns, columns=['importance']).sort_values(by='importance', ascending=False)
        fig, ax = plt.subplots(figsize=(18, 8))
        sns.barplot(data=pi.head(20), x='importance', y=pi.index[:20], ax=ax)
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        ax.set_title('Top 20 Feature Importances')
        plt.tight_layout()
        # Save the plot as an image
        plt.savefig(f'{n_iter}_fold/feature_importance_{timestamp}.png')
        plt.clf() # clear the plot
        plt.close
        
#         #Shap graph
#         explainer = shap.Explainer(model,X_train)
#         shap_values = explainer(X_train)
#         shap.summary_plot(shap_values)    
#         plt.savefig(f'{n_iter}_fold/shap{timestamp}.png')
#         plt.clf() # clear the plot
        
        # Plot POD
        plot_width = 400  #@param {type:'number'}
        plot_height = 350  #@param {type:'number'}
        title_angle = 0 #@param {type:'number'}

        # Create dataframes for actual and predicted values
        actual = pd.DataFrame(y_test.values, columns=['label'])
        predi = pd.DataFrame(y_prob, columns=['Cancer'])
        predi['model'] = f'{model_name}'

        # Concatenate actual and predicted values into a single dataframe
        podi = pd.concat([actual, predi], axis=1)
        podi['label'] = np.where(podi['label'] == 1.0, 'Cancer', 'Control')

        # Create a box plot with Plotly
        fig = px.box(podi, x="label", y="Cancer", facet_col='model', width=plot_width, height=plot_height,
                     color='label', color_discrete_sequence=['#CC0052', '#0052cc'], points='all', 
                     labels={'Cancer' : 'POD (Cancer)'})
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
        fig.update_layout(font=dict(size=13), yaxis=dict(tickfont=dict(size=11)), xaxis=dict(tickfont=dict(size=11)))
        fig.update_xaxes(categoryorder='array', categoryarray=['Control', 'Cancer'])
        fig.update_layout(annotations=[dict(textangle=title_angle) for _ in fig['layout']['annotations']])
        fig.add_hline(y=best_thresh.round(4), col=1, line_dash="dot")

        # save graph
        pio.write_image(fig, file=f"{n_iter}_fold/pod_box_plot_{timestamp}.png")


        
        #Print n-fold results
        fold_results = {
            f'{n_iter}fold_result': {
                'auc': auc.round(4),
                'accuracy': accuracy.round(4),
                'recall': recall.round(4),
                'precision': precision.round(4),
                'f1_score': f1_score.round(4),
            }
        }
        with open(f'fold{n_iter}_result', 'w') as f:
            json.dump(fold_results, f)
            
        print(fold_results)
        if n_iter == args.n_folds:
            break


    # Compute the mean and standard deviation of the evaluation metrics
    mean_auc = np.mean(aucs).round(4)
    std_auc = np.std(aucs).round(4)
    mean_accuracy = np.mean(accuracies).round(4)
    std_accuracy = np.std(accuracies).round(4)
    mean_recall = np.mean(recalls).round(4)
    std_recall = np.std(recalls).round(4)
    mean_precision = np.mean(precisions).round(4)
    std_precision = np.std(precisions).round(4)
    mean_f1_scores = np.mean(f1_scores).round(4)
    std_f1_scores = np.std(f1_scores).round(4)    

    # Save the results to a JSON file
    results = {
        'auc': f'{mean_auc}±{std_auc}'
        ,
        'accuracy': f'{mean_accuracy}±{std_accuracy}'
        ,
        'recall': f'{mean_recall}±{std_recall}'
        ,
        'precision': f'{mean_precision}±{std_precision}'
        ,
        'f1_score':  f'{mean_f1_scores}±{std_f1_scores}'
    }
    print(results)
    with open(args.output_path, 'w') as f:
        json.dump(results, f)







