import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
from sklearn.pipeline import Pipeline
import kds

# Data Visualization
# 1. Numerical histogram


def plot_num_hist(df, numerical_vars):
    num_plots = len(numerical_vars)
    num_cols = 3
    num_rows = math.ceil(num_plots / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))

    for i, var in enumerate(numerical_vars):
        row = i // num_cols
        col = i % num_cols
        sns.histplot(df[var].dropna(), kde=True, bins=30, ax=axes[row, col])
        axes[row, col].set_title(var)

    plt.tight_layout()
    plt.show()

# 2. Box plot for numerical variables


def plot_num_box(df, numerical_vars):
    num_plots = len(numerical_vars)
    num_cols = 3
    num_rows = math.ceil(num_plots / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))

    for i, var in enumerate(numerical_vars):
        row = i // num_cols
        col = i % num_cols
        sns.boxplot(y=var, data=df, ax=axes[row, col])
        axes[row, col].set_title(var)

    plt.tight_layout()
    plt.show()

# 3. Visualize correlation between numerical variables


def plot_corr_heatmap(df, numerical_vars):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numerical_vars].corr(), annot=True,
                cmap='coolwarm_r', center=0)
    plt.title('Correlation Heatmap for Numerical Variables')
    plt.show()

# 4. Visualize bar plot for categorical variables


def plot_cat_bar(df, categorical_vars):
    num_plots = len(categorical_vars)
    num_cols = 4
    num_rows = math.ceil(num_plots / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))

    for i, var in enumerate(categorical_vars):
        row = i // num_cols
        col = i % num_cols
        sns.countplot(x=var, data=df, ax=axes[row, col])
        axes[row, col].set_title(var)

    plt.tight_layout()
    plt.show()

# 5. Plot missing ness proportion between the the columns that contain missing values and other feautes


def plot_missingness_proportions(ct_norm1, ct_norm2, title1, title2):
    plt.figure(figsize=(16, 12))

    # Visualization for the first set of missingness proportions
    plt.subplot(2, 2, 1)
    ct_norm1.plot(kind='bar', stacked=True, title=title1, ax=plt.gca())
    plt.ylabel('Proportion')

    plt.subplot(2, 2, 2)
    ct_norm2.plot(kind='bar', stacked=True, title=title2, ax=plt.gca())
    plt.ylabel('Proportion')

    plt.tight_layout()
    plt.show()

# 6. Percentage Crosstab between categorical variables and the label


def plot_percentage_crosstab(X_train, y_train, low_cardinality):
    num_vars = len(low_cardinality)
    num_cols = num_vars // 2 + num_vars % 2

    # two rows
    fig, axes = plt.subplots(nrows=2, ncols=num_cols,
                             figsize=(8 * num_cols, 10))
    axes = axes.flatten()

    # plot heatmap
    for i, var in enumerate(low_cardinality):
        crosstab = pd.crosstab(
            X_train[var], y_train, normalize='columns') * 100

        order = X_train[var].value_counts().index

        sns.heatmap(crosstab.loc[order], annot=True,
                    fmt=".2f", cmap="YlGnBu", ax=axes[i])
        axes[i].set_title(f'Percentage Crosstab of {var} vs Target')
        axes[i].set_ylabel(var)
        axes[i].set_xlabel('Target')

    # hide unused subplots
    for j in range(num_vars, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# 7.Plot boxplot and histogram for numerical variables vs the label


def plot_boxplot_and_histogram(df, numerical_features, label):
    # Calculate the number of rows needed for subplots
    n_rows = len(numerical_features)
    # Change figsize to a larger size
    fig, axes = plt.subplots(n_rows, 2, figsize=(30, 60))

    for i, col in enumerate(numerical_features):
        # Boxplot on the left
        sns.boxplot(data=df, x=label, y=col, ax=axes[i, 0])
        axes[i, 0].set_title(f"Boxplot of {col} vs {label}")

        # Histogram on the right
        sns.histplot(data=df, x=col, hue=label, ax=axes[i, 1], kde=True)
        axes[i, 1].set_title(f"Distribution of {col}")

    plt.tight_layout()
    plt.show()

# Data Analysis
# --------------------------------------

# Feature Engineering
# 1. Columns drop (Drop columns based on a provided list)


def drop_columns(df, drop_list):
    cols_to_drop = [col for col in drop_list if col in df.columns]
    df = df.drop(cols_to_drop, axis=1)
    return df

# 2. replace missing values with the median of the column


def fill_missing_median(df, missing_vars):
    missing_vars = [var for var in missing_vars if var in df.columns]
    for var in missing_vars:
        median = df[var].median()
        df[var] = df[var].fillna(median)
    return df

# 3. data preprocessor for the validation set


def data_preprocessor(data):
    ''' 
    This function converts the 'addnl_pmt' and 'missed_payment' columns to binary and drop some columns 
    '''

    # Convert 'addnl_pmt' to binary
    data['addnl_pmt'] = data['addnl_pmt'].apply(lambda x: 1 if x > 0 else 0)

    # Convert 'missed_payment' to binary
    data['missed_payment'] = data['days_dlnqn'].apply(
        lambda x: 1 if x > 0 else 0)

    # Convert the event to binary

    data['event'] = data['event'].apply(lambda x: 1 if x == 1 else 0)

    # drop some uncessary columns
    data.drop(['z', 'SEGMENT', 'WIN_FICO', 'fico_bucketsm', 'dstype',
              'days_dlnqn', 'tenure', 'coll_S'], axis=1, inplace=True)

    return data

# 4. Prepare the data for scoring


def scoring_data_creation(data, scoring_time, hit_range):
    """
    Creates the scoring dataset based on scoring time, hit range, and aggregates events.

    Parameters:
    - data: The initial dataset.
    - scoring_time: The specified scoring time.
    - hit_range: The range within which hits are considered.

    Returns:
    - A DataFrame filtered for the scoring date with aggregated event information.
    """
    # Filter based on scoring time and hit range
    scoring_data = data[(data['time'] >= scoring_time) & (
        data['time'] <= scoring_time + hit_range)]

    # Aggregate events and get minimum time for each id
    event_detection = scoring_data.groupby('id').agg(
        {'event': 'sum', 'time': 'min'}).reset_index()
    # Add aggregated event count as 'hit'
    event_detection['hit'] = event_detection['event']

    # Merge the aggregated info back to the scoring data
    scoring_data_merged = scoring_data.merge(
        event_detection[['id', 'hit']], on='id', how='left')

    # Filter to select data for the scoring date only
    final_scoring_data = scoring_data_merged[scoring_data_merged['time'] == scoring_time]

    return final_scoring_data

# --------------------------------------

# Model Building and Prediction
# 1. Model Building: Create a function to automate the pipeline (preq: preprocessor)


def train_model(algorithm, preprocessor, X_train, y_train):
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', algorithm)])
    model = pipeline.fit(X_train, y_train)
    return model

# 2. Model Prediction: Create a function to automate the prediction


def get_predictions(model, X_test):
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    return y_pred, y_pred_prob

# --------------------------------------

# Model Evaluation functions and viz

# 1. Calculate Average squared error


def average_squared_error(y_true, y_pred_prob):
    """
    Calculate the average squared error.

    Parameters:
    - y_true: The true target values.
    - y_pred_prob: The predicted target values.

    Returns:
    - The average squared error.
    """
    return np.mean((y_true - y_pred_prob) ** 2)

# 1. Performance metrics


class PerformanceMetrics:
    def __init__(self, y_true, y_pred, y_pred_prob):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_prob = y_pred_prob
        self.accuracy = accuracy_score(y_true, y_pred)
        self.precision = precision_score(y_true, y_pred)
        self.recall = recall_score(y_true, y_pred)
        self.roc_auc = roc_auc_score(y_true, y_pred_prob)
        self.avg_squared_error = average_squared_error(y_true, y_pred_prob)

    def print_performance_metrics(self, model_name=None):
        print(f'Performance Metrics of {model_name}: \n')
        print(f'Accuracy: {self.accuracy:.4f}')
        print(f'Precision: {self.precision:.4f}')
        print(f'Recall: {self.recall:.4f}')
        print(f'ROC AUC: {self.roc_auc:.4f}')
        print(f'Averaged Squared Error: {self.avg_squared_error:.4f}')

# 2. Obtain prediction scores for different periods


def obtain_prediction_score(valid, logistic_model, us_xgb_pipe, scoring_time=6, hit_range=6):
    scoring_data = scoring_data_creation(valid, scoring_time, hit_range)

    # score the data with the logistic model
    scoring_data['logistic_prediction'] = logistic_model.predict(scoring_data)

    # score the data with the xgboost model
    scoring_data['xgb_prediction'] = us_xgb_pipe.predict(scoring_data)

    # record the prediction probability
    scoring_data['logistic_prediction_prob'] = logistic_model.predict_proba(scoring_data)[
        :, 1]
    scoring_data['xgb_prediction_prob'] = us_xgb_pipe.predict_proba(scoring_data)[
        :, 1]

    # obtain the performance metrics
    logistic_metrics = PerformanceMetrics(
        scoring_data['hit'], scoring_data['logistic_prediction'], scoring_data['logistic_prediction_prob'])
    xgb_metrics = PerformanceMetrics(
        scoring_data['hit'], scoring_data['xgb_prediction'], scoring_data['xgb_prediction_prob'])

    logistic_metrics.print_performance_metrics('Logistic Regression')

    print('\n')

    xgb_metrics.print_performance_metrics('XGBoost')


# 2. ROC curve


def plot_roc_curve(y_true, y_pred_prob, title='ROC Curve'):
    """
    Plot the ROC curve.

    Parameters:
    - y_true: The actual labels.
    - y_pred_prob: The predicted probabilities from the model.
    - title: The title of the plot.

    Returns:
    - The ROC curve.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = roc_auc_score(y_true, y_pred_prob)  # Use renamed auc function
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

# 3. Cumulative Gains Chart


def plot_cumulative_gain(data, target_col, model_pred_prob, title='Cumulative Gains Chart'):
    """
    Plot the cumulative gains chart for a model.

    Parameters:
    - data: The data containing the actual labels and the predicted probabilities.
    - target_col: The name of the target column in the data.
    - model_pred_prob: The predicted probabilities from the model.
    - title: The title of the plot.
    """
    kds.metrics.plot_cumulative_gain(data[target_col], model_pred_prob)
    plt.title(title)
    plt.show()

# 4. feature important for ranfom forest and xgboost


def plot_tree_fi(rf_pipeline, title='Top 10 Features Importance in Random Forest'):
    # Get feature names
    feature_names = rf_pipeline.named_steps['preprocessor'].get_feature_names_out(
    )

    # Get feature importance
    feature_importance = rf_pipeline.named_steps['classifier'].feature_importances_

    # Create DataFrame for feature importance
    feature_importance_df = pd.DataFrame(
        {'feature': feature_names, 'importance': feature_importance})

    # Sort DataFrame by importance
    feature_importance_df = feature_importance_df.sort_values(
        'importance', ascending=False)

    # Plot feature importance for Random Forest
    plt.figure(figsize=(10, 6))
    plt.title(title)

    # Create horizontal bar plot
    plt.barh(feature_importance_df['feature'][:10],
             feature_importance_df['importance'][:10], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.gca().invert_yaxis()

    plt.show()

# 5. threshold precision recall curve


def plot_threshold_precision_recall(y_true, y_pred_prob, title='Threshold-Precision-Recall Curve'):
    """
    Plot the relationship between the threshold and precision, and threshold and recall.

    Parameters:
    - y_true: The true target values.
    - y_pred_prob: The predicted target values.

    Returns:
    - The threshold-precision-recall curve.
    """
    thresholds = np.linspace(0, 1, 100)
    precisions = [precision_score(y_true, y_pred_prob > t) for t in thresholds]
    recalls = [recall_score(y_true, y_pred_prob > t) for t in thresholds]

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precisions, marker='o',
             linestyle='--', color='blue', label='Precision')
    plt.plot(thresholds, recalls, marker='o',
             linestyle='--', color='orange', label='Recall')
    plt.title(title)
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.show()

# 6. plot all roc curves


def plot_all_roc_curves(pred_prob, model_names, y_test):
    for i, pred_prob in enumerate(pred_prob):

        # Calculate the ROC curve
        fpr, tpr, _ = roc_curve(y_test, pred_prob)
        roc_auc = auc(fpr, tpr)

        # Determine the color for the plot
        color = 'red' if model_names[i] == 'XGBoost' else 'grey'

        # Plot the ROC curve
        plt.plot(fpr, tpr, color=color,
                 label=f'{model_names[i]} (AUC: {roc_auc:.2f})')

    # Plot the diagonal line
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# partial dependence plot


def pdp_plot_numeric(X_resampled, var, sample_n, pipeline):
    # var = 'credit_amount'
    pdp_values = pd.DataFrame(X_resampled[var].sort_values().sample(
        frac=0.1).unique(), columns=[var])
    pdp_sample = X_resampled.sample(sample_n).drop(var, axis=1)

    pdp_cross = pdp_sample.merge(pdp_values, how='cross')
    pdp_cross['pred'] = pipeline.predict_proba(pdp_cross)[:, 1]
    plt.figure(figsize=(10, 3))
    sns.lineplot(x=f"{var}", y='pred', data=pdp_cross)
    plt.title(f"Partial Dependance Plot: {var}")
    plt.ylabel('Predicted Probability')
    plt.xticks(rotation=45)
    # plt.ylim(0, 1)
    plt.grid(True)
    plt.show()


def pdp_plot_categorical(X_resampled, var, sample_n, pipeline):
    sns.set_style("whitegrid")  # Try "darkgrid", "ticks", etc.
    # Try "paper", "notebook", "poster" for different sizes
    sns.set_context("notebook")

    pdp_values = pd.DataFrame(
        X_resampled[var].sort_values().unique(), columns=[var])
    pdp_sample = X_resampled.sample(sample_n).drop(var, axis=1)

    pdp_cross = pdp_sample.merge(pdp_values, how='cross')
    pdp_cross['pred'] = pipeline.predict_proba(pdp_cross)[:, 1]
    mean_pred = pdp_cross['pred'].mean()
    pdp_cross['pred'] = pdp_cross['pred'].apply(lambda x: x - mean_pred)
    plt.figure(figsize=(10, 3))
   # sns.lineplot(x=f"{var}", y='pred', data=pdp_cross)
    sns.barplot(x=f"{var}", y='pred',
                ci=None,
                data=pdp_cross,
                estimator="mean")
    plt.title(f"Partial Dependance Plot: {var}")
    plt.ylabel('Predicted Probability')
    plt.xticks(rotation=45)
    # plt.ylim(0, 1)
    plt.grid(True)
    plt.show()
