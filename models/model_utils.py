import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from tqdm import tqdm


def accuracy_score(y_pred, y_test):
    """
    Calculate accuracy score for multiclass classification.

    Parameters:
    y_pred (numpy.ndarray): Predicted labels.
    y_test (numpy.ndarray): Ground truth labels.

    Returns:
    float: Accuracy score.
    """
    # Count the number of correct predictions
    correct = np.sum(y_pred == y_test)

    # Total number of samples
    total = len(y_test)

    # Calculate accuracy
    accuracy = correct / total

    return accuracy


def precision_score(y_pred, y_test, average='macro'):
    """
    Calculate precision score for multiclass classification.

    Parameters:
    y_pred (numpy.ndarray): Predicted labels.
    y_test (numpy.ndarray): Ground truth labels.
    average (str): Type of averaging to calculate precision. Possible values: 'macro', 'micro', 'weighted'.
                   Defaults to 'macro'.

    Returns:
    float: Precision score.
    """
    # Initialize precision dictionary to store precision values for each class
    precision_dict = {}

    # Get unique classes
    unique_classes = np.unique(np.concatenate((y_pred, y_test)))

    # Calculate precision for each class
    for cls in unique_classes:
        # Calculate true positives (TP)
        true_positives = np.sum((y_pred == cls) & (y_test == cls))

        # Calculate total positive predictions (TP + FP)
        total_positives = np.sum(y_pred == cls)

        # Avoid division by zero
        if total_positives == 0:
            precision_dict[cls] = 0
        else:
            # Calculate precision for the class
            precision_dict[cls] = true_positives / total_positives

    # Calculate average precision based on the type of averaging
    if average == 'macro':
        # Macro-average: average precision across all classes
        precision = np.mean(list(precision_dict.values()))
    elif average == 'micro':
        # Micro-average: total true positives / total positive predictions
        precision = np.sum(list(precision_dict.values())) / len(y_test)
    elif average == 'weighted':
        # Weighted-average: weighted precision by class support
        precision = np.sum([precision_dict[cls] * np.sum(y_test == cls) for cls in unique_classes]) / len(y_test)
    else:
        raise ValueError("Invalid averaging type. Possible values: 'macro', 'micro', 'weighted'.")

    return precision


def recall_score(y_pred, y_test, average='macro'):
    """
    Calculate recall score for multiclass classification.

    Parameters:
    y_pred (numpy.ndarray): Predicted labels.
    y_test (numpy.ndarray): Ground truth labels.
    average (str): Type of averaging to calculate recall. Possible values: 'macro', 'micro', 'weighted'.
                   Defaults to 'macro'.

    Returns:
    float: Recall score.
    """
    # Initialize recall dictionary to store recall values for each class
    recall_dict = {}

    # Get unique classes
    unique_classes = np.unique(np.concatenate((y_pred, y_test)))

    # Calculate recall for each class
    for cls in unique_classes:
        # Calculate true positives (TP)
        true_positives = np.sum((y_pred == cls) & (y_test == cls))

        # Calculate total actual positive instances (TP + FN)
        total_positives = np.sum(y_test == cls)

        # Avoid division by zero
        if total_positives == 0:
            recall_dict[cls] = 0
        else:
            # Calculate recall for the class
            recall_dict[cls] = true_positives / total_positives

    # Calculate average recall based on the type of averaging
    if average == 'macro':
        # Macro-average: average recall across all classes
        recall = np.mean(list(recall_dict.values()))
    elif average == 'micro':
        # Micro-average: total true positives / total actual positive instances
        recall = np.sum(list(recall_dict.values())) / len(y_test)
    elif average == 'weighted':
        # Weighted-average: weighted recall by class support
        recall = np.sum([recall_dict[cls] * np.sum(y_test == cls) for cls in unique_classes]) / len(y_test)
    else:
        raise ValueError("Invalid averaging type. Possible values: 'macro', 'micro', 'weighted'.")

    return recall


def f1_score(y_pred, y_test, average='macro'):
    """
    Calculate F1 score for multiclass classification.

    Parameters:
    y_pred (numpy.ndarray): Predicted labels.
    y_test (numpy.ndarray): Ground truth labels.
    average (str): Type of averaging to calculate F1 score. Possible values: 'macro', 'micro', 'weighted'.
                   Defaults to 'macro'.

    Returns:
    float: F1 score.
    """
    # Calculate precision and recall
    precision = precision_score(y_pred, y_test, average=average)
    recall = recall_score(y_pred, y_test, average=average)

    # Avoid division by zero
    if precision + recall == 0:
        f1 = 0
    else:
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall)

    return f1


def roc_curve(y_true, y_pred):
    """
    Compute the Receiver Operating Characteristic (ROC) curve.

    Parameters:
        y_true (array-like): Ground truth binary labels (0 or 1).
        y_pred (array-like): Predicted probabilities or scores.

    Returns:
        fpr (array): False Positive Rate (FPR) values.
        tpr (array): True Positive Rate (TPR) values.
        thresholds (array): Threshold values.
    """
    # Convert y_pred to numpy array if it's not already
    y_pred = np.asarray(y_pred)

    # Sort predictions and true labels by predicted probabilities
    sorted_indices = np.argsort(y_pred)[::-1]
    y_pred_sorted = y_pred[sorted_indices]
    y_true_sorted = np.asarray(y_true)[sorted_indices]

    # Initialize arrays to store true positive rate (TPR), false positive rate (FPR), and thresholds
    tpr = np.zeros(len(y_pred) + 1)
    fpr = np.zeros(len(y_pred) + 1)
    thresholds = np.zeros(len(y_pred) + 1)

    # Initialize counts for true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN)
    TP = 0
    FP = 0
    TN = np.sum(y_true == 0)
    FN = np.sum(y_true == 1)

    # Initialize previous probability
    prev_prob = -np.inf

    # Compute TPR, FPR, and thresholds
    for i, prob in enumerate(np.concatenate([[prev_prob], y_pred_sorted])):
        tpr[i] = TP / (TP + FN)
        fpr[i] = FP / (FP + TN)
        thresholds[i] = prob

        # Update counts
        if i < len(y_pred):
            if y_true_sorted[i] == 1:
                TP += 1
                FN -= 1
            else:
                FP += 1
                TN -= 1

        # Update previous probability
        prev_prob = prob

    return fpr, tpr, thresholds


def auc(fpr, tpr):
    """
    Calculate the Area Under the Receiver Operating Characteristic (ROC) Curve (AUC).

    Parameters:
        fpr (array): False Positive Rate (FPR) values.
        tpr (array): True Positive Rate (TPR) values.

    Returns:
        float: Area under the ROC curve.
    """
    # Sort FPR and TPR arrays by increasing FPR
    sorted_indices = np.argsort(fpr)
    fpr_sorted = fpr[sorted_indices]
    tpr_sorted = tpr[sorted_indices]

    # Calculate area under the ROC curve using trapezoidal rule
    area = np.trapz(tpr_sorted, fpr_sorted)

    return area


def taylor_diagram(y_true, y_pred, label_true='Ground Truth', label_pred='Predicted'):
    """
    Generate a Taylor diagram for comparing two datasets.

    Parameters:
        y_true (array-like): Ground truth values.
        y_pred (array-like): Predicted values.
        label_true (str): Label for the ground truth dataset.
        label_pred (str): Label for the predicted dataset.
    """
    # Compute statistics
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    std_ratio = np.std(y_pred) / np.std(y_true)

    # Create Taylor diagram
    plt.figure(figsize=(8, 8))
    plt.plot(1, corr, marker='o', markersize=10, label='Correlation', color='blue')
    plt.plot(np.sqrt(corr ** 2 + (1 - corr ** 2) * std_ratio ** 2), rmse, marker='o', markersize=10, label='RMSE',
             color='green')
    plt.plot(std_ratio, 0, marker='o', markersize=10, label='Standard deviation ratio', color='red')

    # Add reference lines
    plt.plot([0, 1], [0, 0], '--', color='gray', linewidth=0.5)
    plt.plot([0, 0], [0, rmse], '--', color='gray', linewidth=0.5)
    plt.plot([1, 1], [0, np.sqrt(2 * rmse ** 2)], '--', color='gray', linewidth=0.5)

    # Add labels and legend
    plt.text(1.05, corr, label_true, verticalalignment='center')
    plt.text(np.sqrt(corr ** 2 + (1 - corr ** 2) * std_ratio ** 2), rmse, label_pred, verticalalignment='center')
    plt.text(std_ratio, 0, '1', verticalalignment='center')
    plt.xlabel('Standard deviation ratio')
    plt.ylabel('Correlation / RMSE')
    plt.title('Taylor Diagram')
    plt.legend()
    plt.grid(True)
    plt.show()


def violin_plot(y_true, y_pred, labels=['Ground Truth', 'Predicted']):
    """
    Generate a violin plot for comparing two datasets.

    Parameters:
        y_true (array-like): Ground truth values.
        y_pred (array-like): Predicted values.
        labels (list): Labels for the datasets.
    """
    # Combine the datasets into a single DataFrame
    data = {'Values': np.concatenate([y_true, y_pred]),
            'Dataset': np.repeat(labels, [len(y_true), len(y_pred)])}
    df = pd.DataFrame(data)

    # Create violin plot
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='Dataset', y='Values', data=df)
    plt.xlabel('Dataset')
    plt.ylabel('Values')
    plt.title('Violin Plot')
    plt.show()


def r2_score(y_true, y_pred):
    """
    Compute the R-squared (R2) score for regression task.

    Parameters:
        y_true (array-like): Ground truth values.
        y_pred (array-like): Predicted values.

    Returns:
        float: R-squared (R2) score.
    """
    # Convert input arrays to numpy arrays if they are not already
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Calculate mean of the true values
    mean_true = np.mean(y_true)

    # Calculate total sum of squares (TSS)
    tss = np.sum((y_true - mean_true) ** 2)

    # Calculate residual sum of squares (RSS)
    rss = np.sum((y_true - y_pred) ** 2)

    # Calculate R-squared (R2) score
    r2 = 1 - (rss / tss)

    return r2


def rmse_score(y_true, y_pred):
    """
    Compute the Root Mean Squared Error (RMSE) for regression task.

    Parameters:
        y_true (array-like): Ground truth values.
        y_pred (array-like): Predicted values.

    Returns:
        float: Root Mean Squared Error (RMSE).
    """
    # Convert input arrays to numpy arrays if they are not already
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Calculate squared errors
    squared_errors = (y_true - y_pred) ** 2

    # Calculate mean squared error
    mean_squared_error = np.mean(squared_errors)

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error)

    return rmse


def mae_score(y_true, y_pred):
    """
    Compute the Mean Absolute Error (MAE) for regression task.

    Parameters:
        y_true (array-like): Ground truth values.
        y_pred (array-like): Predicted values.

    Returns:
        float: Mean Absolute Error (MAE).
    """
    # Convert input arrays to numpy arrays if they are not already
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Calculate absolute errors
    absolute_errors = np.abs(y_true - y_pred)

    # Calculate mean absolute error
    mean_absolute_error = np.mean(absolute_errors)

    return mean_absolute_error


def do_mann_whiteney_test(x, y, columns):
    results = []
    x = pd.DataFrame(x, columns=columns)
    for column in tqdm(columns):
        # Get data for the current column
        data = x[column].values.reshape(-1, 1)
        # Initialize lists to store statistics and p-values
        statistic_list = []
        pvalue_list = []
        # Perform Mann-Whitney U test for each class
        for class_label in np.unique(y):
            # Select data for the current class
            data_class = data[y == class_label]
            # Perform Mann-Whitney U test
            statistic, pvalue = mannwhitneyu(data_class, data)
            # Append results to lists
            statistic_list.append(statistic)
            pvalue_list.append(pvalue)
        # Store results in a DataFrame
        result_df = pd.DataFrame({'Class': np.unique(y), 'Statistic': statistic_list, 'P-value': pvalue_list})
        result_df.set_index('Class', inplace=True)
        results.append((column, result_df))
    return results


def visualize_mannwhiteney_results(results, save_paths):
    # TODO: Save the Plots
    plot_save_path = save_paths["plots_path"]

    for column, result_df in results:
        # Plot p-values for each class
        # result_df["P-value"].plot(kind='bar', title=f'Mann-Whitney U Test for {column}', xlabel='Class', ylabel='P-value')
        x = np.array([0, 1, 2, 3, 4]).reshape(-1, 1)
        y = np.array(result_df["P-value"]).reshape(-1, 1)

        plt.bar(x, y)
        plt.axhline(y=0.05, color='r', linestyle='--', label='Significance Level (0.05)')
        plt.legend()
        plt.show()


def do_pc_columns_analysis(x, y, columns, save_paths):
    path = save_paths["plots_path"]

    # Combine x and y into a single DataFrame
    df = pd.DataFrame(x, columns=columns)
    df['Class'] = y

    # Group by class and calculate principal components for each feature
    results = {'Feature': [], 'PC1': [], 'PC2': [], 'PC3': []}  # Add more PCs as needed
    for feature in columns:
        print(feature)
        pca = PCA(n_components=3)  # Specify number of principal components to calculate
        pca.fit(df.groupby('Class')[feature].apply(list).values)
        results['Feature'].append(feature)
        results['PC1'].append(pca.components_[0])
        results['PC2'].append(pca.components_[1])
        results['PC3'].append(pca.components_[2])  # Add more PCs as needed

    # Create DataFrame from results
    pc_df = pd.DataFrame(results)
    return pc_df


def do_clustering_analysis(x, y, save_paths):
    path = save_paths["plots_path"]

    # Dimensionality reduction using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    x_tsne = tsne.fit_transform(x)

    # Clustering using KMeans
    kmeans = KMeans(n_clusters=len(np.unique(y)), random_state=42)
    kmeans.fit(x)
    labels = kmeans.labels_

    # Calculate DBI and SI Index
    dbi = davies_bouldin_score(x, labels)
    si = silhouette_score(x, labels)
    print(f'Davies-Bouldin Index: {dbi}')
    print(f'Silhouette Index: {si}')

    # Visualize clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.title('t-SNE Visualization of Clusters')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.colorbar(label='Cluster')
    plt.show()
    pass
