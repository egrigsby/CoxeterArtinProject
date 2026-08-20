import pandas as pd
import ast
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

#Take the clustered outputs (which have epoch, match ratio, Jaccard similarity, token drop rate, and hamming distnace as features)
#This is to compare each of the clusters by confusion matrix

# Load your CSV
dataframe = pd.read_csv('C:/Users/rigogliv/Desktop/expmmll/0723clustered_output_kmeans5.csv')  # ← change if needed

# Convert sequences from strings to lists
dataframe['epoch'] = dataframe['epoch'].astype(int)
dataframe['tokens'] = dataframe['tokens'].apply(ast.literal_eval)
dataframe['label'] = dataframe['label'].apply(ast.literal_eval)
dataframe['predicted'] = dataframe['predicted'].apply(ast.literal_eval)
dataframe['match_ratio'] = dataframe['match_ratio'].astype(float)
dataframe['jaccard'] = dataframe['jaccard'].astype(float)
dataframe['token_drop_rate'] = dataframe['token_drop_rate'].astype(float)
dataframe['hamming_distance'] = dataframe['hamming_distance'].astype(float)
dataframe['cluster'] = dataframe['cluster'].astype(int)

# Get global set of all token labels
all_labels = sorted(set(dataframe['label'].explode().unique()) | set(dataframe['predicted'].explode().unique()))

#for loop to account for each cluster
for i in range(dataframe['cluster'].max()+1):
    cluster_df = dataframe[dataframe['cluster'] == i]
    y_true = []
    y_pred = []
    for _, row in cluster_df.iterrows():
        true_seq = row['label']
        pred_seq = row['predicted']
        for true_token, pred_token in zip(true_seq, pred_seq):
            y_true.append(true_token)
            y_pred.append(pred_token)
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred, labels=all_labels)
    #plotting confusion matrix for each cluster
    cmdisplay = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=all_labels)
    cmdisplay.plot(cmap=plt.cm.Blues) #change color as desired
    plt.title(f'Confusion Matrix for Cluster {i}')
    plt.savefig(f'confusion_matrix_cluster_{i}of5.png') #change the name for the appropriate amount of total clusters
    plt.show() 

#If kmeans was used, this will not have noise points. Disregard the following.
#Uncomment the following code if DBSCAN was used  - this will handle the outliers and the associated confusion matrix
'''
cluster_df = dataframe[dataframe['cluster'] == -1]  # Noise points in DBSCAN
y_true = []
y_pred = []
for _, row in cluster_df.iterrows():
    true_seq = row['label']
    pred_seq = row['predicted']
    for true_token, pred_token in zip(true_seq, pred_seq):
        y_true.append(true_token)
        y_pred.append(pred_token)
confusion_matrix = metrics.confusion_matrix(y_true, y_pred, labels=all_labels)
cmdisplay = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=all_labels)
cmdisplay.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Noise Points')
plt.savefig('0722confusion_matrix_noise.png')
plt.show()
print('Noise points confusion matrix saved and displayed.')
'''
