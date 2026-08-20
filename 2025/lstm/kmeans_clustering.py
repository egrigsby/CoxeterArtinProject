import pandas as pd
import numpy as np
import ast
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
from collections import Counter

#kmeans with epoch, match ratio, Jaccard similarity, and Hamming distance

# Load your CSV
dataframe = pd.read_csv('C:/Users/rigogliv/Desktop/expmmll/0716wrong_predictions.csv')  # ← change if needed

# Convert sequences from strings to lists
dataframe['label'] = dataframe['label'].apply(ast.literal_eval)
dataframe['predicted'] = dataframe['predicted'].apply(ast.literal_eval)

# Feature: match ratio
def match_ratio(a, b):
    return sum(i == j for i, j in zip(a, b)) / max(len(a), 1)

# Feature: Jaccard similarity
def jaccard(a, b):
    return len(set(a) & set(b)) / len(set(a) | set(b)) if a and b else 0

#Feature: token drop rate
def token_drop_rate(a, b):
    count=Counter(a)
    countpreds=Counter(b)
    # Calculate the number of tokens dropped in predictions
    dropped = sum(count[token] - countpreds[token] for token in count if count[token] > countpreds[token])
    return dropped / max(len(a), 1)

#Feature: hamming distance
def hamming_distance(a, b):
    if len(a) != len(b):
        raise ValueError("Sequences must be of the same length")
    return sum(el1 != el2 for el1, el2 in zip(a, b))


df=dataframe
# Apply features
df['match_ratio'] = dataframe.apply(lambda row: match_ratio(row['label'], row['predicted']), axis=1)
df['jaccard'] = dataframe.apply(lambda row: jaccard(row['label'], row['predicted']), axis=1)
df['token_drop_rate'] = dataframe.apply(lambda row: token_drop_rate(row['label'], row['predicted']), axis=1)
df['hamming_distance'] = dataframe.apply(lambda row: hamming_distance(row['label'], row['predicted']), axis=1)

# Final features for clustering
features = df[['epoch', 'match_ratio', 'jaccard', 'token_drop_rate', 'hamming_distance']]

# Normalize
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(features_scaled)
# Pairplot to visualize
sns.pairplot(df[['epoch', 'match_ratio', 'jaccard', 'token_drop_rate','hamming_distance','cluster']], hue='cluster', palette='tab10')
plt.suptitle('K-Means Clustering (epoch, match ratio, Jaccard, token drop rate, Hamming)', y=1.02)
plt.show()

# Save all data with cluster labels to a new CSV 
df.to_csv('clustered_output_kmeans.csv', index=False) #change filename as needed
