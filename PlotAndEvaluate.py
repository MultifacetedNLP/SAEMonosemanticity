import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
import seaborn as sns
from utility import generalized_js_distance
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score




combination_type = "last_token"  # Options: "last_token", "all_tokens_max"
data_name = "ner"  # Change this to the dataset you want to use, e.g., "ag _news", "imdb", etc.
use_sae = False  # Set to True if you want to use SAE
# sae_id = "layer_25/width_65k/canonical"
# sae_id = "layer_25/width_16k/canonical"
# sae_id = "layer_25/width_16k/average_l0_285"
sae_threshold = 0.1
# Adjust the bw_adjust parameter for smoothing (e.g., 1.5 means 50% more smoothing).
smooth_factor = 1.5
plot_scatter = False  # Set to True if you want to plot scatter plots
plot_distribution = False  # Set to True if you want to plot distributions

model = "Llama-3.1-8B"

if model == "gemma-2-2b":
    storage_path = Path("/u/siddique-d1/Moghis/storage")
    hook_name = "blocks.25.hook_resid_post"
    sae_id = "layer_25/width_65k/average_l0_197"
elif model == "Llama-3.1-8B":
    storage_path = Path("/u/siddique-d1/Moghis/storage/Llama")
    hook_name = "blocks.31.hook_resid_post"
    sae_id = "l31r_800m_slimpajama"

sae_hook_name_post = f"{hook_name}.hook_sae_acts_post"

if data_name == "ag_news":
    distribution_path = storage_path / "plots/dist/news"
    scatter_path = storage_path / "plots/scatter/news"
    labels_dict = {
        0: " world",
        1: " sports",
        2: " business",
        3: " science"
    }
    log_path = storage_path / "news"
elif data_name == "imdb":
    distribution_path = storage_path / "plots/dist/imdb"
    scatter_path = storage_path / "plots/scatter/imdb"
    log_path = storage_path / "imdb"
    labels_dict = {
        0: "negative",
        1: "positive"
    }
elif data_name == "dbpedia":
    distribution_path = storage_path / "plots/dist/dbpedia"
    scatter_path = storage_path / "plots/scatter/dbpedia"
    log_path = storage_path / "dbpedia"
    labels_dict = {
        0: " company",
        1: " educational",
        2: " artist",
        3: " athlete",
        4: " office",
        5: " means",
        6: " building",
        7: " natural",
        8: " village",
        9: " animal",
        10: " plant",
        11: " album",
        12: " film",
        13: " written"
    }
elif data_name == "emotions":
    distribution_path = storage_path / "plots/dist/emotions"
    scatter_path = storage_path / "plots/scatter/emotions"
    log_path = storage_path / "emotions"
    labels_dict = {
        0: " sadness",
        1: " joy",
        2: " love",
        3: " anger",
        4: " fear",
        5: " surprise"
    }

elif data_name == "sst2":
    distribution_path = storage_path / "plots/dist/sst2"
    scatter_path = storage_path / "plots/scatter/sst2"
    log_path = storage_path / "sst2"
    labels_dict = {
        0: "negative",
        1: "positive"
    }
elif data_name == "pos":
    distribution_path = storage_path / "plots/dist/pos"
    scatter_path = storage_path / "plots/scatter/pos"
    log_path = storage_path / "pos"
    labels_dict = {
        "NOUN": " noun",
        "ADV": " adverb",
        "ADJ": " adjective",
        "VERB": " verb"
    }
elif data_name == "ner":
    distribution_path = storage_path / "plots/dist/ner"
    scatter_path = storage_path / "plots/scatter/ner"
    log_path = storage_path / "ner"
    labels_dict = {
        "person": " person",
        "art": " art",
        "organization": " organization",
        "product": " product",
        "event": " event",
        "location": " location",
        "building": " building"
    }
    
classes = list(labels_dict.values())
reversed_labels_dict = {v: k for k, v in labels_dict.items()}
distribution_path.mkdir(parents=True, exist_ok=True)
scatter_path.mkdir(parents=True, exist_ok=True)


if not use_sae:
    # load the logs
    with open(log_path / f"logs_{combination_type}_{hook_name}.json", "r") as f:
        logs = json.load(f)
else:
    # load the logs
    with open(log_path / f"logs_{sae_id.replace('/', '_')}_{combination_type}_{sae_hook_name_post}.json", "r") as f:
        logs = json.load(f)

# list of defaultdict
all_freqs = []
for cats in classes:
    all_freqs.append(defaultdict(list))
all_counts = [0] * len(classes)

labels = []
predictions = []

for label, predicted, activated_features, _ in tqdm(logs):
    labels.append(reversed_labels_dict.get(label, -1))
    predictions.append(reversed_labels_dict.get(predicted, -1))
    if label == predicted:
        class_index = classes.index(label)
        all_counts[class_index] += 1
        if use_sae:
            for feature, activation_value in activated_features:
                all_freqs[class_index][feature].append(activation_value)
        else:
            for feature, activation_value in enumerate(activated_features):
                all_freqs[class_index][feature].append(activation_value)


# print metrics
accuracy = accuracy_score(labels, predictions)
precision_macro = precision_score(labels, predictions, average='macro', zero_division=0)
precision_micro = precision_score(labels, predictions, average='micro', zero_division=0)
precision_weighted = precision_score(labels, predictions, average='weighted', zero_division=0)
recall_macro = recall_score(labels, predictions, average='macro', zero_division=0)
recall_micro = recall_score(labels, predictions, average='micro', zero_division=0)
recall_weighted = recall_score(labels, predictions, average='weighted', zero_division=0)
f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
conf_matrix = confusion_matrix(labels, predictions)

print("-----------------------")
print("Actual activations")
# Output the metrics
print("Accuracy:", accuracy)
print("\nPrecision:")
print("  Macro:", precision_macro)
print("  Micro:", precision_micro)
print("  Weighted:", precision_weighted)

print("\nRecall:")
print("  Macro:", recall_macro)
print("  Micro:", recall_micro)
print("  Weighted:", recall_weighted)

print("\nF1 Score:")
print("  Macro:", f1_macro)
print("  Micro:", f1_micro)
print("  Weighted:", f1_weighted)

print("\nConfusion Matrix:\n", conf_matrix)

features = sorted(set().union(*[set(freq.keys()) for freq in all_freqs]))

all_freqs_lengths = []
for cats in classes:
    all_freqs_lengths.append([])

# Retrieve frequency counts for each feature for both labels
for i, freqs in enumerate(all_freqs):
    for feature in features:
        all_freqs_lengths[i].append(len(freqs.get(feature, [])))
        
"""
# Create a scatter plot for all combinations of labels
all_classes_combinations = [(i, j) for i in range(len(classes)) for j in range(i + 1, len(classes))]
for i, j in all_classes_combinations:
    label_a = classes[i]
    label_b = classes[j]
    freq_a = all_freqs_lengths[i]
    freq_b = all_freqs_lengths[j]
    
    if plot_scatter:
        plt.figure(figsize=(8, 8))
        plt.scatter(freq_a, freq_b, s=20, alpha=0.7)
        plt.xlabel(f"Frequency of features for {label_a}")
        plt.ylabel(f"Frequency of features for {label_b}")
        plt.title(f"Feature Frequency Comparison: {label_a} vs {label_b}")

        # Plot a reference line y = x
        max_val = max(max(freq_a), max(freq_b)) + 1
        plt.plot([0, max_val], [0, max_val], 'r--', label="y = x")

        plt.legend()
        plt.grid(True)

        # Save and show the plot
        plot_path = os.path.join(scatter_path, f"{label_a}_vs_{label_b}_scatter.png")
        plt.savefig(plot_path)
        plt.show()
    
    # Calculate Jaccard similarity
    intersection = np.sum(np.minimum(freq_a, freq_b))
    union = np.sum(np.maximum(freq_a, freq_b))
    jaccard_similarity = intersection / union if union != 0 else 0
    print(f"Jaccard similarity between {label_a} and {label_b}: {jaccard_similarity}")
"""

if plot_distribution:
    for feature in tqdm(features):
        # feature = random.choice(list(frequencies_positive.keys()))
        all_values = []
        class_values = []
        for i, freqs in enumerate(all_freqs):
            values = freqs.get(feature, [])
            class_values.append(values)
            all_values.extend(values)
        
        
        if all(
            len(values) <= sae_threshold * counts
            for values, counts in zip(class_values, all_counts)
        ):
            continue

        # Compute x-range using the available data.
        xmin = min(all_values)
        xmax = max(all_values)

        # Create a new figure for the plot.
        plt.figure(figsize=(10, 6))
        
        # Plot the KDE for each class if the number of samples is above the threshold.
        for i, values in enumerate(class_values):
            if len(values) >= sae_threshold * all_counts[i]:
                sns.kdeplot(values, fill=True, linewidth=2,
                            label=f'{classes[i]} KDE', clip=(xmin, xmax), bw_adjust=smooth_factor)

        # Label the axes and add a title.
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title(f'Smooth Distribution for Feature: {feature}')

        # Add legend and grid.
        plt.legend()
        plt.grid(True)

        # Save and display the figure.
        plt.savefig(os.path.join(distribution_path, f"feature_{feature}_distribution.png"))
        plt.show()



# calculate the polisimanticity of the features
activated_features_counter = 0
polisemanticity_counter = 0
monocsemanticity_counter = 0
polisemanticity_values_max = []
polisemanticity_values_max_poli = []
polisemanticity_values_all = []
polisemanticity_values_all_poli = []
monocsemanticity_values_js = []
monocsemanticity_values_js_poli = []
monocsemanticity_values_aura = []
#monocsemanticity_values_aura_poli = []
for feature in tqdm(features):
    
    all_values = []
    class_values = []
    for i, freqs in enumerate(all_freqs):
        values = freqs.get(feature, [])
        class_values.append(values)
        all_values.extend(values)
    
    activated_counter = 0
    for i, values in enumerate(class_values):
        if len(values) > sae_threshold * all_counts[i] and len(values) > 1:
            activated_counter += 1
            
    if activated_counter == 0:
        # If all distributions are inactivated, skip this feature
        continue
    
    activated_features_counter += 1
    
    if activated_counter == 1:
        # If only one distribution is activated, it is monosemantic
        monocsemanticity_counter += 1
        # print(f"{feature} is monosemantic")
        monocsemanticity_values_js.append(1)
        monocsemanticity_values_aura.append(1)
        polisemanticity_values_all.append(0)
        polisemanticity_values_max.append(0)
        continue
    
    polisemanticity_counter += 1
    
    """
    all_aura = []
    for i, values in enumerate(class_values):
        if len(values) == 0:
            continue
        
        other_values = [val for j, other in enumerate(class_values) if j != i for val in other]
        y_true = np.concatenate([np.ones_like(values), np.zeros_like(other_values)])
        y_scores = np.concatenate([values, other_values])
        
        
        auc = roc_auc_score(y_true, y_scores)
        separability = abs(2 * auc - 1)
        all_aura.append(separability)
    
    monocsemanticity_values_aura.append(np.mean(all_aura))
    """
    xmin = min(all_values)
    xmax = max(all_values)
    
    xs = np.linspace(xmin, xmax, 1000)
   
    
    all_kdes = []
    for i, values in enumerate(class_values):
        if len(values) > sae_threshold * all_counts[i] and len(values) > 1:
            try:
                kde = gaussian_kde(values)(xs)
            except Exception as e:
                print(f"Error calculating KDE for feature {feature} of class {classes[i]}: {e}")
                continue
            all_kdes.append(kde)
        
    kde_arrays = np.stack(all_kdes, axis=0)
    
    joint = np.sort(kde_arrays, axis=0)[-2]
    union = np.sort(kde_arrays, axis=0)[-1]
    ratio_all = sum(joint) / sum(union)
    polisemanticity_values_all.append(ratio_all)
    polisemanticity_values_all_poli.append(ratio_all)


    all_ratios = []
    counter = kde_arrays.shape[0]
    all_classes_combinations = [(i, j) for i in range(counter) for j in range(i + 1, counter)]
    for (i, j) in all_classes_combinations:
        kde_1 = kde_arrays[i]
        kde_2 = kde_arrays[j]
        if kde_1 is not None and kde_2 is not None:
            # Calculate the overlap between the two distributions
            joint = np.minimum(kde_1, kde_2)
            union = np.maximum(kde_1, kde_2)
            ratio = np.sum(joint) / np.sum(union)
            all_ratios.append(ratio)

    # take the max of the ratios
    max_ratio = max(all_ratios)
    polisemanticity_values_max.append(max_ratio)
    polisemanticity_values_max_poli.append(max_ratio)
    
    # calculate the generalized JS divergence
    js_distance = generalized_js_distance(kde_arrays)
    monocsemanticity_values_js.append(js_distance)
    monocsemanticity_values_js_poli.append(js_distance)
    
    
print("Average separability based on aura: ", np.mean(monocsemanticity_values_aura))
    
# print("Average polysemanticity based on overlaping between all distributions: ", np.mean(polisemanticity_values_all))
print("Average separability based on generalized JS divergence: ", np.mean(monocsemanticity_values_js))
print("Average polysemanticity based on taking max of the ratios: ", np.mean(polisemanticity_values_max))

# print("Average polysemanticity based on overlaping between all distributions on polisemantic features: ", np.mean(polisemanticity_values_all_poli))
print("Average separability based on generalized JS divergence on polisemantic features: ", np.mean(monocsemanticity_values_js_poli))
print("Average polysemanticity based on taking max of the ratios on polisemantic features: ", np.mean(polisemanticity_values_max_poli))

# print("Percentage of monosemantic features: ", monocsemanticity_counter / activated_features_counter)
print("Percentage of polisemantic features: ", polisemanticity_counter / activated_features_counter)
# print("Percentage of activated features: ", activated_features_counter / len(features))


all_freqs = [freq for freq in all_freqs if freq != {}]

# 1. Compute mean activation per feature per class
class_feature_means = []
for freq in all_freqs:
    # freq: defaultdict(list) of feature → [activation_values…]
    means = {feat: np.mean(vals) for feat, vals in freq.items()}
    class_feature_means.append(means)

# 2a. take the top-K features per class
top_k = 80  # ← choose how many “highly activated” features per class
top_sets = []
for means in class_feature_means:
    # sort features by mean activation, descending
    sorted_feats = sorted(means, key=lambda f: means[f], reverse=True)
    top_sets.append(set(sorted_feats[:top_k]))

# 3. Intersect across classes
common_topk_features   = set.intersection(*top_sets)

# 4. Union across classes
union_features = set.union(*top_sets)

overlap_percentage = len(common_topk_features) / len(union_features) if union_features else 0
print(f"Overlap percentage: {overlap_percentage}")