from sklearn.datasets import load_wine 
import pandas as pd
import numpy as np
import seaborn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import norm
data = load_wine()

df_data = pd.DataFrame(data=data["data"], columns = data["feature_names"])
pd.set_option('display.max_columns', None)
df_data["target"] = data["target"]

class DecisionTree():
    def __init__(self,max_depth=None, min_samples_split=2):
        self.max_depth = max_depth 
        self.min_samples_split = min_samples_split 
        self.model = None
    
    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.Series, list)):
            y = np.array(y)
            
        self.n_features = X.shape[1]
        self.model = self.make_tree(X, y, depth_now=0)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        predictions = []
        for sample in X:
            predictions.append(self.walk_tree(sample, self.model))
        return predictions
    
    

    def gini_index(self, y_subset):
        total = len(y_subset)
        if total == 0:
            return 0
        label_counts = {}
        for item in y_subset:
            if item in label_counts:
                label_counts[item] += 1
            else:
                label_counts[item] = 1

        score = 0
        for label in label_counts:
            prob = label_counts[label] / total
            score += prob * prob
        return 1 - score

    def make_tree(self, X_arr, y_arr, depth_now):
        n_samples = X_arr.shape[0]
        distinct_labels = len(np.unique(y_arr))

        if n_samples < self.min_samples_split or distinct_labels == 1 or (self.max_depth is not None and depth_now >= self.max_depth):
            return {"leaf": self.majority(y_arr)}



        best_feature = None
        best_thresh = None
        best_val = float("inf")
        best_indices = None
        nS, nF = X_arr.shape

        for feat in range(nF):
            feat_vals = X_arr[:, feat]
            sorted_vals = np.sort(np.unique(feat_vals))
            if len(sorted_vals) == 1:
                continue
            threshold_options = []
            for i in range(len(sorted_vals)-1):
                threshold_options.append((sorted_vals[i] + sorted_vals[i+1]) / 2)
            
            
            
            for thresh in threshold_options:
                left_idx = []
                right_idx = []
                for i in range(len(feat_vals)):
                    if feat_vals[i] < thresh:
                        left_idx.append(i)
                        
                        
                    else:
                        right_idx.append(i)
                if len(left_idx) == 0 or len(right_idx) == 0:
                    continue
                left_gini = self.gini_index(y_arr[left_idx])
                right_gini = self.gini_index(y_arr[right_idx])
                comb_gini = (len(left_idx)/nS)*left_gini + (len(right_idx)/nS)*right_gini

                if comb_gini < best_val:
                    best_val = comb_gini
                    best_feature = feat
                    best_thresh = thresh
                    best_indices = (left_idx, right_idx)

        if best_feature is None:
            return {"leaf": self.majority(y_arr)}



        left_tree = self.make_tree(X_arr[best_indices[0], :], y_arr[best_indices[0]], depth_now+1)
        right_tree = self.make_tree(X_arr[best_indices[1], :], y_arr[best_indices[1]], depth_now+1)

        return {"feature": best_feature,
                "threshold": best_thresh,
                "majority": self.majority(y_arr),
                "left": left_tree,
                "right": right_tree}

    def majority(self, y_subset):
        counts = {}
        for element in y_subset:
            counts[element] = counts.get(element, 0) + 1
        major = None
        max_count = -1
        for lab in counts:
            if counts[lab] > max_count:
                max_count = counts[lab]
                major = lab
        return major

    def walk_tree(self, sample, node):
        if "leaf" in node:
            return node["leaf"]
        
        f_index = node["feature"]
        threshold_value = node["threshold"]

        if sample[f_index] < threshold_value:
            return self.walk_tree(sample, node["left"])
        else:
            return self.walk_tree(sample, node["right"])

    def prune(self, X_val, y_val):
        self.model = self._prune_node(self.model, X_val, y_val)

    def _prune_node(self, node, X_val, y_val):
        if "leaf" in node:
            return node

        feat = node["feature"]
        thresh = node["threshold"]
        left_idx = np.where(X_val[:, feat] < thresh)[0]
        right_idx = np.where(X_val[:, feat] >= thresh)[0]
        X_left, y_left = X_val[left_idx], y_val[left_idx]
        X_right, y_right = X_val[right_idx], y_val[right_idx]

        if X_left.shape[0] > 0:
            node["left"] = self._prune_node(node["left"], X_left, y_left)
        if X_right.shape[0] > 0:
            node["right"] = self._prune_node(node["right"], X_right, y_right)

        subtree_preds = [self.walk_tree(x, node) for x in X_val]
        subtree_acc = np.mean(np.array(subtree_preds) == y_val)

        pruned_pred = node["majority"]
        pruned_preds = [pruned_pred for _ in range(len(y_val))]
        pruned_acc = np.mean(np.array(pruned_preds) == y_val)

        if pruned_acc >= subtree_acc:
            return {"leaf": node["majority"]}
        else:
            return node
    
    
    def custom_prune(self, X_train, y_train, alpha):
        
        # np array
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, (pd.Series, list)):
            y_train = np.array(y_train)
        self.model = self._custom_prune_node(self.model, X_train, y_train, alpha)

    def _custom_prune_node(self, node, X_data, y_data, alpha):
        if "leaf" in node:
            return node
        
        feat = node["feature"]
        thresh = node["threshold"]
        
        left_indices = np.where(X_data[:, feat] < thresh)[0]
        right_indices = np.where(X_data[:, feat] >= thresh)[0]

        fvals = X_data[:, feat]
        mu = np.mean(fvals)
        std = np.std(fvals, ddof=0)
        if std == 0: # division by zero error???
            p_left = 0.5
        else:
            p_left = norm.cdf((thresh - mu) / std)
        p_right = 1 - p_left
        
        if len(left_indices) > 0:
            node["left"] = self._custom_prune_node(node["left"], X_data[left_indices], y_data[left_indices], alpha)
        if len(right_indices) > 0:
            node["right"] = self._custom_prune_node(node["right"], X_data[right_indices], y_data[right_indices], alpha)
        
        if min(p_left, p_right) < alpha:
            return {"leaf": self.majority(y_data)}
        else:
            return node

if __name__ == "__main__":
    features = df_data.columns[:-1]
    X_full = df_data[features]
    y_full = df_data["target"]
    X_trainval, X_test, y_trainval, y_test = train_test_split(X_full, y_full, test_size=0.3, random_state=50)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=50)
    tree_model = DecisionTree(max_depth=16, min_samples_split=8)
    tree_model.fit(X_train, y_train)

    test_preds = tree_model.predict(X_test)
    acc_before = np.mean(np.array(test_preds) == y_test.values)
    print("Test accuracy before pruning:", acc_before)

    X_val_np = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
    y_val_np = y_val.values if isinstance(y_val, pd.Series) else y_val
    tree_model.prune(X_val_np, y_val_np)
    test_preds = tree_model.predict(X_test)
    test_accuracy = np.mean(np.array(test_preds) == y_test.values)
    print("Test accuracy after reduced errror pruning:", test_accuracy)




    alphas = [0.05, 0.1, 0.2]
    for a in alphas:
        custom_pruning_tree_model = DecisionTree(max_depth=16, min_samples_split=8)
        custom_pruning_tree_model.fit(X_train, y_train)
        custom_pruning_tree_model.custom_prune(X_train, y_train, alpha=a)
        test_preds = custom_pruning_tree_model.predict(X_test)
        test_acc = np.mean(np.array(test_preds) == y_test.values)
        print("Test accuracy after custom pruning with α =", a, "is", test_acc)