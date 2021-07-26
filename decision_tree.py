

import pandas as pd
import numpy as np
from functools import reduce

# from google.colab import files
# uploaded = files.upload()

def read_input(filename):
  df = pd.read_csv(filename, header = None).to_numpy()
  # df = np.random.permutation(df)
  y = df[:, [0]]
  X = df[:, range(1, df.shape[1])]
  return X, y

def get_feature_info(data):
  _, n = data.shape
  features_unique_values = []
  for i in range(n):
    features_unique_values.append(np.unique(data[:, i]))
  return features_unique_values

def calculate_entropy(values):
  entropy = 0
  _, unique_counts = np.unique(values, return_counts = True)
  if unique_counts.size == 1:
    return 0
  total_counts = np.sum(unique_counts)
  for count in unique_counts:
    frac = count / total_counts
    entropy -= (frac*np.log2(frac))
  return entropy

def select(data, map):
  if not map:
    return np.where(data[:, 0] != None)
  return np.where(reduce(lambda a, b: a & b, 
                  [data[:, i] == map[i] for i in map]))

class Node:
  def __init__(self,
               X,
               y,
               feature_value_map,
               features_unique_values,
               node_feature_index = None):
    self.X = X
    self.y = y
    self.feature_value_map = dict(feature_value_map)
    self.features_unique_values = features_unique_values
    self.node_feature_index = node_feature_index
    self.selected = select(self.X, self.feature_value_map)
    self.children = dict()
    y_selected = self.y[self.selected]
    self.y_entropy = calculate_entropy(y_selected)
    unique_labels, label_counts = np.unique(y_selected, return_counts = True)
    is_all_same_label = len(unique_labels) == 1
    no_features_left = len(feature_value_map) == len(features_unique_values)
    self.label = unique_labels[np.argmax(label_counts)]
    if not is_all_same_label and not no_features_left:
      self.create_subtrees()
      self.is_leaf = False
    else:
      self.is_leaf = True

  def get_conditional_entropy(self, feature_index):
    avg_entropy = 0
    for v in self.features_unique_values[feature_index]:
      self.feature_value_map[feature_index] = v
      y_selected = self.y[select(self.X, self.feature_value_map)]
      entropy = calculate_entropy(y_selected)
      avg_entropy += y_selected.shape[0] * entropy
    del self.feature_value_map[feature_index]
    avg_entropy /= len(self.selected[0])
    return avg_entropy

  def find_best_feature(self):
    best_index = -1
    min_entropy = float('inf')
    for feature_index in range(self.X.shape[1]):
      if feature_index in self.feature_value_map:
        continue
      entropy = self.get_conditional_entropy(feature_index)
      if entropy <= min_entropy:
        min_entropy = entropy
        best_index = feature_index
    return best_index, min_entropy
  
  def create_subtrees(self):
    self.best_index, self.min_entropy = self.find_best_feature()
    for v in self.features_unique_values[self.best_index]:
      new_feature_value_map = dict(self.feature_value_map)
      new_feature_value_map[self.best_index] = v
      # If there is no data for a particular value, don't create a child node.
      if len(select(self.X, new_feature_value_map)[0]) == 0:
        continue
      self.children[v] = Node(self.X,
                              self.y,
                              new_feature_value_map,
                              self.features_unique_values,
                              self.best_index)
      
  def print_subtree(self, prefix = ""):
    print("{}{} -- {}{}".format(prefix,
                          ("feature[{}]: {}".format(
                              self.node_feature_index,
                              self.feature_value_map[self.node_feature_index]) \
                          if self.node_feature_index else "Root"),
                          ("Label: {}".format(self.label) \
                          if self.label != None else ""),
                          " -- {}".format(self.y_entropy - self.min_entropy) \
                                          if not self.is_leaf else ""))
    if not self.children:
      return
    for v in self.children:
      self.children[v].print_subtree(prefix + "  ")

def traverse(node, x):
  if node.is_leaf:
    return node.label
  
  v = x[node.best_index]
  if v in node.children:
    return traverse(node.children[v], x)
  else:
    return node.label

def accuracy(root, X, y):
  m, n = X.shape
  counter = 0
  for i in range(m):
    y_hat = traverse(root, X[i, :])
    if y[i, 0] == y_hat:
      counter += 1
  return counter / m * 100

def main():
  X_train, y_train = read_input("mush_train.data")
  X_test, y_test = read_input("mush_test.data")
  features_unique_values = get_feature_info(X_train)
  feature_value_map = dict()
  root = Node(X_train, y_train, dict(), features_unique_values)
  root.print_subtree("")

  test_acc = accuracy(root, X_test, y_test)
  print(f"\n\nTest accuracy: {test_acc}.")

main()