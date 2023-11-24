import pandas as pd
import torch
import os.path as osp
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit

def tracker(deep_levels, seed, edges):
  tmp = {seed}
  keep = set()
  counter = 0

  for i in range(deep_levels):
    for j in tmp:
      counter = counter + 1
      one = edges[edges['txId1'] == j]
      #add first
      tmp = set.union(tmp,set(one['txId2'].iloc[:]))

      two = edges[edges['txId2'] == j]
      #add second
      tmp = set.union(tmp,set(two['txId1'].iloc[:]))
      # check if connected group is empty
      if(len(one) == 0 & len(two) == 0):
          edgeList = edges[edges['txId1'].isin(keep)]
          return keep,edgeList,counter,True

      keep.add(j)
      tmp.remove(j)

  edgeList = edges[edges['txId1'].isin(keep)]
  return keep,edgeList,counter,False


def get_random_illicit(features):
  # get a random illicit transaction
  randomIllicit = features[features["class"] == 1].sample(1)["txId"]
  seed = randomIllicit.to_numpy()[0]
  # define the final DF containg the cluster
  return seed


def get_random_licit(features):
  # get a random illicit transaction
  randomLicit = features[features["class"] == 2].sample(1)["txId"]
  seed = randomLicit.to_numpy()[0]
  # define the final DF containg the cluster
  return seed

def load_data(data_path, noAgg=False):

    # Read edges, features and classes from csv files
    df_edges = pd.read_csv(osp.join(data_path, "elliptic_txs_edgelist.csv"))
    df_features = pd.read_csv(osp.join(data_path, "elliptic_txs_features.csv"), header=None)
    df_classes = pd.read_csv(osp.join(data_path, "elliptic_txs_classes.csv"))

    # Name colums basing on index
    colNames1 = {'0': 'txId', 1: "Time step"}
    colNames2 = {str(ii+2): "Local_feature_" + str(ii+1) for ii in range(94)}
    colNames3 = {str(ii+96): "Aggregate_feature_" + str(ii+1) for ii in range(72)}

    colNames = dict(colNames1, **colNames2, **colNames3)
    colNames = {int(jj): item_kk for jj, item_kk in colNames.items()}

    # Rename feature columns
    df_features = df_features.rename(columns=colNames)
    if noAgg:
        df_features = df_features.drop(df_features.iloc[:, 96:], axis = 1)

    # Map unknown class to '0'
    df_classes.loc[df_classes['class'] == 'unknown', 'class'] = '0'

    # Merge classes and features in one Dataframe
    df_feature = pd.merge(df_classes, df_features)

    # Exclude records with unknown class transaction
    df_class_feature = df_feature[df_feature["class"] != '0']
    features_unknown=   df_feature[df_feature["class"] == '0']
  
    # Build Dataframe with head and tail of transactions (edges)
    known_txs = df_class_feature["txId"].values
    df_known_edges = df_edges[(df_edges["txId1"].isin(known_txs)) & (df_edges["txId2"].isin(known_txs))]
  
    unknown_txs = features_unknown["txId"].values
    df_unkown_edges = df_edges[(df_edges["txId1"].isin(unknown_txs)) | (df_edges["txId2"].isin(unknown_txs))]

    # Build indices for features and edge types


    features_idx = {name: idx for idx, name in enumerate(sorted(df_feature["txId"].unique()))}
    class_idx = {name: idx for idx, name in enumerate(sorted(df_feature["class"].unique()))}

  
    features_unknown_idx = {name: idx for idx, name in enumerate(sorted(df_feature["txId"].unique()))}
    class_unknown_idx = {name: idx for idx, name in enumerate(sorted(df_feature["class"].unique()))}


    # Apply index encoding to features
    df_class_feature["txId"] = df_class_feature["txId"].apply(lambda name: features_idx[name])
    df_class_feature["class"] = df_class_feature["class"].apply(lambda name: class_idx[name])

    features_unknown["txId"] = features_unknown["txId"].apply(lambda name: features_unknown_idx[name])
    features_unknown["class"] = features_unknown["class"].apply(lambda name: class_unknown_idx[name])

    # Apply index encoding to edges
    df_known_edges["txId1"] = df_known_edges["txId1"].apply(lambda name: features_idx[name])
    df_known_edges["txId2"] = df_known_edges["txId2"].apply(lambda name: features_idx[name])

     df_unkown_edges["txId1"] = df_unkown_edges["txId1"].apply(lambda name: features_unknown_idx[name])
    df_unkown_edges["txId2"] = df_unkown_edges["txId2"].apply(lambda name: features_unknown_idx[name])
    
    return df_class_feature, features_unknown, df_edges,df_known_edges, df_unkown_edges


def data_to_pyg(df_class_feature, df_edges):

    # Define PyTorch Geometric data structure with Pandas dataframe values
    edge_index = torch.tensor([df_edges["txId1"].values,
                            df_edges["txId2"].values], dtype=torch.long)
    x = torch.tensor(df_class_feature.iloc[:, 3:].values, dtype=torch.float)
    y = torch.tensor(df_class_feature["class"].values, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    data = RandomNodeSplit(num_val=0.15, num_test=0.3)(data)

    return data
