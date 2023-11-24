import warnings
import torch
import pandas as pd
import utils as u
import os
from loader import load_data, data_to_pyg,tracker,get_random_illicit
from train import train, test
from models import models
from argparse import ArgumentParser
from models.custom_gat.model import GAT
import matplotlib.pyplot as plt
import networkx as nx


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

parser = ArgumentParser()
parser.add_argument("-d", "--data", dest="data_path", help="Path of data folder")
command_line_args = parser.parse_args()
data_path = command_line_args.data_path

print("Loading configuration from file...")
args = u.get_config()
print("Configuration loaded successfully")
print("="*50)
print("Loading graph data...")
data_path = args.data_path if data_path is None else data_path

features, edges = load_data(data_path)
features_noAgg, edges_noAgg = load_data(data_path, noAgg=True)

deep_levels = 20
keep = {}
tries = 0
minimum_nodes = 160

while(len(keep) < minimum_nodes):
  seed = get_random_illicit(features)
  keep,edgeList,counter,complete = tracker(deep_levels,seed,edges)
  tries = tries + 1

print("is the graph complete: ",complete)
print("iterations-> ",counter)
print("transactions found->",len(keep))
print("seed : ",seed)
print("seed tried:", tries)


plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)  
    
X = features_unknown[tx_features+agg_features]
preds = pd.DataFrame(clf.predict(X),columns=['class'])
features_unknown = features_unknown[['txId']+tx_features+agg_features].join(preds)
features_unknown['class'] = features_unknown['class'].apply(lambda x: 2 if x==0 else 1)
features_classified = features_known.append(features_unknown)
features_unknown['class'].hist()
features_unknown['class'].value_counts(normalize=True) * 100

plt.subplot(2, 2, 2) 
features_known['class'].hist()
features_known['class'].value_counts(normalize=True) * 100

plt.subplot(2, 2, 3)  

transaction_graph = nx.from_pandas_edgelist(edgeList,source='txId1', target='txId2')
transaction_type = features[features["txId"].isin(list(transaction_graph.nodes))]["class"].sort_index()
transaction_type = transaction_type.apply(lambda x: 'gray' if x == 0 else x)
transaction_type = transaction_type.apply(lambda x: 'red' if x == 1 else x)
transaction_type = transaction_type.apply(lambda x: 'green' if x == 2 else x)
my_pos = nx.spring_layout(transaction_graph, seed = 100)
nx.draw(transaction_graph,node_size=50, pos=my_pos,node_color=list(transaction_type),width=1)

plt.subplot(2, 2, 4)  # 设置子图位置
transaction_graph = nx.from_pandas_edgelist(edgeList,source='txId1', target='txId2')
transaction_type = features_classified[features_classified["txId"].isin(list(transaction_graph.nodes))]["class"].sort_index()
transaction_type = transaction_type.apply(lambda x: 'gray' if x == 0 else x)
transaction_type = transaction_type.apply(lambda x: 'red' if x == 1 else x)
transaction_type = transaction_type.apply(lambda x: 'green' if x == 2 else x)
my_pos = nx.spring_layout(transaction_graph, seed = 100)
nx.draw(transaction_graph,node_size=50, pos=my_pos,node_color=list(transaction_type),width=1)


plt.tight_layout()  # 调整子图的布局
plt.show()








u.seed_everything(42)

data = data_to_pyg(features, edges)
data_noAgg = data_to_pyg(features_noAgg, edges_noAgg)

print("Graph data loaded successfully")
print("="*50)
args.use_cuda = (torch.cuda.is_available() and args.use_cuda)
args.device = 'cpu'
if args.use_cuda:
    args.device = 'cuda'
print ("Using CUDA: ", args.use_cuda, "- args.device: ", args.device)

models_to_train = {
    'GCN Convolution (tx)': models.GCNConvolution(args, data_noAgg.num_features, args.hidden_units_noAgg).to(args.device),
    'GCN Convolution (tx+agg)': models.GCNConvolution(args, data.num_features, args.hidden_units).to(args.device),
    'GAT Convolution (tx)': models.GATConvolution(args, data_noAgg.num_features, args.hidden_units_noAgg).to(args.device),
    'GAT Convolution (tx+agg)': models.GATConvolution(args, data.num_features, args.hidden_units).to(args.device),
    'SAGE Convolution (tx)': models.SAGEConvolution(args, data_noAgg.num_features, args.hidden_units_noAgg).to(args.device),
    'SAGE Convolution (tx+agg)': models.SAGEConvolution(args, data.num_features, args.hidden_units).to(args.device),
    'Chebyshev Convolution (tx)': models.ChebyshevConvolution(args, [1, 2], data_noAgg.num_features, args.hidden_units_noAgg).to(args.device),
    'Chebyshev Convolution (tx+agg)': models.ChebyshevConvolution(args, [1, 2], data.num_features, args.hidden_units).to(args.device),
    'GATv2 Convolution (tx)': models.GATv2Convolution(args, data_noAgg.num_features, args.hidden_units_noAgg).to(args.device),
    'GATv2 Convolution (tx+agg)': models.GATv2Convolution(args, data.num_features, args.hidden_units).to(args.device),
    'Custom GAT': GAT(num_of_layers=3, num_heads_per_layer=[1, 4, 1],
                     num_features_per_layer=[args.num_features, args['hidden_units'],
                     args['hidden_units']//2, args['num_classes']], device=args.device).to(args.device)
}

compare_illicit = pd.DataFrame(columns=['model','Precision','Recall', 'F1', 'F1 Micro AVG'])
print("Starting training models")
print("="*50)

model_list = list(models_to_train.items())

for i in range(0, len(model_list), 2):

    (name, model) = model_list[i]
    data_noAgg = data_noAgg.to(args.device)
    print('-'*50)
    print(f"Training model: {name}")
    print('-'*50)
    train(args, model, data_noAgg)
    print('-'*50)
    print(f"Testing model: {name}")
    print('-'*50)
    test(model, data_noAgg)
    print('-'*50)
    print(f"Computing metrics for model: {name}")
    print('-'*50)
    compare_illicit = compare_illicit.append(u.compute_metrics(model, name, data_noAgg, compare_illicit), ignore_index=True)

    (name, model) = model_list[i + 1]
    data = data.to(args.device)
    print('-'*50)
    print(f"Training model: {name}")
    print('-'*50)
    train(args, model, data)
    print('-'*50)
    print(f"Testing model: {name}")
    print('-'*50)
    test(model, data)
    print('-'*50)
    print(f"Computing metrics for model: {name}")
    compare_illicit = compare_illicit.append(u.compute_metrics(model, name, data, compare_illicit), ignore_index=True)
    print('-'*50)
    

compare_illicit.to_csv(os.path.join(data_path, 'metrics.csv'), index=False)
print('Results saved to metrics.csv')

u.plot_results(compare_illicit)

u.aggregate_plot(compare_illicit)

