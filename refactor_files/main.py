import torch
import numpy as np
import pandas as pd
from torch_geometric.nn import global_mean_pool as gap
from attention_pooling_layer import visualize, GraphNeuralNetwork, predict, MyAttentionModule, MyAttentionModule3, MyAttentionModule4, GraphNeuralNetwork, train_best
from tdc import Evaluator
from data_extractor import GetDataCSV, GetDataSDFHuman, GetDataSDFRat
from argument_parser import parse_args


params = parse_args('main')

rmse = Evaluator(name = params.evaluator)

#########
dataset = params.dataset

y_column = params.y_column
smiles = "smiles"
path = f"../Datasets/{dataset}"

if dataset in ['qm9.csv', 'esol.csv']:
    X_train, y_train, X_valid, y_valid, X_test, y_test, train_loader1, valid_loader1, test_loader1, train_loader10, valid_loader10, test_loader10, train_loader, valid_loader, test_loader = GetDataCSV(path, y_column, smiles)
elif dataset in ['human_halflifetime.sdf']:
    X_train, y_train, X_valid, y_valid, X_test, y_test, train_loader1, valid_loader1, test_loader1, train_loader10, valid_loader10, test_loader10, train_loader, valid_loader, test_loader = GetDataSDFHuman(path, y_column, smiles)
else:
    X_train, y_train, X_valid, y_valid, X_test, y_test, train_loader1, valid_loader1, test_loader1, train_loader10, valid_loader10, test_loader10, train_loader, valid_loader, test_loader = GetDataSDFRat(path, y_column, smiles)

num_of_feats = params.number_of_features_before_layer
num_of_feats_after = params.number_of_features_after_layer

num_of_convs = params.number_of_convs
num_of_channels = params.number_of_channels

num_of_epochs = params.number_of_epochs

if params.module == "MyAttentionModule":
    layer = MyAttentionModule(num_of_feats)
elif params.module == "MyAttentionModule3":
    layer = MyAttentionModule3(num_of_feats)
elif params.module == "MyAttentionModule4":
    layer = MyAttentionModule4(num_of_feats)

#

# m = GraphNeuralNetwork(512, my_layer=MyAttentionModule4(3), features_after_layer=3)
# visualize(m, train_loader, valid_loader, test_loader, rmse, y_valid, epochs=100, saveImg=True, title="MyAttentionModule4(3)_esol")

# m = GraphNeuralNetwork(512, my_layer=MyAttentionModule4(35), features_after_layer=35)
# visualize(m, train_loader, valid_loader, test_loader, rmse, y_valid, epochs=100, saveImg=True, title="MyAttentionModule4(35)_esol")

# m = GraphNeuralNetwork(512, my_layer=MyAttentionModule4(100), features_after_layer=100)
# visualize(m, train_loader, valid_loader, test_loader, rmse, y_valid, epochs=100, saveImg=True, title="MyAttentionModule4(100)_esol")




# ######################## tabelka ##########################################################
# df = pd.DataFrame({"Repr 1": [], "Repr 10": [],
#                    "Atention Pooling v2 - size = 3": [], "Atention Pooling v2 - size = 35": [], "Atention Pooling v2 - size = 100": []})
# pd.set_option("display.precision", 2)
# n_times = 1

# for n_convs in [1, 3, 5]:
#     for n_channels in [64, 512]:
#         row = []

#         #########################
#         scores = []
#         for _ in range(n_times):
#             m =  GraphNeuralNetwork(n_channels, n_convs=n_convs, features_after_layer=26)
#             m = train_best(m, train_loader1, valid_loader1, rmse, y_valid, epochs=70)
#             predictions, att = predict(m, test_loader1)
#             rmse_score = rmse(y_test, predictions.flatten())
#             scores.append("{:.2f}".format(rmse_score))
#         row.append(" | ".join(scores))

#         #########################
#         scores = []
#         for _ in range(n_times):
#             m =  GraphNeuralNetwork(n_channels, n_convs=n_convs, features_after_layer=25)
#             m = train_best(m, train_loader10, valid_loader10, rmse, y_valid, epochs= 70)
#             predictions, att = predict(m, test_loader10)
#             rmse_score = rmse(y_test, predictions.flatten())
#             scores.append("{:.2f}".format(rmse_score))
#         row.append(" | ".join(scores))

#         #########################
#         for vect_size in [3, 35, 100]:
#             scores = []
#             for _ in range(n_times):
#                 m =  GraphNeuralNetwork(n_channels, n_convs=n_convs, my_layer=MyAttentionModule4(vect_size), features_after_layer=vect_size)
#                 m = train_best(m, train_loader, valid_loader, rmse, y_valid, epochs=70)
#                 predictions, att = predict(m, test_loader)
#                 rmse_score = rmse(y_test, predictions.flatten())
#                 scores.append("{:.2f}".format(rmse_score))
#             row.append(" | ".join(scores))

#         df.loc[str(n_convs) + " convs, " + str(n_channels) + " channels"] = row

# df.to_csv("esol_out.csv")



m =  GraphNeuralNetwork(hidden_size=num_of_channels, n_convs=num_of_convs, my_layer=layer, features_after_layer=num_of_feats_after)
m = train_best(m, train_loader, valid_loader, rmse, y_valid, epochs=num_of_epochs)
predictions, att = predict(m, test_loader)
rmse_score = rmse(y_test, predictions.flatten())
print("{:.2f}".format(rmse_score))

###################### atencja #################################

df_single = pd.DataFrame({"AtomicNum": [], "Degree": [], "TotalNumHs": [], "ImplicitValence": [], "Hybridization": [], "FormalCharge": [],
                          "IsInRing": [], "IsAromatic": [], "NumRadicalElectrons": []})
df_single.style.set_caption("Hello World")

df_batch = pd.DataFrame({"AtomicNum": [], "Degree": [], "TotalNumHs": [], "ImplicitValence": [], "Hybridization": [], "FormalCharge": [],
                          "IsInRing": [], "IsAromatic": [], "NumRadicalElectrons": []})
df_batch.style.set_caption("Hello World")

preds_batches = []
with torch.no_grad():
    for data in test_loader:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        preds, att = m(x, edge_index, batch)
        preds_batches.append(preds.cpu().detach().numpy())
        att = att.squeeze()
        df_single.loc[len(df_single)] = att[0].tolist()
        df_batch.loc[len(df_single)] = torch.mean(gap(att, batch), dim=0).tolist()
preds = np.concatenate(preds_batches)

rmse_score = rmse(y_test, predictions.flatten())
print(f'RMSE = {rmse_score:.2f}')
df_single.to_csv("esol_att_single.csv")
df_batch.to_csv("esol_att_batch.csv")
