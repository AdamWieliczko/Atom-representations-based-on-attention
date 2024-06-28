import torch
import numpy as np
import pandas as pd
from torch_geometric.nn import global_mean_pool as gap
from attention_pooling_layer import visualize, GraphNeuralNetwork, predict, MyAttentionModule, MyAttentionModule3, MyAttentionModule4, GraphNeuralNetwork, train_best
from tdc import Evaluator
from data_extractor import GetDataCSV, GetDataSDFHuman, GetDataSDFRat
from argument_parser import parse_args
import copy
import neptune

params = parse_args('main')

rmse = Evaluator(name = params.evaluator)

#########
dataset = params.dataset

y_column = params.y_column
smiles = "smiles"
path = f"../Datasets/{dataset}"

seed = params.seed

batch_size = params.batch_size

#DOROBIĆ OBSŁUGĘ INNEGO BATCH_SIZA

if dataset in ['qm9.csv', 'esol.csv']:
    X, y, test_dataset_loader, dataset_loaders1, dataset_loaders10, dataset_loaders = GetDataCSV(path, y_column, smiles, seed, batch_size)
elif dataset in ['human_halflifetime.sdf']:
    X, y, test_dataset_loader, dataset_loaders1, dataset_loaders10, dataset_loaders = GetDataSDFHuman(path, y_column, smiles, seed, batch_size)
else:
    X, y, test_dataset_loader, dataset_loaders1, dataset_loaders10, dataset_loaders = GetDataSDFRat(path, y_column, smiles, seed, batch_size)

type_of_layer = params.type_of_layer
batch_norm = params.batch_norm

num_of_feats = params.number_of_features_before_layer
num_of_feats_after = params.number_of_features_after_layer

num_of_convs = params.number_of_convs
num_of_channels = params.number_of_channels

num_of_epochs = params.number_of_epochs

fine_tune = params.fine_tune
path_to_fine_tune = params.path_to_fine_tune
y_column_fine_tune = params.y_column_fine_tune

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
# print(y)
# y_test = y[len(dataset_loader)*0.9:]
# test_loader = dataset_loader[len(dataset_loader)*0.9:]
# dataset_loader = dataset_loader[:len(dataset_loader)*0.9]
#y_test = y[:980]

best_rmse_score = 10000

neptune_project = params.neptune_project
neptune_api_key = params.neptune_api_key

if neptune_project is not None and neptune_api_key is not None:
    neptune_run = neptune.init_run(
        project=neptune_project,
        api_token=neptune_api_key,
    )
else:
    neptune_run = None


if neptune_run is not None:
    neptune_run[f"input_data/number_of_convs"].append(num_of_convs)
    neptune_run[f"input_data/number_of_channels"].append(num_of_channels)
    neptune_run[f"input_data/batch_norm"].append(batch_norm)
    neptune_run[f"input_data/layer_type"].append(type_of_layer)


i = 0

bayesian_search_chosen_metadata = []






#PODŁĄCZYĆ DANE Z FORÓW


for channels in num_of_channels:
    for convs in num_of_convs:
        for epochs in num_of_epochs:
            for feats in num_of_feats:
                if params.module == "MyAttentionModule3":
                    layer = MyAttentionModule3(feats)
                elif params.module == "MyAttentionModule4":
                    layer = MyAttentionModule4(feats)
                for current_valid_loader in dataset_loaders:
                    print("Changing test loader")
                    copy_of_layer = copy.deepcopy(layer)
                    current_m =  GraphNeuralNetwork(hidden_size=num_of_channels, n_convs=num_of_convs, my_layer=copy_of_layer, features_after_layer=num_of_feats_after, layer_type = type_of_layer, batch_bool = batch_norm)
                    train_loaders = [loader for loader in dataset_loaders if loader != current_valid_loader]
                    current_m = train_best(current_m, train_loaders, current_valid_loader, rmse, epochs=num_of_epochs, seed=seed, neptune_run=neptune_run, run_number=i)
                    predictions, att = predict(current_m, test_dataset_loader)
                    rmse_score = rmse(y, predictions.flatten())
                    if best_rmse_score == 10000 or best_rmse_score > rmse_score:
                        best_rmse_score = rmse_score
                        m = current_m
                    i = i + 1

torch.save(m.state_dict(), "best_model.pth")

print("{:.2f}".format(best_rmse_score))

###################### atencja #################################

df_single = pd.DataFrame({"AtomicNum": [], "Degree": [], "TotalNumHs": [], "ImplicitValence": [], "Hybridization": [], "FormalCharge": [],
                          "IsInRing": [], "IsAromatic": [], "NumRadicalElectrons": []})
df_single.style.set_caption("Single")

df_batch = pd.DataFrame({"AtomicNum": [], "Degree": [], "TotalNumHs": [], "ImplicitValence": [], "Hybridization": [], "FormalCharge": [],
                          "IsInRing": [], "IsAromatic": [], "NumRadicalElectrons": []})
df_batch.style.set_caption("Batch")

preds_batches = []
with torch.no_grad():
    for data in test_dataset_loader:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        preds, att = m(x, edge_index, batch)
        preds_batches.append(preds.cpu().detach().numpy())
        att = att.squeeze()
        df_single.loc[len(df_single)] = att[0].tolist()
        df_batch.loc[len(df_single)] = torch.mean(gap(att, batch), dim=0).tolist()
preds = np.concatenate(preds_batches)

rmse_score = rmse(y, preds.flatten())
print(f'RMSE = {rmse_score:.2f}')
df_single.to_csv("model_single.csv")
df_batch.to_csv("model_batch.csv")

if neptune_run is not None:
    neptune_run[f"test/test_loader/rmse_score"].append(rmse_score)


if fine_tune == True:
    if dataset in ['qm9.csv', 'esol.csv']:
        X, y, test_dataset_loader, dataset_loaders1, dataset_loaders10, dataset_loaders = GetDataCSV(path_to_fine_tune, y_column_fine_tune, smiles, seed)
    elif dataset in ['human_halflifetime.sdf']:
        X, y, test_dataset_loader, dataset_loaders1, dataset_loaders10, dataset_loaders = GetDataSDFHuman(path_to_fine_tune, y_column_fine_tune, smiles, seed)
    else:
        X, y, test_dataset_loader, dataset_loaders1, dataset_loaders10, dataset_loaders = GetDataSDFRat(path_to_fine_tune, y_column_fine_tune, smiles, seed)
        
    i = 0
    best_rmse_score = 10000
    
    print("Start of fine tuning:")
    for current_valid_loader in dataset_loaders:
        
        print("Changing test loader")
        copy_of_layer = copy.deepcopy(layer)
        current_m =  GraphNeuralNetwork(hidden_size=num_of_channels, n_convs=num_of_convs, my_layer=copy_of_layer, features_after_layer=num_of_feats_after, layer_type = type_of_layer, batch_bool = batch_norm)
        train_loaders = [loader for loader in dataset_loaders if loader != current_valid_loader]
        current_m = train_best(current_m, train_loaders, current_valid_loader, rmse, epochs=num_of_epochs, seed=seed, neptune_run=neptune_run, run_number=i)
        predictions, att = predict(current_m, test_dataset_loader)
        rmse_score = rmse(y, predictions.flatten())
        if best_rmse_score == 10000 or best_rmse_score > rmse_score:
            best_rmse_score = rmse_score
            fine_tuned_m = current_m
        i = i + 1

    torch.save(fine_tuned_m.state_dict(), "best_fine_tuned_model.pth")

    print("{:.2f}".format(best_rmse_score))

    ###################### atencja #################################

    df_single = pd.DataFrame({"AtomicNum": [], "Degree": [], "TotalNumHs": [], "ImplicitValence": [], "Hybridization": [], "FormalCharge": [],
                            "IsInRing": [], "IsAromatic": [], "NumRadicalElectrons": []})
    df_single.style.set_caption("Single")

    df_batch = pd.DataFrame({"AtomicNum": [], "Degree": [], "TotalNumHs": [], "ImplicitValence": [], "Hybridization": [], "FormalCharge": [],
                            "IsInRing": [], "IsAromatic": [], "NumRadicalElectrons": []})
    df_batch.style.set_caption("Batch")

    preds_batches = []
    with torch.no_grad():
        for data in test_dataset_loader:
            x, edge_index, batch = data.x, data.edge_index, data.batch
            
            preds, att = m(x, edge_index, batch)
            preds_batches.append(preds.cpu().detach().numpy())
            att = att.squeeze()
            df_single.loc[len(df_single)] = att[0].tolist()
            df_batch.loc[len(df_single)] = torch.mean(gap(att, batch), dim=0).tolist()
    preds = np.concatenate(preds_batches)

    rmse_score = rmse(y, preds.flatten())
    print(f'RMSE = {rmse_score:.2f}')
    df_single.to_csv("fine_tuned_model_single.csv")
    df_batch.to_csv("fine_tuned_model_batch.csv")