import numpy as np
import torch
from matplotlib import pyplot as plt
from torch_geometric.nn import GCNConv, GINConv, GATConv
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap
from copy import deepcopy
from utility import attSequential


class MyAttentionModule(torch.nn.Module): # zakladamy ze atom ma 49 featerow
    def __init__(self, gates, feats, groupFeatures=1):
        super().__init__()
        self.groupFeatures = groupFeatures
        self.gates = gates
        self.feats = feats

    def forward(self, x, edge_index, batch):
        gates = []
        gates.append(self.gates['AtomicNum'](x[:,0:12], edge_index))
        gates.append(self.gates['Degree'](x[:,12:18], edge_index))
        gates.append(self.gates['TotalNumHs'](x[:,18:23], edge_index))
        gates.append(self.gates['ImplicitValence'](x[:,23:29], edge_index))
        gates.append(self.gates['Hybridization'](x[:,29:34], edge_index))
        gates.append(self.gates['FormalCharge'](x[:,34:35], edge_index))
        gates.append(self.gates['IsInRing'](x[:,35:36], edge_index))
        gates.append(self.gates['IsAromatic'](x[:,36:37], edge_index))
        gates.append(self.gates['NumRadicalElectrons'](x[:,37:38], edge_index))
        logits = torch.cat(gates, dim=-1)
        attention = torch.softmax(logits, dim=-1).unsqueeze(-1)
        subgroups = []
        subgroups.append(self.feats['AtomicNum'](x[:,0:12]) * attention[:,0])
        subgroups.append(self.feats['Degree'](x[:,12:18]) * attention[:,1])
        subgroups.append(self.feats['TotalNumHs'](x[:,18:23]) * attention[:,2])
        subgroups.append(self.feats['ImplicitValence'](x[:,23:29]) * attention[:,3])
        subgroups.append(self.feats['Hybridization'](x[:,29:34]) * attention[:,4])
        subgroups.append(self.feats['FormalCharge'](x[:,34:35]) * attention[:,5])
        subgroups.append(self.feats['IsInRing'](x[:,35:36]) * attention[:,6])
        subgroups.append(self.feats['IsAromatic'](x[:,36:37]) * attention[:,7])
        subgroups.append(self.feats['NumRadicalElectrons'](x[:,37:38]) * attention[:,8])
        x = torch.stack(subgroups, dim=-2)
        x = torch.sum(x, dim=-2)
        
        return x, attention
    
    def optimize_parameters(self, optimizer, criterion, x, edge_index, batch, y):
        optimizer.zero_grad()
        output, attention = self.forward(x, edge_index, batch)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        return loss


#warstwa attention pooling
class MyAttentionModule3(MyAttentionModule): # zakladamy ze atom ma 49 featerow
    def __init__(self, groupFeatures=1):
        super().__init__(torch.nn.ModuleDict({ # do wyliczenia atencji dla kazdej grupy cech - jest ich 9
            'AtomicNum': GCNConv(12, 1),
            'Degree': GCNConv(6, 1),
            'TotalNumHs': GCNConv(5, 1),
            'ImplicitValence': GCNConv(6, 1),
            'Hybridization': GCNConv(5, 1),
            'FormalCharge': GCNConv(1, 1),
            'IsInRing': GCNConv(1, 1),
            'IsAromatic': GCNConv(1, 1),
            'NumRadicalElectrons': GCNConv(1, 1)
        }), torch.nn.ModuleDict({ # do transformacji grupy cech w wektor, na razie dziala tylko dla groupFeatures=1
            'AtomicNum': torch.nn.Linear(12, groupFeatures),
            'Degree': torch.nn.Linear(6, groupFeatures),
            'TotalNumHs': torch.nn.Linear(5, groupFeatures),
            'ImplicitValence': torch.nn.Linear(6, groupFeatures),
            'Hybridization': torch.nn.Linear(5, groupFeatures),
            'FormalCharge': torch.nn.Linear(1, groupFeatures),
            'IsInRing': torch.nn.Linear(1, groupFeatures),
            'IsAromatic': torch.nn.Linear(1, groupFeatures),
            'NumRadicalElectrons': torch.nn.Linear(1, groupFeatures)
        }), groupFeatures)


#warstwa attention pooling
class MyAttentionModule4(MyAttentionModule): # zakladamy ze atom ma 49 featerow
    def __init__(self, groupFeatures=1):
        super().__init__(torch.nn.ModuleDict({ # do wyliczenia atencji dla kazdej grupy cech - jest ich 9
            'AtomicNum': GINConv(attSequential(12), train_eps=True),
            'Degree': GINConv(attSequential(6), train_eps=True),
            'TotalNumHs': GINConv(attSequential(5), train_eps=True),
            'ImplicitValence': GINConv(attSequential(6), train_eps=True),
            'Hybridization': GINConv(attSequential(5), train_eps=True),
            'FormalCharge': GINConv(attSequential(1), train_eps=True),
            'IsInRing': GINConv(attSequential(1), train_eps=True),
            'IsAromatic': GINConv(attSequential(1), train_eps=True),
            'NumRadicalElectrons': GINConv(attSequential(1), train_eps=True)
        }),
        torch.nn.ModuleDict({ # do transformacji grupy cech w wektor, na razie dziala tylko dla groupFeatures=1
            'AtomicNum': torch.nn.Linear(12, groupFeatures),
            'Degree': torch.nn.Linear(6, groupFeatures),
            'TotalNumHs': torch.nn.Linear(5, groupFeatures),
            'ImplicitValence': torch.nn.Linear(6, groupFeatures),
            'Hybridization': torch.nn.Linear(5, groupFeatures),
            'FormalCharge': torch.nn.Linear(1, groupFeatures),
            'IsInRing': torch.nn.Linear(1, groupFeatures),
            'IsAromatic': torch.nn.Linear(1, groupFeatures),
            'NumRadicalElectrons': torch.nn.Linear(1, groupFeatures)
        }), groupFeatures)
    


class GraphNeuralNetwork(torch.nn.Module):  # TODO: assign hyperparameters to attributes and define the forward pass
    def __init__(self, hidden_size, n_convs=3, my_layer=None, features_after_layer=26, n_features=49, dropout=0.2, layer_type = "GCN", batch_bool = False):
        super().__init__()
        self.myAttentionModule = my_layer
        self.dropout = dropout
        self.batch_norm_bool = batch_bool

        convs = torch.nn.ModuleList()
        if layer_type == "GIN":
            convs.append(GINConv(features_after_layer, hidden_size))
        elif layer_type == "GAT":
            convs.append(GATConv(features_after_layer, hidden_size))
        else:
            convs.append(GCNConv(features_after_layer, hidden_size))

        for i in range(1, n_convs):
            if layer_type == "GIN":
                convs.append(GINConv(hidden_size, hidden_size))
            elif layer_type == "GAT":
                convs.append(GATConv(hidden_size, hidden_size))
            else:
                convs.append(GCNConv(hidden_size, hidden_size))
        self.convs = convs
        self.linear = torch.nn.Linear(hidden_size, 1)
    
    def forward(self, x, edge_index, batch):
        att = None
        if self.myAttentionModule is not None:
            x, att = self.myAttentionModule(x, edge_index, batch)
        for i in range(0, len(self.convs)-1):
            x = self.convs[i](x, edge_index)
            if self.batch_norm_bool:
                x = torch.nn.functional.batch_norm(x)

            x = x.relu()
        x = self.convs[-1](x, edge_index)
        
        x = gap(x, batch)
        
        x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.linear(x)

        return out, att
    

def train(model, train_loader, valid_loader, epochs=20, learning_rate = 0.01):
        model.train()
        
        # training loop
        optimizer = torch.optim.Adam(model.parameters(), learning_rate) # TODO: define an optimizer
        loss_fn = torch.nn.MSELoss()  # TODO: define a loss function
        for epoch in 1, epochs + 1:
            for data in train_loader:
                x, edge_index, batch, y = data.x, data.edge_index, data.batch, data.y
                model.zero_grad()
                preds, att = model(x, edge_index, batch)
                loss = loss_fn(preds, y.reshape(-1, 1))
                loss.backward()
                # print("==============")
                # for par in model.myAttentionModule.parameters():
                #     print(par)
                optimizer.step()
        return model


def predict(model, test_loader):
    # evaluation loop
    preds_batches = []
    with torch.no_grad():
        for data in test_loader:
            x, edge_index, batch = data.x, data.edge_index, data.batch
            preds, att = model(x, edge_index, batch)
            preds_batches.append(preds.cpu().detach().numpy())
    preds = np.concatenate(preds_batches)
    return preds, att

def train_best(model, train_loaders, valid_loader, rmse, epochs=20, learning_rate=0.01, seed=1, run_number=0, neptune_run=None, saveImg=False, title="", is_fine_tuning=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    torch.save(model.state_dict(), "train.pth")
    best_val = 1000000
    
    # training loop
    optimizer = torch.optim.Adam(model.parameters(), learning_rate) 
    loss_fn = torch.nn.MSELoss()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for train_loader in train_loaders:
            for data in train_loader:
                x, edge_index, batch, y = data.x.to(device), data.edge_index.to(device), data.batch.to(device), data.y.to(device)
                model.zero_grad()
                preds, att = model(x, edge_index, batch)
                loss = loss_fn(preds, y.reshape(-1, 1))

                loss.backward()
                optimizer.step()

        # evaluation loop
        preds_batches = []
        y_batches = []

        with torch.no_grad():
            for data in valid_loader:
                x, edge_index, batch, y = data.x.to(device), data.edge_index.to(device), data.batch.to(device), data.y.to(device)
                preds, att = model(x, edge_index, batch)
                loss = loss_fn(preds, y.reshape(-1, 1))
                preds_batches.append(preds.cpu().detach().numpy())
                y_batches.append(data.y)

        y_valid = np.concatenate(y_batches)
        preds = np.concatenate(preds_batches)
        mae = rmse(y_valid, preds.flatten())
        if mae < best_val:
            torch.save(model.state_dict(), "train.pth")
            best_val = mae
        if is_fine_tuning:
            neptune_run[f"fine_tuning/train_best/{run_number}_loader/epoch"].append(mae)
        else:
            neptune_run[f"train_best/{run_number}_loader/epoch"].append(mae)
    if is_fine_tuning:
        neptune_run[f"fine_tuning/train_best/{run_number}_loader/best_val"].append(mae)
    else:
        neptune_run[f"train_best/{run_number}_loader/best_val"].append(best_val)
    print("Training best val: " + str(best_val))
    model.load_state_dict(torch.load("train.pth"))
    model.eval()
    return model




def visualize(model, train_loader, valid_loader, test_loader, rmse, y_valid, epochs=20, learning_rate = 0.01, saveImg=False, title=""):
    model.train()

    best_state = deepcopy(model.state_dict())
    best_val = 1000000
    
    # training loop
    optimizer = torch.optim.Adam(model.parameters(), learning_rate) # TODO: define an optimizer
    loss_fn = torch.nn.MSELoss()  # TODO: define a loss function
    train_losses = []
    val_losses = []
    train_errors = []
    val_errors = []
    for epoch in range(1, epochs + 1):
        # preds_batches = []
        running_loss = 0.0
        for data in train_loader:
            x, edge_index, batch, y = data.x, data.edge_index, data.batch, data.y
            model.zero_grad()
            preds, att = model(x, edge_index, batch)
            loss = loss_fn(preds, y.reshape(-1, 1))
            # print(len(train_dataset))

            running_loss += loss.item()
            # preds_batches.append(preds.cpu().detach().numpy())

            loss.backward()
            optimizer.step()
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        # preds = np.concatenate(preds_batches)
        # mae = rmse(y_train, preds.flatten())
        # train_errors.append(mae)

        # evaluation loop
        preds_batches = []
        running_loss = 0.0
        with torch.no_grad():
            for data in valid_loader:
                x, edge_index, batch, y = data.x, data.edge_index, data.batch, data.y
                preds, att = model(x, edge_index, batch)
                loss = loss_fn(preds, y.reshape(-1, 1))
                # print(len(train_dataset))

                running_loss += loss.item()
                preds_batches.append(preds.cpu().detach().numpy())
        epoch_loss = running_loss / len(valid_loader)
        val_losses.append(epoch_loss)
        preds = np.concatenate(preds_batches)
        mae = rmse(y_valid, preds.flatten())
        if mae < best_val:
            best_state = deepcopy(model.state_dict())
            best_val = mae
            print(best_val)
        val_errors.append(mae)

    model.load_state_dict(best_state)

    ##### visualize ########
    plt.plot(train_losses, label='train_loss')
    plt.plot(val_losses, label='val_loss')
    plt.legend()
    plt.show()
    if saveImg:
        plt.savefig(title + "_loss.png")

    # plt.plot(train_errors,label='train_errors')
    plt.plot(val_errors, label='val_RMSE')
    plt.legend()
    plt.show()
    if saveImg:
        plt.savefig(title + "_val_error.png")
    return model


