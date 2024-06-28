import argparse

def parse_args(script):
    parser = argparse.ArgumentParser(description= 'Atom Attention Pooling script %s' %(script), formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='qm9.csv', help='qm9.csv/esol.csv/human_halflifetime.sdf/rat_halflifetime.sdf')
    parser.add_argument('--module', default='MyAttentionModule4', help='MyAttentionModule3/MyAttentionModule4')
    parser.add_argument('--evaluator', default='RMSE', help='RMSE/MAE')
    parser.add_argument('--y_column', default='measured log solubility in mols per litre', help='')
    parser.add_argument('--number_of_convs', type=int, default=3, nargs='+', help='3/4/5')
    parser.add_argument('--number_of_channels', type=int, default=512, nargs='+', help='64/256/512')
    parser.add_argument('--number_of_epochs', type=int, default=70, nargs='+', help='Number of epochs')
    parser.add_argument('--number_of_features_before_layer', type=int, default=25, help='')
    parser.add_argument('--number_of_features_after_layer', type=int, default=25, help='')
    parser.add_argument('--seed', type=int, default=1, help='Int that indicates seed')
    parser.add_argument('--neptune_project', default=None, help='Neptune Project Name (optional)')
    parser.add_argument('--neptune_api_key', default=None, help='Neptune API Key (optional)')
    parser.add_argument('--fine_tune', default=False, help='Bool indicating if fine tunning should be used')
    parser.add_argument('--path_to_fine_tune', default=None, help='Path to fine tunning dataset (optional, insert if fine_tune == True)')
    parser.add_argument('--y_column_fine_tune', default=None, help='(optional)')
    parser.add_argument('--type_of_layer', default=None, nargs='+', help='GIN/GAT/GCN')
    parser.add_argument('--batch_norm', type=bool, default=False, help='Bool that indicates if you want model to use batch norm (default is False)')
    parser.add_argument('--batch_size', type=int, nargs='+', default=64, help='Size of each batch in loaders')

    return parser.parse_args()