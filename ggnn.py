import pandas as pd
import random
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from preprocessing import smile_to_graph,molgraph_collate_fn
from modules import GraphGather, FeedForwardNetwork
from aggregation import SummationMPNN
from torch.utils.data import DataLoader
criterion = nn.MSELoss()


class GGNN(SummationMPNN):
    def __init__(self, node_features, edge_features, message_size, message_passes, out_features,
                 msg_depth=4, msg_hidden_dim=200, msg_dropout_p=0.0,
                 gather_width=100,
                 gather_att_depth=3, gather_att_hidden_dim=100, gather_att_dropout_p=0.0,
                 gather_emb_depth=3, gather_emb_hidden_dim=100, gather_emb_dropout_p=0.0,
                 out_depth=2, out_hidden_dim=100, out_dropout_p=0.0, out_layer_shrinkage=1.0):
        super(GGNN, self).__init__(node_features, edge_features,message_size, message_passes, out_features)


        self.msg_nns = nn.ModuleList()
        for _ in range(edge_features):
            self.msg_nns.append(
                FeedForwardNetwork(node_features, [msg_hidden_dim] * msg_depth, message_size, dropout_p=msg_dropout_p, bias=False)
            )
        self.gru = nn.GRUCell(input_size=message_size, hidden_size=node_features, bias=False)
        self.gather = GraphGather(
            node_features, gather_width,
            gather_att_depth, gather_att_hidden_dim, gather_att_dropout_p,
            gather_emb_depth, gather_emb_hidden_dim, gather_emb_dropout_p
        )
        out_layer_sizes = [
            round(out_hidden_dim * (out_layer_shrinkage ** (i / (out_depth - 1 + 1e-9)))) for i in range(out_depth)
        ]
        self.out_nn = FeedForwardNetwork(gather_width, out_layer_sizes, out_features, dropout_p=out_dropout_p)

    def message_terms(self, nodes, node_neighbours, edges):
        edges_v = edges.view(-1, self.edge_features, 1)
        node_neighbours_v = edges_v * node_neighbours.view(-1, 1, self.node_features)
        terms_masked_per_edge = [
            edges_v[:, i, :] * self.msg_nns[i](node_neighbours_v[:, i, :]) for i in range(self.edge_features)
        ]
        return sum(terms_masked_per_edge)


    def update(self, nodes, messages):
        return self.gru(messages, nodes)

    def readout(self, hidden_nodes, input_nodes, node_mask):
        graph_embeddings = self.gather(hidden_nodes, input_nodes, node_mask)
        return self.out_nn(graph_embeddings)

    def prepare_data(self):
        data = pd.read_csv('delaney-processed.csv' )
        data_list=[]
        
        for index in range(len(data)):
           adjacency, nodes, edges = smile_to_graph(data['smiles'][index])
           targets = np.expand_dims(data['measured log solubility in mols per litre'][index], axis=0)
           data_list.append(((adjacency, nodes, edges), targets))

        shuff=list(range(len(data)))
        random.shuffle(shuff)
        data_list1=[data_list[i] for i in shuff ]
        self.ggnn_train=tuple(data_list1[:900])
        self.ggnn_test=tuple(data_list1[1000:])
        self.ggnn_val=tuple(data_list1[900:1000])


    def train_dataloader(self):
        return DataLoader(self.ggnn_train, batch_size=50, shuffle=True, collate_fn=molgraph_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.ggnn_val, batch_size=10, shuffle=False, collate_fn=molgraph_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.ggnn_test, batch_size=10, shuffle=False, collate_fn=molgraph_collate_fn)
      
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1.176e-5)
        return optimizer
      
    def training_step(self,batch,batch_idx):
        adjacency, nodes, edges, target = batch
        output = self.forward(adjacency, nodes, edges)
        loss = criterion(output, target)
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self,batch,batch_idx):
        adjacency, nodes, edges, target = batch
        output = self.forward(adjacency, nodes, edges)
        loss = criterion(output, target)
        return {'val_loss':loss}

    def validation_epoch_end(self,outputs):
        avg_loss=torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs={'val_loss':avg_loss}
        return {'avg_val_loss':avg_loss,'log':tensorboard_logs}

    def test_step(self, batch, batch_idx):
        adjacency, nodes, edges, target = batch
        output = self(adjacency, nodes, edges)
        loss = criterion(output, target)
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'test_loss': avg_loss, 'log': tensorboard_logs}
    
    
if __name__=='__main__':
   model = GGNN(node_features=75, edge_features=4,message_size=25, message_passes=1, out_features=1)
   trainer = pl.Trainer(max_epochs=600)
   trainer.fit(model)
   trainer.test(ckpt_path=None)
