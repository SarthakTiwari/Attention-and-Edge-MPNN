import pandas as pd
import numpy as np
import random
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from preprocessing import smile_to_graph,molgraph_collate_fn
from modules import GraphGather, FeedForwardNetwork
from emn_aggregation import EMN
from torch.utils.data import DataLoader
criterion = nn.MSELoss()

# commented lines for classification tasks

# criterion = nn.BCELoss()
# from pytorch_lightning.metrics.classification import AUROC
# from tensorflow.keras.metrics import AUC
# m = nn.Sigmoid()
# from pytorch_lightning.metrics.functional import accuracy
# metric = AUC(multi_label=True)

torch.manual_seed(1234)
np.random.seed(1234)


class EMNImplementation(EMN):

    def __init__(self, node_features, edge_features, message_passes, out_features,
                 edge_embedding_size,
                 edge_emb_depth=3, edge_emb_hidden_dim=150, edge_emb_dropout_p=0.0,
                 att_depth=3, att_hidden_dim=80, att_dropout_p=0.0,
                 msg_depth=3, msg_hidden_dim=80, msg_dropout_p=0.0,
                 gather_width=100,
                 gather_att_depth=3, gather_att_hidden_dim=100, gather_att_dropout_p=0.0,
                 gather_emb_depth=3, gather_emb_hidden_dim=100, gather_emb_dropout_p=0.0,
                 out_depth=2, out_hidden_dim=100, out_dropout_p=0, out_layer_shrinkage=1.0):
        super(EMNImplementation, self).__init__(
            edge_features, edge_embedding_size, message_passes, out_features
        )
        self.embedding_nn = FeedForwardNetwork(
            node_features * 2 + edge_features, [edge_emb_hidden_dim] * edge_emb_depth, edge_embedding_size, dropout_p=edge_emb_dropout_p
        )

        self.emb_msg_nn = FeedForwardNetwork(
            edge_embedding_size, [msg_hidden_dim] * msg_depth, edge_embedding_size, dropout_p=msg_dropout_p
        )
        self.att_msg_nn = FeedForwardNetwork(
            edge_embedding_size, [att_hidden_dim] * att_depth, edge_embedding_size, dropout_p=att_dropout_p
        )


        self.gru = nn.GRUCell(edge_embedding_size, edge_embedding_size, bias=False)
        self.gather = GraphGather(
            edge_embedding_size, gather_width,
            gather_att_depth, gather_att_hidden_dim, gather_att_dropout_p,
            gather_emb_depth, gather_emb_hidden_dim, gather_emb_dropout_p
        )
        out_layer_sizes = [ 
            round(out_hidden_dim * (out_layer_shrinkage ** (i / (out_depth - 1 + 1e-9)))) for i in range(out_depth)
        ]
        self.out_nn = FeedForwardNetwork(gather_width, out_layer_sizes, out_features, dropout_p=out_dropout_p)

    def preprocess_edges(self, nodes, node_neighbours, edges):
        cat = torch.cat([nodes, node_neighbours, edges], dim=1)
        return torch.tanh(self.embedding_nn(cat))

    def propagate_edges(self, edges, ingoing_edge_memories, ingoing_edges_mask):
        BIG_NEGATIVE = -1e6
        energy_mask = ((1 - ingoing_edges_mask).float() * BIG_NEGATIVE).unsqueeze(-1)

        cat = torch.cat([edges.unsqueeze(1), ingoing_edge_memories], dim=1)
        embeddings = self.emb_msg_nn(cat)

        edge_energy = self.att_msg_nn(edges)
        ing_memory_energies = self.att_msg_nn(ingoing_edge_memories) + energy_mask
        energies = torch.cat([edge_energy.unsqueeze(1), ing_memory_energies], dim=1)
        attention = torch.softmax(energies, dim=1)

        message = (attention * embeddings).sum(dim=1)
        return self.gru(message) 

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
        self.emn_train=tuple(data_list1[:900])
        self.emn_test=tuple(data_list1[1000:])
        self.emn_val=tuple(data_list1[900:1000])


    def train_dataloader(self):
        return DataLoader(self.emn_train, batch_size=50, shuffle=True, collate_fn=molgraph_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.emn_val, batch_size=10, shuffle=False, collate_fn=molgraph_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.emn_test, batch_size=10, shuffle=False, collate_fn=molgraph_collate_fn)
      
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
      
    def training_step(self,batch,batch_idx):
        adjacency, nodes, edges, target = batch
        output = self.forward(nodes, edges,adjacency)
        # output= m(output)
        loss = criterion(output, target)
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self,batch,batch_idx):
        adjacency, nodes, edges, target = batch
        output = self.forward(nodes, edges,adjacency)
        loss = criterion(output, target)
        return {'val_loss':loss}

    def validation_epoch_end(self,outputs):
        avg_loss=torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs={'val_loss':avg_loss}
        return {'avg_val_loss':avg_loss,'log':tensorboard_logs}

    def test_step(self, batch, batch_idx):
        adjacency, nodes, edges, target = batch
        output = self(nodes, edges,adjacency)
        #output=m(output)
        #output=(output>=0.5).float()

        #auroc = metric(output, target).numpy()
        #acc=accuracy(output, target)
        
        loss = criterion(output, target)
        return {'test_loss': loss }# {'test_auroc': auroc,'accu':acc}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        #avg_auroc = sum([x['test_auroc'] for x in outputs])/len([x['test_auroc'] for x in outputs])
        #avg_acc = sum([x['accu'] for x in outputs])/len([x['accu'] for x in outputs])
        #logs = {'test_auroc': avg_auroc}
        #print(avg_acc)
        tensorboard_logs = {'test_loss': avg_loss}
        return {'test_loss': avg_loss, 'log': tensorboard_logs} #{'test_auroc': avg_auroc, 'log': logs}
    
    
if __name__=='__main__':
   model = EMNImplementation(node_features=75, edge_features=4,edge_embedding_size=50, message_passes=8, out_features=1)
   trainer = pl.Trainer(max_epochs=600)
   trainer.fit(model)
   trainer.test(model=model)
