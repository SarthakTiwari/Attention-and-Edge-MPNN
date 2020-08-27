import torch
from preprocessing import smile_to_graph,molgraph_collate_fn
from  edge_memory_network import EMNImplementation



def property_prediction(smiles,Property):
    valid = {"t_half","logD","hml_clearance"}
    if Property not in valid:
       raise ValueError("property must be one of %r." % valid)


    adjacency, nodes, edges = smile_to_graph(smiles)

    adjacency, nodes, edges=molgraph_collate_fn(((adjacency, nodes, edges),))
      
    if Property=="t_half":
       model = EMNImplementation(node_features=40, edge_features=4,edge_embedding_size=50, message_passes=6, out_features=1,
                 edge_emb_depth=3, edge_emb_hidden_dim=120,
                 att_depth=3, att_hidden_dim=80,
                 msg_depth=3, msg_hidden_dim=80,
                 gather_width=100,
                 gather_att_depth=3, gather_att_hidden_dim=80,
                 gather_emb_depth=3, gather_emb_hidden_dim=80,
                 out_depth=2, out_hidden_dim=60)
       
       checkpoint = torch.load(r"checkpoints/t_half.ckpt")
       model.load_state_dict(checkpoint['state_dict'])

       model.eval()
       output = model.forward(nodes, edges,adjacency)
       return output

    
    if Property=="logD":
       model = EMNImplementation(node_features=40, edge_features=4,edge_embedding_size=50, message_passes=6, out_features=1,
                 edge_emb_depth=3, edge_emb_hidden_dim=120,
                 att_depth=3, att_hidden_dim=80,
                 msg_depth=3, msg_hidden_dim=80,
                 gather_width=100,
                 gather_att_depth=3, gather_att_hidden_dim=80,
                 gather_emb_depth=3, gather_emb_hidden_dim=80,
                 out_depth=2, out_hidden_dim=60)
       checkpoint = torch.load(r"checkpoints/logD.ckpt")
       model.load_state_dict(checkpoint['state_dict'])
       model.eval()
       output = model.forward(nodes, edges,adjacency)
       return output

    if Property=="hml_clearance":
       model = EMNImplementation(node_features=40, edge_features=4,edge_embedding_size=50, message_passes=6, out_features=1,
                 edge_emb_depth=3, edge_emb_hidden_dim=120,
                 att_depth=3, att_hidden_dim=60,
                 msg_depth=3, msg_hidden_dim=60,
                 gather_width=80,
                 gather_att_depth=3, gather_att_hidden_dim=80,
                 gather_emb_depth=3, gather_emb_hidden_dim=80,
                 out_depth=2, out_hidden_dim=60)
       checkpoint = torch.load(r"checkpoints/hml_clearance.ckpt")
       model.load_state_dict(checkpoint['state_dict'])
       model.eval()
       output = model.forward(nodes, edges,adjacency)
       return torch.exp(output)

if __name__=='__main__':

  smiles="CCOc1ccc(Nc2c(C)c(N[C@H]3CCCNC3)nc4ccnn24)cc1"
  Property="hml_clearance"   ## one of {"t_half","logD","hml_clearance"}
  
  print(float(property_prediction(smiles,Property)))