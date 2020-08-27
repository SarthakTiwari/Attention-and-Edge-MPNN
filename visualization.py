import pandas as pd
import numpy as np
import torch

# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm


from io import BytesIO
from PIL import Image

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions

from captum.attr import IntegratedGradients

from preprocessing import smile_to_graph,molgraph_collate_fn
from  edge_memory_network import EMNImplementation

#model input -->  nodes, edges,adjacency, target 



def get_colors(attr, colormap):
    attr2=attr.sum(dim=1)
    vmin=-max(attr.abs().max(), 1e-16)
    vmax=max(attr.abs().max(), 1e-16)
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(attr2))

def moltopng(mol,node_colors, edge_colors, molSize=(450,150),kekulize=True):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DCairo(molSize[0],molSize[1])
    drawer.drawOptions().useBWAtomPalette()
    drawer.drawOptions().padding = .2
    drawer.DrawMolecule(
        mc,
        highlightAtoms=[i for i in range(len(node_colors))], 
        highlightAtomColors={i: tuple(c) for i, c in enumerate(node_colors)}, 
        highlightBonds=[i for i in range(len(edge_colors))],
        highlightBondColors={i: tuple(c) for i, c in enumerate(edge_colors)},
        highlightAtomRadii={i: .5 for i in range(len(node_colors))}
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def visualize_importances(feature_names, importances, title="Feature Importances", plot=True, axis_title="Features"):
    print(title)
    for i in range(len(feature_names)):
        print(feature_names[i], ": ", '%.3f'%(importances[i]))
    x_pos = (np.arange(len(feature_names)))
    if plot:
        plt.figure(figsize=(20,10))
        plt.bar(x_pos, importances, align='center')
        plt.xticks(x_pos, feature_names,rotation='vertical')
        plt.xlabel(axis_title)
        plt.title(title)

# make sure feature_list has same len as node feature len
def visualizations(model,smile,feature_list,color_map= plt.cm.bwr):
    
    model.eval()

    adjacency, nodes, edges = smile_to_graph(smile)

    mols = Chem.MolFromSmiles(smile)
    ig = IntegratedGradients(model)
    adjacency, nodes, edges=molgraph_collate_fn(((adjacency, nodes, edges),)) 
    attr= ig.attribute(nodes,additional_forward_args= (edges,adjacency),target=0 )

    attr1=torch.squeeze(attr, dim=0)
    attr2=attr1.sum(dim=1)
    vmax = max(attr2.abs().max(), 1e-16)
    vmin = -vmax

    node_colors = get_colors(attr1, color_map)
    node_colors=node_colors[:,:3]

    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)
    norm = plt.Normalize(vmin, vmax)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=color_map),cax=ax, orientation='horizontal', label='color_bar')

    b = BytesIO(); b.write(moltopng(mols, node_colors=node_colors, edge_colors={}, molSize=(600,600))); b.seek(0)
    display(Image.open(b))
    b.close()

    symbols= {i: f'{mols.GetAtomWithIdx(i).GetSymbol()}{i}' for i in range(mols.GetNumAtoms())}


    x_pos = (np.arange(len(feature_list)))
    y_pos = (np.arange(len(list(symbols.values()))))
    plt.matshow(attr1,cmap=color_map)
    plt.xticks(x_pos,feature_list,rotation='vertical')
    plt.yticks(y_pos,list(symbols.values()))
    plt.show()

    visualize_importances(list(symbols.values()), attr2)





if __name__=='__main__':

    feature_list=[
        'C',
        'N',
        'O',
        'S',
        'F',
        'Si',
        'P',
        'Cl',
        'Br',
        'I',
        'H',  # H?
        'Unknown',
        'Degree_0',
        'Degree_1',
        'Degree_2',
        'Degree_3',
        'Degree_4', 
        'Degree_5',
        'Degree_6',
        'ImplicitValence_0',
        'ImplicitValence_1',
        'ImplicitValence_2',
        'ImplicitValence_3',
        'ImplicitValence_4',
        'ImplicitValence_5',
        'ImplicitValence_6',
        'FormalCharge',
        'NumRadicalElectrons',
        'Hybridization_SP',
        'Hybridization_SP2',
        'Hybridization_SP3',
        'Hybridization_SP3D',
        'Hybridization_SP3D2',
        'atomic_mass',
        'IsAromatic',
        'NumHs_0',
        'NumHs_1',
        'NumHs_2',
        'NumHs_3',
        'NumHs_4']
    
    #Loading pretrained model 
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
    smile="CCOc1ccc(Nc2c(C)c(N[C@H]3CCCNC3)nc4ccnn24)cc1"

    # function which returns :-
    # molecule visualization with highlighted atoms
    # nodes vs feature matrix visualization
    # nodes attribution score
    visualizations(model,smile,feature_list,color_map= plt.cm.bwr)
