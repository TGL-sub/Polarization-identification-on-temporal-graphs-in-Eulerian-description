import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from utils.utils import RandEdgeSampler
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm as cmx
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import pandas as pd
import networkx as nx
import community as community_louvain
import tqdm

def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
  # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
  # negatives for validation / test set)
  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  val_ap, val_auc = [], []
  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    for k in range(num_test_batch):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      sources_batch = data.sources[s_idx:e_idx]
      destinations_batch = data.destinations[s_idx:e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

      size = len(sources_batch)
      _, negative_samples = negative_edge_sampler.sample(size)

      pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                            negative_samples, timestamps_batch,
                                                            edge_idxs_batch, n_neighbors)

      pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
      true_label = np.concatenate([np.ones(size), np.zeros(size)])

      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))

  return np.mean(val_ap), np.mean(val_auc)


def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors):
  pred_prob = np.zeros(len(data.sources))
  num_instance = len(data.sources)
  num_batch = math.ceil(num_instance / batch_size)

  with torch.no_grad():
    decoder.eval()
    tgn.eval()
    for k in range(num_batch):
      s_idx = k * batch_size
      e_idx = min(num_instance, s_idx + batch_size)

      sources_batch = data.sources[s_idx: e_idx]
      destinations_batch = data.destinations[s_idx: e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = edge_idxs[s_idx: e_idx]

      source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                   destinations_batch,
                                                                                   destinations_batch,
                                                                                   timestamps_batch,
                                                                                   edge_idxs_batch,
                                                                                   n_neighbors)
      pred_prob_batch = decoder(source_embedding).sigmoid()
      pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()

  auc_roc = roc_auc_score(data.labels, pred_prob)
  return auc_roc

def eval_embedding(model, data, n_neighbors, time_step, batch_size=200, anomaly = False, neg = False, grads = False):
  rand_sampler = RandEdgeSampler(data.sources, data.destinations, seed=0)
  #with torch.no_grad():
    #model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
  TEST_BATCH_SIZE = batch_size
  num_test_instance = len(data.sources)
  num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

  embeddings = []
  memory= []
  if anomaly:
    anomaly_edges = []
  if neg:
    neg_edges = []
    neg_edges_prob = []
  if grads:
    grads_edges = []

  nodes = np.arange(model.n_nodes)
  step = 0

  embeddings.append((step, 0, model.embedding_module.compute_embedding(memory=model.memory.get_memory(list(range(model.n_nodes))),
                                            source_nodes=nodes,
                                            timestamps=np.full((1,nodes.shape[0]), max(model.memory.get_last_update(list(range(model.n_nodes)))))[0],
                                            n_layers=model.n_layers,
                                            n_neighbors=model.n_neighbors).detach().numpy()))
  memory.append((step, 0, model.memory.get_memory(list(range(model.n_nodes))).detach().numpy()))
  step += 1


    #----------------------- Reliability module -------------------------------

  criterion = torch.nn.BCELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
  pos_label = torch.ones(1, dtype=torch.float)
  layers=[x for x in model.parameters()]
  for l in range(2,14):
    layers[l].requires_grad = False

    #------------------------------------------------------

  for k in range(num_test_batch):
    s_idx = k * TEST_BATCH_SIZE
    e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
    sources_batch = data.sources[s_idx:e_idx]
    destinations_batch = data.destinations[s_idx:e_idx]
    timestamps_batch = data.timestamps[s_idx:e_idx]
    edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

    """model.compute_temporal_embeddings_exp(sources_batch, destinations_batch,
                                                            timestamps_batch, edge_idxs_batch, n_neighbors)"""
    size = len(sources_batch)
    _, negatives_batch = rand_sampler.sample(size)
    pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                            negatives_batch, timestamps_batch,
                                                            edge_idxs_batch, n_neighbors)
    if anomaly:
      anomaly_edges_scores = 1 - pos_prob.detach().numpy()
      anomaly_edges.extend(anomaly_edges_scores.tolist())
      
    if neg:
      neg_edges.extend(negatives_batch.tolist())
      neg_edges_prob.extend(neg_prob.detach().numpy().tolist())
      
      #------------------------------------------------------
    if grads:
      for i in range(e_idx-s_idx):
        optimizer.zero_grad()
        #print(pos_prob.squeeze()[i])
        #print(pos_label[0])
        loss = criterion(pos_prob.squeeze()[i], pos_label[0])
        loss.backward(retain_graph=True)
        gradients = 0
        gradients += np.linalg.norm(layers[0].grad.view(-1).detach().numpy())
        gradients += np.linalg.norm(layers[1].grad.view(-1).detach().numpy())
        for l in range(10,24):
          if layers[l].grad != None:
            gradients += np.linalg.norm(layers[l].grad.view(-1).detach().numpy())
        #gradients = torch.cat(gradients)
        grads_edges.append(gradients)
        print(i)
      model.memory.detach_memory()
      print(k, '/', num_test_batch)
      #------------------------------------------------------
        
    if step*time_step <= data.timestamps[s_idx]:# and data.timestamps[s_idx] < (step+1)*time_step:
        
      time = max(model.memory.get_last_update(list(range(model.n_nodes))))
      embeddings.append((step, time, model.embedding_module.compute_embedding(memory=model.memory.get_memory(list(range(model.n_nodes))),
                                            source_nodes=nodes, timestamps=np.full(nodes.shape[0], time),
                                            n_layers=model.n_layers, n_neighbors=model.n_neighbors).detach().numpy()))
      memory.append((step, time, model.memory.get_memory(list(range(model.n_nodes))).detach().numpy()))
      print("Etape ", step," sur ", int(data.timestamps[-1]/time_step))
      step += 1
        
  last_memory, _ = model.get_updated_memory(list(range(model.n_nodes)), model.memory.messages)

  embeddings.append((step, data.timestamps[-1], model.embedding_module.compute_embedding(memory=last_memory, source_nodes=nodes,
                                            timestamps=np.full(nodes.shape[0], data.timestamps[-1]),
                                            n_layers=model.n_layers, n_neighbors=model.n_neighbors).detach().numpy()))
  memory.append((step, data.timestamps[-1], last_memory.detach().numpy()))

  output = [embeddings, memory]
  if anomaly:
    output.append(anomaly_edges)
  if neg:
    output.append(neg_edges)
    output.append(neg_edges_prob)
  if grads:
    output.append(grads_edges)
  return output

def eval_dynamic(model, components, data, n_neighbors, time_step, batch_size=200,
        values = [-1,1,-1,1], step_arrow = 0.1, threshold_tile = 10, clip_value = 10, plt_emb = False, nb_com = 1):
  rand_sampler = RandEdgeSampler(data.sources, data.destinations, seed=0)
  #with torch.no_grad():
    #model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
  TEST_BATCH_SIZE = batch_size
  num_test_instance = len(data.sources)
  num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

  X_grid = np.arange(values[0], values[1], step_arrow)
  Y_grid = np.arange(values[2], values[3], step_arrow)

  arrows_list = []

  nodes = np.arange(model.n_nodes)
  step = 0
  last_time = 0

  df_graph = pd.DataFrame(np.vstack((data.sources, data.destinations)).T)
  graph = nx.from_pandas_edgelist(df_graph, source=df_graph.keys()[0], target=df_graph.keys()[1], edge_attr=None)
  partition = community_louvain.best_partition(graph, resolution = 1)

  embedding_last = model.embedding_module.compute_embedding(memory=model.memory.get_memory(list(range(model.n_nodes))),
                                            source_nodes=nodes,
                                            timestamps=np.full((1,nodes.shape[0]), max(model.memory.get_last_update(list(range(model.n_nodes)))))[0],
                                            n_layers=model.n_layers,
                                            n_neighbors=model.n_neighbors).detach().numpy()@components.T
  #memory_last = model.memory.get_memory(list(range(model.n_nodes))).detach().numpy()
  if plt_emb:
    print("Embeddings at time ", last_time)
    affichage_resultats(embedding_last[1:], partition, nb_com, n_size = 1, text = True)
  step += 1

  for k in range(num_test_batch):
    s_idx = k * TEST_BATCH_SIZE
    e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
    sources_batch = data.sources[s_idx:e_idx]
    destinations_batch = data.destinations[s_idx:e_idx]
    timestamps_batch = data.timestamps[s_idx:e_idx]
    edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

    size = len(sources_batch)
    _, negatives_batch = rand_sampler.sample(size)
    pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                            negatives_batch, timestamps_batch,
                                                            edge_idxs_batch, n_neighbors)
    
    if step*time_step <= data.timestamps[s_idx]:# and data.timestamps[s_idx] < (step+1)*time_step:
        
      time = max(model.memory.get_last_update(list(range(model.n_nodes))))
      embedding = model.embedding_module.compute_embedding(memory=model.memory.get_memory(list(range(model.n_nodes))),
                                            source_nodes=nodes, timestamps=np.full(nodes.shape[0], time),
                                            n_layers=model.n_layers, n_neighbors=model.n_neighbors).detach().numpy()@components.T
      #memory = model.memory.get_memory(list(range(model.n_nodes))).detach().numpy()
      print("Etape ", step," sur ", int(data.timestamps[-1]/time_step))
      print("Evolutions between embeddings at time ",time, " and at time ", last_time, ".")
      list_persist_delta_on_last = {}
      for node in nodes:
        list_persist_delta_on_last[node] = embedding[node] - embedding_last[node]
      coord_b, coord_e = arrows(list_persist_delta_on_last, embedding_last, nodes, values, step_arrow, threshold_tile, clip_value, X_grid, Y_grid)
      arrows_list.append((coord_b, coord_e))
      embedding_last = embedding
      if plt_emb:
        print("Embeddings at time ", time)
        affichage_resultats(embedding_last[1:], partition, nb_com, n_size = 1)
      #memory_last = memory
      last_time = time
      step += 1
    
  last_memory, _ = model.get_updated_memory(list(range(model.n_nodes)), model.memory.messages)

  embedding = model.embedding_module.compute_embedding(memory=last_memory, source_nodes=nodes,
                                            timestamps=np.full(nodes.shape[0], data.timestamps[-1]),
                                            n_layers=model.n_layers, n_neighbors=model.n_neighbors).detach().numpy()@components.T
  #memory = last_memory.detach().numpy()

  list_persist_delta_on_last = {}
  for node in nodes:
    list_persist_delta_on_last[node] = embedding[node] - embedding_last[node]

  coord_b, coord_e = arrows(list_persist_delta_on_last, embedding_last, nodes, values, step_arrow, threshold_tile, clip_value, X_grid, Y_grid)
  arrows_list.append((coord_b, coord_e))
  if plt_emb:
    print("Final embeddings")
    affichage_resultats(embedding[1:], partition, nb_com, n_size = 1, text = True)

  return arrows_list

def arrows(list_persist_delta_on_last, proj, nodes, values, step, threshold_tile, clip_value, X_grid, Y_grid):
  ### NODES COUNTING PER TILE
  count_grid = np.zeros((len(X_grid), len(Y_grid)))
  for node in nodes:
      x, y = proj[node]
      if x < values[1] and x > values[0] and y < values[3] and y > values[2]:
          x_pos = int((x - values[0])//step)
          y_pos = int((y - values[2])//step)
          count_grid[x_pos,y_pos]+=1

    #ax = sns.heatmap(count_grid, linewidth=0.5)
    #plt.show()

    ### SELECTING TILES WITH ENOUGH NODES
    
  list_x_pos_to_use = []
  list_y_pos_to_use = []
  for x_pos in range(len(count_grid)):
      for y_pos in range(len(count_grid[0])):
          if count_grid[x_pos, y_pos]> threshold_tile:
              list_x_pos_to_use.append(x_pos)
              list_y_pos_to_use.append(y_pos)

    ### CREATING THE VALUES TO PLOT
  list_delta_grid = [[[] for _ in range(len(count_grid[0]))] for _ in range(len(count_grid))]
  for node in nodes:
      x, y = proj[node]
      if x < values[1] and x > values[0] and y < values[3] and y > values[2]:        
          x_pos = int((x - values[0])//step)
          y_pos = int((y - values[2])//step)
          if x_pos in list_x_pos_to_use and y_pos in list_y_pos_to_use:            
              list_delta_grid[x_pos][y_pos].append(list_persist_delta_on_last[node])
  for x_pos in range(len(count_grid)):
      for y_pos in range(len(count_grid[0])):
          if len(list_delta_grid[x_pos][y_pos])==0:
              list_delta_grid[x_pos][y_pos] = [np.array([0., 0.])]
  list_mean_delta_grid = [[np.mean(np.array(list_delta_grid[x_pos][y_pos]), axis = 0) for y_pos in range(len(count_grid[0]))] for x_pos in range(len(count_grid))]
  true_grid = [[np.array([x, y]) for y in Y_grid] for x in X_grid]
    
    
  colors = np.array([[np.arctan2(list_mean_delta_grid[x_pos][y_pos][0], list_mean_delta_grid[x_pos][y_pos][1]) for y_pos in range(len(count_grid[0]))] for x_pos in range(len(count_grid))]).reshape(-1)
  norm = Normalize()
  norm.autoscale(colors)
  colormap = cm.inferno
  plt.figure(figsize=(15,10))
  plt.quiver(np.array([[true_grid[x_pos][y_pos][0] for y_pos in range(len(count_grid[0]))] for x_pos in range(len(count_grid))]).reshape(-1),
           np.array([[true_grid[x_pos][y_pos][1] for y_pos in range(len(count_grid[0]))] for x_pos in range(len(count_grid))]).reshape(-1),
            np.array([[max(min(list_mean_delta_grid[x_pos][y_pos][0]/10, clip_value),-clip_value) for y_pos in range(len(count_grid[0]))] for x_pos in range(len(count_grid))]).reshape(-1),
          np.array([[max(min(list_mean_delta_grid[x_pos][y_pos][1]/10, clip_value),-clip_value) for y_pos in range(len(count_grid[0]))] for x_pos in range(len(count_grid))]).reshape(-1),
            color=colormap(norm(colors)))
  plt.xlim([-1.05, 1.05])
  plt.ylim([-1.05, 1.05])
  plt.show()
  #print(np.array([[true_grid[x_pos][y_pos][0] for y_pos in range(len(count_grid[0]))] for x_pos in range(len(count_grid))]).reshape(-1))
  #print(np.array([[true_grid[x_pos][y_pos][0] for y_pos in range(len(count_grid[0]))] for x_pos in range(len(count_grid))]).reshape(-1).shape)
  coord_b = np.hstack((np.array([[true_grid[x_pos][y_pos][0] for y_pos in range(len(count_grid[0]))] for x_pos in range(len(count_grid))]).reshape(len(count_grid)*len(count_grid[0]),1),
           np.array([[true_grid[x_pos][y_pos][1] for y_pos in range(len(count_grid[0]))] for x_pos in range(len(count_grid))]).reshape(len(count_grid)*len(count_grid[0]),1)))
  coord_e = np.hstack((np.array([[list_mean_delta_grid[x_pos][y_pos][0] for y_pos in range(len(count_grid[0]))] for x_pos in range(len(count_grid))]).reshape(len(count_grid)*len(count_grid[0]),1),
          np.array([[list_mean_delta_grid[x_pos][y_pos][1] for y_pos in range(len(count_grid[0]))] for x_pos in range(len(count_grid))]).reshape(len(count_grid)*len(count_grid[0]),1)))
  return coord_b, coord_e
  

def eval_prob(model, node, batch_size=200):
  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    TEST_BATCH_SIZE = batch_size
    num_test_instance = model.n_nodes-1
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    part = []
    nodes = np.arange(num_test_instance+1)
    #time = max(model.memory.get_last_update(nodes).detach().numpy())

    for k in range(num_test_batch):
      s_idx = k * TEST_BATCH_SIZE + 1
      e_idx = min(num_test_instance + 1, s_idx + TEST_BATCH_SIZE)
      batch = nodes[s_idx:e_idx]
      part.extend(model.compute_edge_affinity_wum(np.full(len(batch), node), batch).detach().numpy().tolist())

    return np.array(part).reshape(1,num_test_instance)[0]

def partition_dict_to_lists(partition_dict): #partition_dict[noeud_i] = partition
  dict_of_list_partition = {} #dict_of_list_partition[commu_i] = [liste des noeuds dans la commu]
    
  for node, com in tqdm.tqdm(partition_dict.items()):
      if com in dict_of_list_partition:
          dict_of_list_partition[com].append(node)
      else:
          dict_of_list_partition[com] = [node]
  return dict_of_list_partition

def affichage_resultats(embedding, partition, nb_com, n_size = 1, text = False):
  embedding = embedding.copy()
    
  res = partition_dict_to_lists(partition)
  size_com = np.zeros(len(res))
  for i in range(len(res)):
      size_com[i] = len(res[i])
  ind_size = np.argsort(size_com)[::-1]
  select_com = {}
  for i in range(nb_com):
      select_com[ind_size[i]] = i
    
  partition = partition.items()
  partition = list(partition)
  partition = np.array(partition)
  ind_part = np.argsort(partition[:,0])
  partition = partition[ind_part]
    
  noeuds_residuels = 0
  for i in range(partition.shape[0]):
      if partition[i][1] in select_com.keys():
          partition[i][1] = select_com[partition[i][1]]
      else:
          partition[i][1] = -1
          noeuds_residuels +=1

  df_plot = pd.DataFrame(np.hstack((embedding,partition[:,[1]])))
    
  plt.figure(figsize=(15,10))
  scatter = plt.scatter(df_plot[0], df_plot[1], np.full((df_plot.shape[0]), n_size), c=df_plot[2],cmap=plt.cm.Spectral)
    
  plt.legend(*scatter.legend_elements(num=nb_com), loc="lower right", title="Communautés")
  #ax.add_artist(legend1)
  plt.xlim([-1.05, 1.05])
  plt.ylim([-1.05, 1.05])
  #plt.plot()
  plt.show()

  if text:
    for i in range(nb_com):
        print("Community ", i, " consists of ", int(size_com[ind_size[i]]), " users, i.e. ",
                size_com[ind_size[i]]/partition.shape[0], "% of the total number of users.")
    print("Community -1 (other users) consists of ", noeuds_residuels,
          " users, i.e. ", noeuds_residuels/partition.shape[0], "% of the total number of users.")
  
  return