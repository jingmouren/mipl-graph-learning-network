import sys
import os
import numpy as np

sys.path.append('datasets')
import graph_communty_dataset
import graph_surface_dataset
import geometric_shape_dataset
sys.path.append('../')

def clear_and_create_paste(tb_logs):
  os.system('mkdir results >&-')
  list_paste = ['comp', 'graphs_gt', 'graphs_pred', 'gt', 'H_in', 'H_out', 'input', 'pred', 'show_tsne', 'tsne']
  os.system('mkdir results/ >&-')
  for paste in list_paste:
    os.system('mkdir results/'+paste+' >&-')
  os.system('rm -r ' + tb_logs + ' >&-')
  os.system('find results -name "*.png" -type f -delete -print >&-')
  os.system('find results -name "*.csv" -type f -delete -print >&-')
  os.system('find results -name "*.txt" -type f -delete -print >&-')

def show_save_community(save_dir, counter, test_gt_graph, test_set_feature, pred_adj, hidden_H, num_clus):
  '''
  Show and save the results of Community dataset
  '''
  for k in range(test_gt_graph.shape[0]):
    graph_surface_dataset.draw_surface(name=save_dir+'/pred/pred_'+str(counter), points_coord=test_set_feature[k], adj=pred_adj[k])
    graph_surface_dataset.draw_surface(name=save_dir+'/gt/gt_'+str(counter), points_coord=test_set_feature[k], adj=test_gt_graph[k])

    graph_communty_dataset.save_graph(name=save_dir+'/graphs_pred/pred_'+str(counter), points_coord=test_set_feature[k], adj=pred_adj[k], dim=2)
    graph_communty_dataset.save_graph(name=save_dir+'/graphs_gt/gt_'+str(counter), points_coord=test_set_feature[k], adj=test_gt_graph[k], dim=2)
    graph_communty_dataset.save_H(name=save_dir+'/H_out/H_'+str(counter), features=hidden_H[k])
    graph_communty_dataset.save_H(name=save_dir+'/H_in/H_'+str(counter), features=test_set_feature[k])
    graph_communty_dataset.save_hyperspace(save_dir=save_dir, number=counter, features=hidden_H[k], reduce_dim_to=3, n_clusters=num_clus)
    counter += 1

def show_save_surface(save_dir, counter, test_gt_graph, test_set_feature, pred_adj, hidden_H):
  '''
  Show and save the results of Surface dataset
  '''
  for k in range(test_gt_graph.shape[0]):
    graph_surface_dataset.draw_surface(name=save_dir+'/gt/gt_'+str(counter), points_coord=test_set_feature[k], adj=test_gt_graph[k])
    graph_surface_dataset.draw_surface(name=save_dir+'/input/in_'+str(counter), points_coord=test_set_feature[k], adj=test_in_graph[k])
    graph_surface_dataset.draw_surface(name=save_dir+'/pred/pred_'+str(counter), points_coord=test_set_feature[k], adj=pred_adj[k])

    graph_surface_dataset.save_graph(name=save_dir+'/graphs_pred/pred_'+str(counter), points_coord=test_set_feature[k], adj=pred_adj[k], dim=3)
    graph_surface_dataset.save_graph(name=save_dir+'/graphs_gt/gt_'+str(counter), points_coord=test_set_feature[k], adj=test_gt_graph[k], dim=3)
    graph_surface_dataset.save_H(name=save_dir+'/H_out/H_'+str(counter), features=hidden_H[k])
    graph_surface_dataset.save_H(name=save_dir+'/H_in/H_'+str(counter), features=test_set_feature[k])
    counter += 1

def show_save_geo_fig(save_dir, counter, test_gt_graph, test_set_feature, pred_adj, hidden_H):
  '''
  Show and save the results of Geometric images dataset
  '''
  display = geometric_shape_dataset.CDisplay()
  shape_img = test_set_feature.shape
  dim_h, dim_w = int(np.sqrt(shape_img[1])), int(np.sqrt(shape_img[1]))
  img_set_feature_t = test_set_feature.reshape(shape_img[0], dim_h, dim_w, shape_img[2])
  img_set_feature_t = img_set_feature_t[:,:,:,0:3]
  for k in range(test_gt_graph.shape[0]):
    display.display_neighborhood(img_set_feature_t[k], img_set_feature_t[k],\
                test_gt_graph[k], pred_adj[k], dim_h, dim_w, save_dir+'/comp/comp_'+str(counter)+'.png')
    geometric_shape_dataset.save_img(save_dir+'/input/img_in_' + str(counter),img_set_feature_t[k])
    #geometric_shape_dataset.save_img(save_dir+'gt/img_gt_' + str(counter),set_class_batch[k])
    geometric_shape_dataset.save_graph_img(save_dir+'/graphs_gt/graph_gt_' + str(counter), test_gt_graph[k])
    geometric_shape_dataset.save_graph_img(save_dir+'/graphs_pred/graph_out_' + str(counter), pred_adj[k])
    geometric_shape_dataset.save_H(name=save_dir+'/H_out/H_'+str(counter), features=hidden_H[k])
    geometric_shape_dataset.save_H(name=save_dir+'/H_in/H_'+str(counter), features=test_set_feature[k])
    counter += 1

def save_conf_matrix(saved_dir, test_confusion_matrix, test_acc, test_iou, test_recall, test_precision, mmd_degree, mmd_clustering, mmd_4orbits):
  num_classes_bin = 2 # edge(1) or unedge(0)
  f = open(saved_dir+'/confusion_matrix.txt', 'w')
  f.write("Confusion Matrix of the Graphs\n")
  f.write("[")
  for c1 in range(num_classes_bin):
      f.write("[")
      for c2 in range(num_classes_bin):
          f.write(str(test_confusion_matrix[c1,c2])),
          if( c2 + 1 < num_classes_bin ):  f.write(", "),
          else: f.write(']')
      if( c1 + 1 < num_classes_bin): f.write(",\n")
      else: f.write("]\n")

  f.write("Total Accuracy for test graphs: " + str(np.mean(test_acc)) + "\n")
  f.write("UoI for test graphs: " + str(np.mean(test_iou)) + "\n")
  f.write("Recall for test graphs: " + str(np.mean(test_recall)) + "\n")
  f.write("Precision for test graphs: " + str(np.mean(test_precision)) + "\n")
  f.write("\n")
  f.write("mmd_degree: " + str(mmd_degree) + "\n")
  f.write("mmd_clustering: " + str(mmd_clustering) + "\n")
  f.write("orbits: " + str(mmd_4orbits) + "\n")

  f.close()

