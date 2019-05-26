from sklearn import metrics
import time
import tensorflow as tf
import math
import numpy as np
import os
import sys
"""
sys.path.append('datasets')
import graph_communty_dataset
import graph_surface_dataset
import geometric_shape_dataset
sys.path.append('../')
"""
from utils import *

import eval.stats
import networkx as nx

#######################################################################################

def pick_connected_component_new(G):
      num_nodes = G.number_of_nodes()
      adj_list = [[] for x in range(num_nodes)]
      for edge_ in G.edges():
        adj_list[edge_[0]].append(edge_[1])
        adj_list[edge_[1]].append(edge_[0])
      for id,adj in enumerate(adj_list):
          id_min = min(adj)
          if id<id_min and id>=1:
          # if id<id_min and id>=4:
              break
      node_list = list(range(id)) # only include node prior than node "id"
      G = G.subgraph(node_list)
      G = max(nx.connected_component_subgraphs(G), key=len)
      return G

# check and remove selfloop on the graphs
def check_graph_list(graph_list,is_real=True):
    for i in range(len(graph_list)):
        edges_with_selfloops = list(graph_list[i].selfloop_edges())
        if len(edges_with_selfloops)>0:
            graph_list[i].remove_edges_from(edges_with_selfloops)
        if is_real:
            graph_list[i] = max(nx.connected_component_subgraphs(graph_list[i]), key=len)
            graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])
        else:
            graph_list[i] = pick_connected_component_new(graph_list[i])
    return graph_list

#######################################################################################

#logs_path = '/work/tensorboard_logs'
#logs_path = '/drw/ICLR/Code/log'
#logs_path = '/home/darwin/Dropbox/UNICAMP/Research/Code/ICLR/Code/log'
tf.reset_default_graph()

# Set random seed
#seed = 123
#np.random.seed(seed)
#tf.set_random_seed(seed)
#np.random.seed(seed=int(time.time()))
#tf.set_random_seed(seed=int(time.time()))


class GLN:

  def __init__(self, num_nodes, num_features, hidden_layers, tb_logs, RNN):
    self.N = num_nodes
    self.features = num_features
    self.hidden_layers = hidden_layers
    self.tb_logs = tb_logs
    self.rnn = RNN
    if self.rnn == True:
      h_values = len(np.unique(hidden_layers))
      if h_values > 1:
        print('the hidden values must be the same')
        exit(0)
    clear_and_create_paste(self.tb_logs)    

  def set_session(self, session):
    self.session = session

  def fully_connected(self, scope, input, in_dimension, out_dimension):
    '''Fully connection layer
    https://github.com/nilboy/tensorflow-yolo/blob/python2.7/yolo/net/net.py
    Args:
      scope: variable_scope name
      input: [batch_size, ???]
      out_dimension: int32
    Return:
      output: 2-D tensor [batch_size, out_dimension]
    '''
    with tf.variable_scope(scope) as scope:
      reshape = tf.reshape(input, [tf.shape(input)[0], -1])
      #with tf.device('/cpu:0'):
      weights = tf.get_variable(initializer=tf.truncated_normal([int(in_dimension), int(out_dimension)],stddev=0.1),name='weights')
      biases = tf.get_variable(name='biases', initializer=tf.constant_initializer(0.1), shape=[out_dimension])

      local = tf.add( tf.matmul(reshape, weights), biases, name=scope.name)
    return local

  def batch_matmul(self, batch, w):
    '''batch:(20, 100, 3) * w:(3,64) --> (20, 100, 64)'''
    shape_batch = batch.shape.as_list()
    shape_w = w.shape.as_list()
    batch_reshape = tf.reshape(batch, [-1, shape_w[0]])
    batch_res = tf.matmul(batch_reshape, w)
    result = tf.reshape(batch_res, [-1,shape_batch[1],shape_w[1]])
    return result

  def spectral_norm(self, A):
    '''
    Spectral normalization on adj matrice A, where D is matrix degree, eq: D^(-1/2)*A^*D^(-1/2)
    A = (?,N,N)
    '''
    epsilon = 1e-12
    I = np.eye(A.shape[1], A.shape[2], dtype=np.float32)
    A_ = tf.add(tf.cast(A,tf.float32),I) #A^ = A+I
    # degree matrix
    D_vec = tf.reduce_sum(input_tensor=A_, axis=2, keepdims=True) #vector degree of A^
    D_vec = tf.maximum( D_vec, epsilon )

    '''A_normalize = D^(-1/2)*A^*D^(-1/2)'''
    D_pow = tf.pow( D_vec, -0.5 )
    D_ = tf.multiply( I, D_pow ) #D^(-1/2)
    A_normalize = tf.matmul( D_, tf.matmul( A_, D_ ) ) #D^(-1/2)*A^*D^(-1/2)
    return  A_normalize

  def node_embedding(self, H_in, A, num_kernel, shape, name, act_func='relu', type_conv='local', reuse_vars=False):
    '''
    Create the next node embedding. Graph convolutional operation
    H_in: tensor of node embedding, (?,N,D)
    A: tensor of adjacency matrices, (?,N,N)
    shape: shape of the weights, (D,D+1)
    num_kernel: number of kernels,
    '''
    epsilon = 1e-12
    list_h_embedding_kernel = list()
    with tf.variable_scope(name, reuse=reuse_vars) as scope:
      for n_k in range(num_kernel):
        #with tf.device('/cpu:0'):
        W_nk = tf.get_variable(initializer=tf.truncated_normal(shape,stddev=0.1),name='W_'+str(n_k))
        W_nk = tf.nn.l2_normalize(W_nk, [0,1])
        H_t_nk = self.batch_matmul(H_in,W_nk)
        if type_conv == 'local':
          H_t_nk = tf.matmul( A, H_t_nk )
        else: #type_conv == 'global':
          pass
        batch_mean_nk, batch_var_nk = tf.nn.moments( H_t_nk, [0] )
        #with tf.device('/cpu:0'):
        scale_nk = tf.Variable(tf.ones([shape[1]]),name='scale_'+str(n_k))
        beta_nk = tf.Variable(tf.zeros([shape[1]]),name='beta_'+str(n_k))
        BN_nk = tf.nn.batch_normalization(H_t_nk,batch_mean_nk,batch_var_nk,beta_nk,scale_nk,epsilon)

        if act_func == 'relu':
          H_out_nk = tf.nn.relu(BN_nk, name=name+str(n_k))
        elif act_func == 'sigmoid':
          H_out_nk = tf.nn.sigmoid(BN_nk, name=name+str(n_k))
        elif act_func == 'leaky':
          H_out_nk = c(BN_nk, name=scope+str(n_k))
        elif act_func == 'tanh':
          H_out_nk = tf.nn.tanh(BN_nk, name=name+str(n_k))
        else: #None activation function
          H_out_nk = tf.identity(BN_nk, name=name+str(n_k))

        #H_out = tf.nn.dropout( H_out, self.dropout_pl )
        list_h_embedding_kernel.append(H_out_nk)
      node_embedding = tf.add_n(list_h_embedding_kernel)
      return node_embedding

  def node_embedding_to_graph(self, A_in, H, name, act_func='relu', reuse_vars=False):
    '''
    Create the next graph from sorted node embedding.
    A_in: tensor of adjacency matrices, (?,N,N)
    H: tensor of node embedding, (?,N,D+1)
    '''
    H_local = self.node_embedding(H, A_in, 1, [int(H.shape[2]),int(H.shape[2])], 'H_local_'+name, act_func, type_conv='local', reuse_vars=reuse_vars)
    H_global = self.node_embedding(H, A_in, 1, [int(H.shape[2]),int(H.shape[2])], 'H_global_'+name, act_func, type_conv='global', reuse_vars=reuse_vars)
    epsilon = 1e-12
    with tf.variable_scope(name, reuse=reuse_vars) as scope:
      #with tf.device('/cpu:0'):
      Q = tf.get_variable(initializer=tf.truncated_normal([int(H.shape[2]),int(H.shape[2])],stddev=0.1),name='Q_')
      V = tf.get_variable(initializer=tf.truncated_normal([self.N, self.N],stddev=0.1),name='V_')
      tf.add_to_collection('collection_weights', Q)
      tf.add_to_collection('collection_weights', V)

      g_next = tf.matmul(self.batch_matmul(H_local, Q), tf.transpose(H_global, perm=[0,2,1]))
      A_out = tf.map_fn(lambda g_n: tf.matmul(tf.matmul(V, g_n), tf.transpose(V, perm=[1,0])), g_next)
      batch_mean, batch_var = tf.nn.moments( A_out, [0] )
      #with tf.device('/cpu:0'):
      scale = tf.Variable(tf.ones([int(H.shape[1])]),name='scale_')
      beta = tf.Variable(tf.zeros([int(H.shape[1])]),name='beta_')
      BN = tf.nn.batch_normalization(A_out,batch_mean,batch_var,beta,scale,epsilon)

      if act_func == 'relu':
        A_out = tf.nn.relu(BN, name=name)
      elif act_func == 'sigmoid':
        A_out = tf.nn.sigmoid(BN, name=name)
      elif act_func == 'leaky':
        A_out = tf.nn.leaky_relu(BN, name=name)
      else: #None activation function
        A_out = tf.identity(BN, name=name)
      return A_out

  def forward(self, H, A, hidden_layers):
    '''
    Building the architecture for GLN model
    '''
    self.model = dict()
    self.model['H0'] = tf.reshape( H, shape=[ -1, self.N, self.features ] )
    self.model['A0'] = A
    self.model['A0_norm'] = self.spectral_norm( self.model['A0'] )

    l = 0
    H_shape = [self.features if l == 0 else self.hidden_layers[l], self.hidden_layers[l+1]]
    self.model['H'+str(l+1)] = self.node_embedding(self.model['H'+str(l)], self.model['A'+str(l)+'_norm'], 3, H_shape, 'H'+str(l+1))
    self.model['A'+str(l+1)] = self.node_embedding_to_graph(self.model['A'+str(l)+'_norm'], self.model['H'+str(l+1)], 'A'+str(l+1))
    self.model['A'+str(l+1)+'_norm'] = self.spectral_norm( self.model['A'+str(l+1)] )
    for l in range(1,len(self.hidden_layers)-1):
      if self.rnn:
        name_H = 'H_cell_gln'
        name_A = 'A_cell_gln'
        reuse_v  = False if l==1 else True
      else:
        name_H = 'H'+str(l+1)
        name_A = 'A'+str(l+1)
        reuse_v  = False
      H_shape = [self.features if l == 0 else self.hidden_layers[l], self.hidden_layers[l+1]]
      self.model['H'+str(l+1)] = self.node_embedding(self.model['H'+str(l)], self.model['A'+str(l)+'_norm'], 3, H_shape, name_H, reuse_vars=reuse_v)
      self.model['A'+str(l+1)] = self.node_embedding_to_graph(self.model['A'+str(l)+'_norm'], self.model['H'+str(l+1)], name_A, reuse_vars=reuse_v)
      self.model['A'+str(l+1)+'_norm'] = self.spectral_norm( self.model['A'+str(l+1)] )

    in_dim_adjmtrx = self.model['A'+str(len(self.hidden_layers)-1)].shape[1]*self.model['A'+str(len(self.hidden_layers)-1)].shape[2]
    self.FC = tf.nn.sigmoid(self.fully_connected('FC_A', self.model['A'+str(len(self.hidden_layers)-1)], in_dim_adjmtrx, self.N*self.N))
    self.model['A_f'] = tf.reshape(self.FC, [-1, self.N, self.N], name='A_f')

    return self.model['H'+str(len(self.hidden_layers)-1)], self.model['A_f']

  def build_model(self):
    with tf.device('/cpu:0'):
      self.A_pl =  tf.placeholder( tf.float32, shape=( None, self.N, self.N ), name="A" )
      self.X_pl = tf.placeholder( tf.float32, shape=( None, self.N, self.features ), name="X" )
      self.A_groundthuth_pl = tf.placeholder( tf.float32, shape=(None, self.N, self.N), name="A_groundthuth" )
      self.dropout_pl = tf.placeholder( tf.float32 )
      self.learning_rate_pl = tf.placeholder( tf.float32 )
      self.batch_size_pl = tf.placeholder(tf.int32, shape=[], name="batch_size")

    with tf.name_scope('Model'):
      self.logits_H, self.logits_A = self.forward(self.X_pl, self.A_pl, self.hidden_layers)

    with tf.name_scope('Loss_Func'):
      self.loss_adjacency = self.func_loss(type_loss='hed', logits_=self.logits_A, labels_=self.A_groundthuth_pl, batch_size=self.batch_size_pl)
      self.loss_IoU = self.func_loss(type_loss='iou', logits_=self.logits_A, labels_=self.A_groundthuth_pl, batch_size=self.batch_size_pl)
      #L2_reg = tf.reduce_sum(tf.reduce_sum(tf.get_collection('collection_weights')))
      self.loss = 1.0*self.loss_adjacency + 1.0*self.loss_IoU # + 0.05*L2_reg

    with tf.name_scope('Optimizer'):
      self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_pl).minimize(self.loss)

    with tf.name_scope('Prediction'):
      self.prediction_adj = tf.cast(tf.greater_equal(self.logits_A, 0.5), tf.int32)

    with tf.name_scope('Metrics_Losses'):
      self.accuracy_ph = tf.placeholder(tf.float32,shape=None, name='acc')
      self.iou_ph = tf.placeholder(tf.float32,shape=None, name='iou')
      self.recall_ph = tf.placeholder(tf.float32,shape=None, name='recall')
      self.precision_ph = tf.placeholder(tf.float32,shape=None, name='precision')

      self.loss_ph = tf.placeholder(tf.float32,shape=None, name='loss_')
      self.loss_adj_ph = tf.placeholder(tf.float32,shape=None, name='loss_adj_')
      self.loss_IoU_ph = tf.placeholder(tf.float32,shape=None, name='loss_iou_')

    # create a summary for our cost and accuracy
    with tf.name_scope('Loss'):
      self.loss_iou_summary = tf.summary.scalar("Loss IoU", self.loss_IoU_ph)
      self.loss_adj_summary = tf.summary.scalar("Loss Adjacency", self.loss_adj_ph)
      self.loss_summary = tf.summary.scalar("Loss General", self.loss_ph)

    with tf.name_scope('Performance'):
      self.acc_summary = tf.summary.scalar("Accuracy", self.accuracy_ph)
      self.iou_summary = tf.summary.scalar("IoU", self.iou_ph)
      self.recall_summary = tf.summary.scalar("Recall", self.recall_ph)
      self.precision_summary = tf.summary.scalar("Precision", self.precision_ph)

  def func_loss(self, type_loss, logits_, labels_, batch_size):
    '''
    Define different type of function losses
    '''
    def HED_loss(logits, labels, batch_size):
      '''
      Compute average cross entropy and add to loss collection.
      https://arxiv.org/pdf/1504.06375.pdf
      '''
      labels_ = labels
      labels = tf.to_int64(labels)
      count_neg = tf.reduce_sum(tf.abs(1.0 - labels_)) # the number of 0 in y
      count_pos = tf.reduce_sum(labels_) # the number of 1 in y (less than count_neg)

      beta = count_neg / (count_neg + count_pos)
      weight_per_label = tf.scalar_mul(1.0 - beta, tf.cast(tf.equal(labels, 0),tf.float32)) + \
                         tf.scalar_mul(beta, tf.cast(tf.equal(labels, 1),tf.float32))
      logits = tf.clip_by_value(logits,1e-15,1.0-(1e-7))
      binary_cross_entropy = -tf.reduce_sum (labels_*tf.log(logits) + (1.0-labels_)*tf.log(1.0-logits))
      cost = tf.multiply(weight_per_label, binary_cross_entropy)
      cost = tf.reduce_mean(cost, name='HED_loss')
      return tf.divide( cost, tf.cast(batch_size, tf.float32)*self.N)

    def Dice_square_loss(logits, labels):
      '''
      Approximate Dice coef. loss from
      https://arxiv.org/abs/1606.04797 
      (V-net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation )
      '''
      labels = tf.reshape(labels, [-1])
      logits = tf.reshape(logits, [-1])
      labels_f32 = tf.cast(labels, tf.float32)
      inter = tf.reduce_sum(tf.multiply(logits, labels_f32, name='mul_b_sum'))
      denominator = tf.add(tf.reduce_sum(tf.square(logits)), tf.reduce_sum(tf.square(labels_f32)))
      total_dice_loss = tf.subtract(tf.constant(1.0, dtype=tf.float32), tf.div(2.0*inter, denominator+1e-10))
      return tf.abs(total_dice_loss)

    def IoU_loss(logits, labels):
      '''
      Approximate IoU loss from
      http://www.cs.umanitoba.ca/~ywang/papers/isvc16.pdf
      '''
      labels = tf.reshape(labels, [-1])
      logits = tf.reshape(logits, [-1])

      labels_f32 = tf.cast(labels, tf.float32)
      inter = tf.reduce_sum(tf.multiply(logits, labels_f32, name='mul_b_sum'))
      union = tf.reduce_sum(tf.subtract(tf.add(logits, labels_f32), tf.multiply(logits, labels_f32)))

      total_loss = tf.subtract(tf.constant(1.0, dtype=tf.float32), tf.div(inter, union))
      return tf.abs(total_loss)

    if type_loss == 'iou':
      return IoU_loss(logits_, labels_)
    elif type_loss == 'dice':
      return Dice_square_loss(logits_, labels_)
    else: #'hed'
      return HED_loss(logits_, labels_, batch_size)

  def calc_metrics(self, y_true, y_pred):
    '''y_true is the groundthuth and
      y_pred is our prediction. For binary class'''
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
    iou = np.count_nonzero(np.logical_and(y_true,y_pred))/( np.count_nonzero(np.logical_or(y_true,y_pred)) + 1e-12 )
    recall = metrics.recall_score(y_true=y_true, y_pred=y_pred)
    precision = metrics.precision_score(y_true=y_true, y_pred=y_pred)
    #f1 = metrics.f1_score(y_true=y_true, y_pred=y_pred)
    return acc, iou, recall, precision


def training_and_test( dataset, type_data, model, epochs, learning_rate, batch_size):

  _dropout = 0.5
  is_summary = False
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.gpu_options.per_process_gpu_memory_fraction = 0.9
  with tf.Session(config=config) as sess:
    #with tf.device('/cpu'):
    model.set_session(sess)
    model.build_model()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    with tf.device('/cpu'):
      if is_summary:
        #with tf.device('/cpu'):
        writer = tf.summary.FileWriter(model.tb_logs, graph=tf.get_default_graph())

      iteration = 0
      for epoch in range(epochs):
        print("\n########## epoch " + str(epoch+1) + " ##########")
        gen_trainig = dataset.train_generator(batch_size=batch_size)

        train_loss = []
        train_loss_adj = []
        train_loss_iou = []
        train_loss_symm = []
        train_acc = []
        train_iou = []
        train_recall = []
        train_precision = []
        for gt_graph, set_feature, in_graph in gen_trainig:
          feed_dict_train={model.X_pl:set_feature, model.A_pl:in_graph, model.A_groundthuth_pl:gt_graph,\
            model.dropout_pl:_dropout, model.batch_size_pl:gt_graph.shape[0], model.learning_rate_pl:learning_rate}

          # Training step
          request_train = [model.optimizer, model.loss, model.loss_adjacency,\
                      model.loss_IoU, model.prediction_adj]#model.summary_op
          _, loss_, loss_adj_, loss_iou_, pred_  = sess.run(request_train, feed_dict=feed_dict_train)
          acc_adj, iou_adj, recall_adj, precision_adj = model.calc_metrics(y_true=gt_graph, y_pred=pred_)

          train_loss.append(loss_)
          train_loss_adj.append(loss_adj_)
          train_loss_iou.append(loss_iou_)
          train_acc.append(acc_adj)
          train_iou.append(iou_adj)
          train_recall.append(recall_adj)
          train_precision.append(precision_adj)

          #summary = outs[4]
          #writer.add_summary(summary, iteration)
        acc_adj = np.mean(train_acc);  iou_adj = np.mean(train_iou); recall_adj = np.mean(train_recall);
        precision_adj = np.mean(train_precision); loss_ = np.mean(train_loss); loss_adj_ = np.mean(train_loss_adj);
        loss_iou_ = np.mean(train_loss_iou);

        if is_summary:
          #with tf.device('/cpu'):
          req = [ model.loss_summary, model.loss_adj_summary, model.loss_iou_summary, \
                  model.acc_summary, model.iou_summary, model.recall_summary, model.precision_summary]
          feed_summary = {model.loss_ph:loss_, model.loss_adj_ph:loss_adj_, model.loss_IoU_ph:loss_iou_, \
                  model.accuracy_ph:acc_adj, model.iou_ph:iou_adj, \
                  model.recall_ph:recall_adj, model.precision_ph:precision_adj}
          loss_summary, loss_adj_summary, loss_iou_summary, acc_summary, iou_summary, recall_summary, \
                  precision_summary = sess.run(req, feed_dict=feed_summary)
          writer.add_summary(loss_summary, epoch)
          writer.add_summary(loss_adj_summary, epoch)
          writer.add_summary(loss_iou_summary, epoch)
          writer.add_summary(acc_summary, epoch)
          writer.add_summary(iou_summary, epoch)
          writer.add_summary(recall_summary, epoch)
          writer.add_summary(precision_summary, epoch)
          writer.flush()
          #iteration+=1


        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss_),\
          "loss_adjacency=", "{:.5f}".format(loss_adj_),"train_acc_adj=", "{:.5f}".format(acc_adj),\
          "train_iou_adj=", "{:.5f}".format(iou_adj),"recall_adj=", "{:.5f}".format(recall_adj),\
          "precision_adj=", "{:.5f}".format(precision_adj))

        print("start validating.......")
        gen_validation = dataset.valid_generator( batch_size = batch_size )

        val_acc = []
        val_iou = []
        val_recall = []
        val_precision = []
        for val_gt_graph, val_set_feature, val_in_graph in gen_validation:
          feed_dict_val={model.X_pl:val_set_feature, model.A_pl:val_in_graph, model.A_groundthuth_pl:val_gt_graph,\
            model.dropout_pl:1.0, model.batch_size_pl: gt_graph.shape[0]}

          outs = sess.run([model.loss, model.prediction_adj], feed_dict=feed_dict_val)
          val_acc_adj, val_iou_adj, val_recall_adj, val_precision_adj = model.calc_metrics(y_true=val_gt_graph, y_pred=outs[1])

          val_acc.append(val_acc_adj)
          val_iou.append(val_iou_adj)
          val_recall.append(val_recall_adj)
          val_precision.append(val_precision_adj)

        print("val_adj_acc=", "{:.5f}".format(np.mean(val_acc)),\
          "val_adj_IoU=", "{:.5f}".format(np.mean(val_iou)),"val_adj_recall=", "{:.5f}".format(np.mean(val_recall)),\
          "val_adj_precision=", "{:.5f}".format(np.mean(val_precision)))

      print("Optimization Finished!")

      gen_testing = dataset.test_generator( batch_size = batch_size )
      # Testing
      print("Starting Testing")
      save_dir = 'results'
      num_sample_test = dataset.num_test
      print("number of test samples: ", num_sample_test)
      test_acc = []
      test_iou = []
      test_recall = []
      test_precision = []
      test_confusion_matrix = np.zeros((2,2), dtype=np.float32)
      graph_list_pred = []
      graph_list_gt = []

      counter = 0
      for test_gt_graph, test_set_feature, test_in_graph in gen_testing:
        feed_dict_test={model.X_pl:test_set_feature, model.A_pl:test_in_graph, model.A_groundthuth_pl:test_gt_graph,\
          model.dropout_pl:1.0, model.batch_size_pl: gt_graph.shape[0]}

        request = [model.loss, model.loss_adjacency, model.prediction_adj, model.logits_H]
        test_cost, test_cost_adj, pred_adj, hidden_H = sess.run(request, feed_dict=feed_dict_test)

        test_acc_adj, test_iou_adj, test_recall_adj, test_precision_adj = model.calc_metrics(y_true=test_gt_graph, y_pred=pred_adj)
        test_confusion_matrix += metrics.confusion_matrix(y_true=test_gt_graph.flatten(), y_pred=pred_adj.flatten())

        if type_data == 'caveman_2' or type_data == 'caveman_4':
          show_save_community(save_dir, counter, test_gt_graph, test_set_feature, pred_adj, hidden_H, int(type_data.split('_')[-1]))
        if type_data == 'surf':
          show_save_surface(save_dir, counter, test_gt_graph, test_set_feature, pred_adj, hidden_H)
        if type_data == 'img':
          show_save_geo_fig(save_dir, counter, test_gt_graph, test_set_feature, pred_adj, hidden_H)

        test_acc.append(test_acc_adj)
        test_iou.append(test_iou_adj)
        test_recall.append(test_recall_adj)
        test_precision.append(test_precision_adj)

        for k in range(test_gt_graph.shape[0]):
          G_pred = nx.to_networkx_graph(data=pred_adj[k])
          graph_list_pred.append(G_pred)
          G_gt = nx.to_networkx_graph(data=test_gt_graph[k])
          graph_list_gt.append(G_gt)

      print("Total Accuracy for test graphs: ", np.mean(test_acc))
      print("UoI for test graphs: ", np.mean(test_iou))
      print("Recall for test graphs: ", np.mean(test_recall) )
      print("Precision for test graphs: ", np.mean(test_precision) )

      #graph_list_gt = dataset.graphs_test
      graph_list_gt = check_graph_list(graph_list_gt)
      graph_list_pred = check_graph_list(graph_list_pred)

      #utils.save_graph_list(graph_list_pred, 'pred.dat')
      mmd_degree = eval.stats.degree_stats(graph_list_gt, graph_list_pred)
      mmd_clustering = eval.stats.clustering_stats(graph_list_gt, graph_list_pred)
      try:
        mmd_4orbits = eval.stats.orbit_stats_all(graph_list_gt, graph_list_pred)
      except:
        mmd_4orbits = -1
      print("mmd_degree: ", mmd_degree)
      print("mmd_clustering: ", mmd_clustering)
      print("orbits: ", mmd_4orbits)

      save_conf_matrix(save_dir, test_confusion_matrix, test_acc, test_iou, test_recall, test_precision, \
                      mmd_degree, mmd_clustering, mmd_4orbits)

  sess.close()


def main():
  #'''
  #------------------ Community synthetic data ------------------
  type_data = 'caveman_4' #['caveman_2', 'caveman_4']
  comm_dataset = graph_communty_dataset.GenerateDataGraph(type_data, num_perm=0)
  num_nodes = comm_dataset.num_nodes
  dim_features = comm_dataset.num_features
  ep = 150
  lr = 1e-5
  type_dataset = type_data
  dataset = comm_dataset
  #'''
  '''
  #------------------ 3D surface synthetic data ------------------
  type_data = 'elliptic_paraboloid' #['elliptic_paraboloid', 'saddle', 'torus', 'ellipsoid', 'elliptic_hyperboloid', 'another']
  num_samples = 2000
  num_nodes = 10*10 #100 #square
  dim_features = 3
  surf_dataset = graph_surface_dataset.GenerateDataGraphSurface(type_dataset=type_data, num_surfaces=num_samples, \
      num_points=num_nodes)
  ep = 100
  lr = 1e-6
  type_dataset = 'surf'
  dataset = surf_dataset
  '''
  '''
  #------------------ Geometric shape synthetic data ------------------
  type_dataset = 'img'
  num_samples = 3000 #3000
  num_nodes = 10*10 #square
  dim_features = 5
  ep = 150
  lr = 1e-6

  dim_h = int(np.sqrt(num_nodes))
  dim_w = int(np.sqrt(num_nodes))
  type_dist = 'D4' #['D4', 'D8']
  img_dataset = geometric_shape_dataset.GenerateImg(dim_x=dim_h, dim_y=dim_w, proportion=(0.05, 0.2, num_samples))
  img_dataset.load_data()
  dataset = img_dataset
  '''

  gln = GLN(num_nodes, dim_features, [64,32,16,8], '/work/tensorboard_logs/GLN', RNN=False)
  #gln = GLN(num_nodes, dim_features, [32,32,32,32], '/work/tensorboard_logs/GLN', RNN=True)
  training_and_test(dataset=dataset, type_data=type_dataset, model=gln, epochs=ep, learning_rate=lr, batch_size=20) #40


if __name__ == "__main__":
    main()
