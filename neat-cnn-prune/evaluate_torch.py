
import torch
import torch.nn as nn
import torch.nn.functional as F
import neat1.genome

import numpy as np


class Net(nn.Module):

    def __init__(self, config, genome: neat1.genome.DefaultGenome):
        nn.Module.__init__(self)
        # 根据genome的连接、节点数，设置边的权重和节点的权重
        self.old_connections = genome.connections    
        self.old_layer = genome.layer
        self.old_nodes = genome.nodes
        self.num_cnn_layer = config.genome_config.num_cnn_layer
        self.num_layer = config.genome_config.num_layer
        self.num_inputs = config.genome_config.num_inputs
        self.num_outputs = config.genome_config.num_outputs
        self.num_first_fc_layer_node = config.genome_config.num_first_fc_layer_node
        
        self.nodes = {}
                                    
        self.set_layers(genome)
        self.set_parameters(genome)
        
    def forward(self, x):
        l = list(self.children())
        
        for i in range(self.num_cnn_layer):
            # max_index = 2*(num_cnn_layer-1)+1=2*num_cnn_layer-1
            x = l[1 * i](x)
            #x = nn.BatchNorm2d(num_features=x.shape[1])(x)
            
            x = F.relu(x)
            x = F.max_pool2d(x, 3)
        # 输出x的长宽 (4 x  4) , 第一层fc层需要flatten成 
        # num_final_cnn_channels * (4 x 4) = num_final_cnn_channels * 16          
        
        #x = nn.AvgPool2d(2)(x)            
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        #x = x.view(-1, self.num_flat_features)
        
        # final fc-layer  is not activated            
        for i in range(self.num_cnn_layer, self.num_layer - 1):
            # start index =  max_index +1 = 2*num_cnn_layer
            x = F.relu(l[i + 0](x))
        #bn = nn.BatchNorm1d(num_features=10)
        x = l[-1](x)                 
        #x = bn(x)        
        #print(F.softmax(x, dim=1))
        #return F.softmax(x, dim=1)
        return (x)
    
    def forward_with_dropout(self, x):
        l = list(self.children())  
        dropout = nn.Dropout(p=0.25)

        for i in range(self.num_cnn_layer):            
            x = l[1 * i](x)
            #x = nn.BatchNorm2d(num_features=x.shape[1])(x)            
            x = F.relu(x)
            x = F.max_pool2d(x, 2)                
            
        #x = nn.AvgPool2d(2)(x)            
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
            
        # final fc-layer  is not activated            
        for i in range(self.num_cnn_layer, self.num_layer - 1):
            x = dropout(x)        
            x = F.relu(l[i + 0](x))
        
        x = dropout(x)            
        #bn = nn.BatchNorm1d(num_features=10)
        x = l[-1](x) 
        
        return (x)
       
    def compute_in_out_degree(self): # compute in_degree & out_degree
        
        in_dict = {}
        out_dict = {}
        
        for in_, out in self.old_connections:
            
            if in_dict.__contains__(out) != False:
                in_dict[out].add(in_)
            else:
                in_dict[out] = set()
                in_dict[out].add(in_) 
                
            if out_dict.__contains__(in_) != False:
                out_dict[in_].add(out)
            else:
                out_dict[in_] = set()
                out_dict[in_].add(out) 
                
        return in_dict, out_dict                
    
    def min_val_index(self, value_this_layer):
        
        min_value, min_idx = 1.1, None
        #print(len(list(value_this_layer)))
        
        for i, value in enumerate(list(value_this_layer)):                        
            
            # find the index of the smallest-L2-norm filter               
             if value < min_value:
                min_value = value
                min_idx = i

        if min_idx is None:
            raise RuntimeError('min_idx is None')
            
        return min_value, min_idx           
    
    def prune_fc_weight(self):
        values = []
                        
        for p in self.parameters():
            if len(p.data.size()) == 2: # fc层 的 weight
                
                if len(values) == self.num_layer - self.num_cnn_layer -1:
                    break
                p_np = p.data.cpu().numpy()
                #print('size() ', p.data.size())
                
                value_this_layer = np.abs(p_np).sum(axis=1) / \
                    (p_np.shape[1])
                # normalize
                value_this_layer = value_this_layer / \
                    (np.abs(value_this_layer).sum() + 1e-5 )
                
                
                min_value, min_idx = self.min_val_index(value_this_layer)
                                              
                values.append([min_value, min_idx])                       
                
        values = np.array(values)
        #print(values.shape)
        assert len(values) == self.num_layer - self.num_cnn_layer -1
        
        #print(values[0])
        # 最小L1-Norm filter 所在的 层layer index
        to_prune_layer_idx = np.argmin(values[:, 0]) + self.num_cnn_layer
        #print('to_prune_layer ', to_prune_layer_idx)
        
        pruned_filter_idx = int(values[to_prune_layer_idx-self.num_cnn_layer, 1])                
        #print('pruned_filter_idx ', pruned_filter_idx)
        to_prune_layer_idx *= 1
        #l = list(self.old_layer[int(to_prune_layer_idx/2)][1]) # l是 结点 id 的 list
        l = list(self.old_layer[int(to_prune_layer_idx/1)][1]) # l是 结点 id 的 list
        l.sort()# 如果改变了 old_layer列表，就不能用sort
        #print('len(l) ', len(l))
        if len(l) == 1:
            print('cannot prune  layer %d again'%(len(self.old_layer[to_prune_layer_idx][1])))
            return None, None, False
        #print(l)
        
        del_node = l[pruned_filter_idx] # 删除的结点的id
        #print(' del_node  ', del_node )  

        del_node_list = []                                
        del_connects = []
        
        in_dict, out_dict = self.compute_in_out_degree()
                                      
        for in_node,out_node in self.old_connections:
            if del_node == in_node or del_node == out_node:
                # 输出层结点
                if out_node < self.num_outputs and out_node >= 0:
                    #print(out_node,' in_dict --> ', in_dict[out_node])
                    if len(in_dict[out_node]) == 1:
                        print('prune ' , in_node, ' ==> output ', out_node, ' forbid')
                        print('cannot prune  layer %d with %d nodes '%(to_prune_layer_idx ,len(self.old_layer[to_prune_layer_idx][1])))
                        return None, None, False
                        #continue
                del_connects.append((in_node, out_node))
                
                in_dict[out_node].remove(in_node)
                out_dict[in_node].remove(out_node)
                
                if del_node == in_node and len(in_dict[out_node]) == 0:
                                     
                    l1 = [out_node]
                    #fa = {out_node: del_node}
                    while  len(l1) >0:# 入度in为零的结点的list
                        l2 = []
                        
                        for out in l1: # 要删除的结点                                                                                         
                            for out_ in out_dict[out]: # (out --> out_)                            
                                in_dict[out_].remove(out)
#                                fa[ out_ ] = out # record the previous node
                                
                                if len(in_dict[out_]) == 0:
                                     # 输出层结点要小心
                                    if out_ < self.num_outputs and out_ >=0:
                                        in_dict[out_].add(out) # 注意要补回来
                                        print('ouput node ' ,out_ ,' <-- ', out ,' cannot deleted')
                                        # TODO: out_ <-- out <-- fa[out] < --...should be saved
                                        #fa[out]
#                                        x = fa[out]
#                                        while len(in_dict[ x ]) == 0:
#                                            x = fa[x]
                                        #assert x == del_node
                                        print('cannot prune  layer %d '%(len(self.old_layer[to_prune_layer_idx][1])))
                                        return None, None, False
                                        #continue # 跳过下面的删除连接
                                    else:
                                        l2.append(out_)
                                        
                                # 待删除的连接                                        
                                del_connects.append((out, out_))                                               
                            out_dict[out] = set()
                            
                            #self.old_layer[int(to_prune_layer_idx/2)][1].remove(del_node)
                            del_node_list.append(out)
                            #self.old_layer[ self.old_nodes[out].layer ][1].remove(out)
                            #del self.old_nodes[out] # 删除 结点
                            
                        l1 = l2                                                                
                if del_node == out_node and len(out_dict[in_node]) == 0:
                    
                    l1 = [in_node]
                    
                    while  len(l1) >0:# 出度out为零的结点的list
                        l2 = []
                        
                        for in_ in l1:  # 出度out为零的结点                                                                                        
                            for in_1 in in_dict[in_]:  # previous layer node
                                out_dict[in_1].remove( in_ )
                                
                                if len(out_dict[in_1]) == 0:
                                     # 输出层结点
                                    if in_1 < 0: # input pins 
                                        out_dict[in_1].add(in_) # 补回来
                                        print('in pin ' ,in_1 ,' <-- ', in_ ,' cannot delete')
                                        
                                        return None, None, False
                                        
                                    else:
                                        l2.append(in_1)
                                # 待删除的连接                                        
                                del_connects.append((in_1, in_))                                               

                            in_dict[in_] = set()                                                                                    
                            del_node_list.append(in_)                            
                            
                        l1 = l2     
                                    
        for del_con in del_connects:
            del self.old_connections[del_con]
            # cnn layer nodes
            if self.old_nodes[del_con[1]].layer < self.num_cnn_layer:
                # delete in_nodes
                #print( del_con[1],' <- ',del_con[0] )
                #print('in_nodes', self.old_nodes[del_con[1]].in_nodes)
                
                for idx_ in range(len(self.old_nodes[del_con[1]].in_nodes)):
                    
                    if self.old_nodes[del_con[1]].in_nodes[idx_] == del_con[0]:
                                                                        
                        j = idx_+1
                        while j < len(self.old_nodes[del_con[1]].in_nodes):
                            self.old_nodes[del_con[1]].in_nodes[j-1] = \
                                self.old_nodes[del_con[1]].in_nodes[j]
                            
                            self.old_nodes[ del_con[1] ].kernal[ j-1 ] = \
                                self.old_nodes[ del_con[1] ].kernal[j ]
                            j += 1                                
                        
                        if len(self.old_nodes[del_con[1]].kernal) > 1:                            
                            self.old_nodes[ del_con[1] ].kernal = \
                                self.old_nodes[del_con[1]].kernal[:-1]
                        
                        del self.old_nodes[del_con[1]].in_nodes[ -1 ] 
                        
                        assert  len(self.old_nodes[ del_con[1] ].in_nodes)== 0 or \
                                    len(self.old_nodes[del_con[1]].kernal) == \
                                        len(self.old_nodes[ del_con[1] ].in_nodes)
                                                   
                        break
                                    
                            
        
        if self.old_nodes.__contains__(del_node):
            
            del self.old_nodes[del_node] # 删除 结点
            #self.old_layer[int(to_prune_layer_idx/2)][1].remove(del_node)
            self.old_layer[int(to_prune_layer_idx/1)][1].remove(del_node)
        else:
            print(del_node , ' already deleted ')
    
                
        for del_no in del_node_list: # 因为 del_node 的delete 而 delete 的 node
            
            if self.old_nodes.__contains__(del_no):          
                                
                node_type = 2
                prune_layer = self.old_nodes[del_no].layer
                
                if prune_layer < self.num_cnn_layer:
                    node_type = 4
                cur_idx = 0
                print('hhhhhh')
                for p in self.parameters():
                    print(p.data.size())
                    if len(p.data.size()) == 2 or len(p.data.size()) == 4:
                        
                        if cur_idx == prune_layer:
                                                            
                            del_no_pos = self.nodes[ del_no ][1]                                
                            p.data = torch.cat([p.data[0: del_no_pos,:], p.data[del_no_pos+1 :,:] ], dim=0 )
                                                                                                                                                        
                        elif cur_idx == prune_layer + 1:
                            
                            del_no_pos = self.nodes[ del_no ][1]     
                            #print('Before delete ', del_no,' ',p.data.size())                                 
                            p.data = torch.cat( [p.data[:, 0: del_no_pos]  , p.data[:, del_no_pos+1:] ] ,dim=1  )
                            #print('After delete ', p.data.size())
                            for nod in self.old_layer[prune_layer][1]:
                                if self.nodes[nod][1] > del_no_pos:
                                    self.nodes[nod][1] -= 1
                            
                            self.nodes[del_no] = -123                            
                            del self.nodes[del_no]
                            
                            break
                        
                        cur_idx += 1
                        
                    elif len(p.data.size()) == 1: # bias                        
                        
                        if cur_idx == prune_layer + 1:
                            del_no_pos = self.nodes[ del_no ][1]   
                            p.data = torch.cat([p.data[: del_no_pos] , p.data[del_no_pos+1:]], dim=0)
                            # add 
                            if node_type == 4 and prune_layer == self.num_cnn_layer -1:
                                node_type = 2
                                
                self.old_layer[ self.old_nodes[del_no].layer ][1].remove( del_no )
                del self.old_nodes[del_no] # 删除 结点        
            else:
                print(del_no , ' has been deleted ')              
                        
        idx = self.num_cnn_layer-1             
                        
        for p in self.parameters():            

            if len(p.data.size()) == 2 : # fc weight                
                idx += 1
                
                if idx == to_prune_layer_idx: # fc
                        
                    p.data = torch.cat([p.data[0: pruned_filter_idx,:], p.data[pruned_filter_idx+1 :,:] ], dim=0  )                                            
                
                elif idx == to_prune_layer_idx + 1: # next fc
                    p.data = torch.cat( [p.data[:, 0: pruned_filter_idx]  , p.data[:, pruned_filter_idx+1:] ] ,dim=1  )
                    
                    del_no = -444          
                    for k, pos in self.nodes.items():
                        if pos[0] == to_prune_layer_idx and pos[1] == pruned_filter_idx:
                            del_no = k
                            break
                    
                    assert del_no > 0
                    
                    for nod in self.old_layer[to_prune_layer_idx][1]:
                        
                        if self.nodes[nod][1] >  pruned_filter_idx:
                            self.nodes[nod][1] -= 1
                    
                    self.nodes[ del_no ] = -123                            
                    del self.nodes[del_no]
                    
                    break
                
            elif len(p.data.size()) == 1 : # bias                
                if idx  == to_prune_layer_idx:
                    p.data = torch.cat([p.data[: pruned_filter_idx] , p.data[pruned_filter_idx+1:]], dim=0)
                    
                                            
#        print('Prune fc-filter #{} gene_id #{} in layer #{}'.format(
#            pruned_filter_idx,
#            del_node,
#            to_prune_layer_idx))
        
        return del_node, del_connects, True
    
    def prune_one_filter(self):
        values = []            
        for p in self.parameters():
            if len(p.data.size()) == 4: # conv weight (out_channel,in_channel,width,height) 

                p_np = p.data.cpu().numpy()
                #print('size() ', p.data.size())
#                if p.data.size()[1] ==1:
#                    print(p.data)
                
                # L2 norm for each filter in this layer                
                # 此处没有考虑剪掉前面的filter 对后面的 filter 输入通道数的影响
                value_this_layer = np.square(p_np).sum(axis=(1,2,3)) / \
                            (p_np.shape[1]*p_np.shape[2]*p_np.shape[3])
                
                # normalize   如果全为0
                value_this_layer = value_this_layer / \
                (np.sqrt(np.square(value_this_layer).sum()) + 1e-5 )
                
                min_value, min_idx = self.min_val_index(value_this_layer)
                                               
                values.append([min_value, min_idx])   
                
        values = np.array(values)
        #print(values.shape)
        #print(values[0])
        
        # 最小L2-Norm filter 所在的 层layer index
        to_prune_layer_idx = np.argmin(values[:, 0])
        #print('prune_layer ', to_prune_layer_idx)
        pruned_filter_idx = int(values[to_prune_layer_idx, 1])
        #print('filter_id', pruned_filter_idx)
        
        to_prune_layer_idx *= 1
        #l = list(self.old_layer[int(to_prune_layer_idx/2)][1]) # l是 结点 id 的 list
        l = list(self.old_layer[int(to_prune_layer_idx/1)][1]) # l是 结点 id 的 list
        l.sort()# 如果改变了 old_layer列表，就不能用sort
        #print(len(l))
        if len(l) == 1:
            print('cannot prune  layer %d again'%(len(self.old_layer[to_prune_layer_idx][1])))
            return None, None, False
        #print(l)
        # 删除的结点的id
        del_node = l[pruned_filter_idx]
        
        del_node_list = []                
        del_connects = []
        
        in_dict, out_dict = self.compute_in_out_degree()
                               
        for in_node,out_node in self.old_connections:
            if del_node == in_node or del_node == out_node:
                # 输出层结点
                if out_node < self.num_outputs and out_node >= 0:
                    if len(in_dict[out_node]) == 1:
                        print('prune ' , in_node, '---> output ', out_node, ' forbid')
                        return None, None, False
                # 输入层结点
                elif in_node < 0 and len(out_dict[in_node]) == 1:
                    print('in pins ' , in_node, '---> output ', out_node, ' forbid')
                    return None, None, False    
                
                del_connects.append((in_node, out_node))
                
                in_dict[out_node].remove(in_node)
                out_dict[in_node].remove(out_node)
                
                if del_node == in_node and len(in_dict[out_node]) == 0:
                                     
                    l1 = [out_node]
                    
                    while  len(l1) >0:# 入度in为零的结点的list
                        l2 = []
                        
                        for out in l1:                                                                                          
                            for out_ in out_dict[out]:                            
                                in_dict[out_].remove(out)
                                
                                if len(in_dict[out_]) == 0:
                                     # 输出层结点要小心
                                    if out_ < self.num_outputs and out_ >=0:
                                        in_dict[out_].add(out) # 注意要补回来
                                        print('ouput node ' ,out_ ,' <-- ', out ,' cannot deleted')
                                        # TODO: out_ <-- out <-- ...should be saved
                                        return None, None, False
                                        
                                    else:
                                        l2.append(out_)
                                # 待删除的连接                                        
                                del_connects.append((out, out_))                                               
                            out_dict[out] = set()
                            
                            #self.old_layer[int(to_prune_layer_idx/2)][1].remove(del_node)
                            
                            del_node_list.append(out)
                            #self.old_layer[ self.old_nodes[out].layer ][1].remove(out)
                            #del self.old_nodes[out] # 删除 结点
                            
                        l1 = l2     
                                                           
                if del_node == out_node and len(out_dict[in_node]) == 0:
                
                    l1 = [in_node]
                    
                    while  len(l1) >0:# 出度out为零的结点的list
                        l2 = []
                        
                        for in_ in l1:  # 出度out为零的结点                                                                                        
                            for in_1 in in_dict[in_]:  # previous layer node
                                out_dict[in_1].remove( in_ )
                                
                                if len(out_dict[in_1]) == 0:
                                     # 输出层结点
                                    if in_1 < 0: # input pins 
                                        out_dict[in_1].add(in_) # 补回来
                                        print('in pin ' ,in_1 ,' <-- ', in_ ,' cannot delete')
                                        
                                        return None, None, False
                                        
                                    else:
                                        l2.append(in_1)
                                # 待删除的连接                                        
                                del_connects.append((in_1, in_))                                               
                            in_dict[in_] = set()
                            
                            #self.old_layer[int(to_prune_layer_idx/2)][1].remove(del_node)
                            
                            del_node_list.append(in_)
                            #self.old_layer[ self.old_nodes[in_].layer ][1].remove(in_)
                            #del self.old_nodes[in_] # 删除 结点
                            
                        l1 = l2     
                                        
                    
        #del self.old_connections[(in_node, out_node)]
        for del_con in del_connects:
            del self.old_connections[del_con]  
            # cnn layer nodes
            if self.old_nodes[del_con[1]].layer < self.num_cnn_layer:
                # delete in_nodes
                #print( del_con[1],' <- ',del_con[0] )
                #print('in_nodes', self.old_nodes[del_con[1]].in_nodes)
                
                for idx_ in range(len(self.old_nodes[del_con[1]].in_nodes)):
                    
                    if self.old_nodes[del_con[1]].in_nodes[idx_] == del_con[0]:
                        j = idx_+1
                        while j < len(self.old_nodes[del_con[1]].in_nodes):
                            self.old_nodes[del_con[1]].in_nodes[j - 1] = \
                                self.old_nodes[del_con[1]].in_nodes[ j ]
                                
                            self.old_nodes[ del_con[1] ].kernal[ j -1 ] = \
                                self.old_nodes[ del_con[1] ].kernal[ j ]
                                
                            j += 1                  
                            
                        if len(self.old_nodes[del_con[1]].kernal) > 1:                            
                            self.old_nodes[ del_con[1] ].kernal = \
                                self.old_nodes[del_con[1]].kernal[:-1]
                            
                        del self.old_nodes[del_con[1]].in_nodes[ -1 ]  
                        
                        assert  len(self.old_nodes[ del_con[1] ].in_nodes)== 0 or \
                                    len(self.old_nodes[del_con[1]].kernal) == \
                                        len(self.old_nodes[ del_con[1] ].in_nodes)
                        
                        break
                           
                            
        if self.old_nodes.__contains__(del_node):
            
            del self.old_nodes[del_node] # 删除 结点
            #self.old_layer[int(to_prune_layer_idx/2)][1].remove(del_node)
            self.old_layer[int(to_prune_layer_idx/1)][1].remove(del_node)
        else:
            print(del_node , ' already deleted ')

        for del_no in del_node_list: # 因为 del_node 的delete 而 delete 的 node
            
            if self.old_nodes.__contains__(del_no):

                node_type = 2
                prune_layer = self.old_nodes[del_no].layer
                
                if prune_layer < self.num_cnn_layer:
                    node_type = 4
                cur_idx = 0
                for p in self.parameters():
                    
                    if len(p.data.size()) == 2 or len(p.data.size()) == 4: 
                        
                        if cur_idx == prune_layer:
                                                            
                            del_no_pos = self.nodes[ del_no ][1]                                
                            p.data = torch.cat([p.data[0: del_no_pos,:], p.data[del_no_pos+1 :,:] ], dim=0 )
                                                                                    
                        elif cur_idx == prune_layer + 1:
                            
                            del_no_pos = self.nodes[ del_no ][1]                                      
                            p.data = torch.cat( [p.data[:, 0: del_no_pos]  , p.data[:, del_no_pos+1:] ] ,dim=1  )
                            
                            for nod in self.old_layer[prune_layer][1]:
                                if self.nodes[nod][1] > del_no_pos:
                                    self.nodes[nod][1] -= 1
                            
                            self.nodes[del_no] = -123                            
                            del self.nodes[del_no]
                            
                            break
                        
                        cur_idx += 1
                        
                    elif len(p.data.size()) == 1: # bias                        
                        
                        if cur_idx == prune_layer + 1:
                            del_no_pos = self.nodes[ del_no ][1]   
                            p.data = torch.cat([p.data[: del_no_pos] , p.data[del_no_pos+1:]], dim=0)
                            
                            if node_type == 4 and prune_layer == self.num_cnn_layer -1: # add
                                node_type = 2          
                            
                self.old_layer[ self.old_nodes[del_no].layer ][1].remove( del_no )
                del self.old_nodes[del_no] # 删除 结点        
            else:
                print(del_no , ' has been deleted ')                
            
        idx = 0 
        #TODO: idx 不能代表层数 idx < self.num_cnn_layer
        for p in self.parameters():            
            if len(p.data.size()) == 4: # conv weight
                if idx < to_prune_layer_idx:
                    idx += 1
                    continue
                if idx == to_prune_layer_idx:
                    
                    p.data = torch.cat([p.data[: pruned_filter_idx] , p.data[pruned_filter_idx+1:]], dim=0)
                
                elif idx == to_prune_layer_idx + 1 and \
                    idx < 1 * self.num_cnn_layer: # 下一层卷积层
                    # 这里和一般的卷积层不一样，第二维度固定是1，第一维度和输入维度相同                    
                    #p.data = torch.cat([p.data[0: pruned_filter_idx,:,:,: ] , p.data[pruned_filter_idx+1:,:, :,:]],dim=0)
                    p.data = torch.cat([p.data[:,0: pruned_filter_idx,:,: ] , p.data[:,pruned_filter_idx+1:, :,:]],dim=1) 
                    
                    del_no = -234          
                    for k, pos in self.nodes.items():
                        if pos[0] == to_prune_layer_idx and pos[1] == pruned_filter_idx:
                            del_no = k
                            break
                    
                    assert del_no > 0
                    
                    for nod in self.old_layer[to_prune_layer_idx][1]:
                        
                        if self.nodes[nod][1] >  pruned_filter_idx:
                            self.nodes[nod][1] -= 1
                    
                    self.nodes[ del_no ] = -123                            
                    del self.nodes[del_no]
                    
                    
                    break # 跳出循环                 

                       
                idx += 1

            elif len(p.data.size()) == 2 : # fc weight                
                if idx == 1 * self.num_cnn_layer:
                    #print('p.data.size() ', p.data.size())
                    p.data = torch.cat([p.data[:, 0: pruned_filter_idx], p.data[:, pruned_filter_idx+1 :] ], dim=1  )                                            
                    #print('After prune , p.data.size() ', p.data.size())
                    del_no = -333          
                    for k, pos in self.nodes.items():
                        if pos[0] == to_prune_layer_idx and pos[1] == pruned_filter_idx:
                            del_no = k
                            break
                    
                    assert del_no > 0
                    
                    for nod in self.old_layer[to_prune_layer_idx][1]:
                        
                        if self.nodes[nod][1] >  pruned_filter_idx:
                            self.nodes[nod][1] -= 1
                    
                    self.nodes[ del_no ] = -123                            
                    del self.nodes[del_no]
                    
                    break
                
            elif len(p.data.size()) == 1 : # bias
                
                if idx  == to_prune_layer_idx + 1 :
                    p.data = torch.cat([p.data[: pruned_filter_idx] , p.data[pruned_filter_idx+1:]], dim=0)
                    
                                            
#        print('Prune filter #{} gene_id #{} in layer #{}'.format(
#            pruned_filter_idx,
#            del_node,
#            to_prune_layer_idx))                
            
        return del_node, del_connects, True
    
    def set_layers(self, genome: neat1.genome.DefaultGenome):
        #calculate channel for every cnn layers
        cnn_channel = list()
        cnn_channel.append(self.num_inputs) # 输入图片RGB的通道数 num_inputs=3
        
        for i in range(self.num_cnn_layer):
            # 每一层卷积层的卷积核数量=num_feature_map=输出的通道数=第i层结点数量
            cnn_channel.append(len(genome.layer[i][1]))

        #add cnn layers
        layer_id = 0
        for i in range(self.num_cnn_layer):
            #  torch.nn.Module.add_module(self, name, module)
#            self.add_module(str(layer_id), nn.Conv2d(in_channels = cnn_channel[i], out_channels = cnn_channel[i+1], kernel_size = 1, padding = 0))
#            layer_id += 1
            
            #self.min_cnn_channels.append(cnn_channel[i+1])
            
            # groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
            #self.add_module(str(layer_id), nn.Conv2d(in_channels = cnn_channel[i+1], out_channels = cnn_channel[i+1], kernel_size = 3, padding = 1, groups = cnn_channel[i+1]))
            self.add_module(str(layer_id), nn.Conv2d(in_channels = cnn_channel[i], out_channels = cnn_channel[i+1], kernel_size = 3, padding = 1))
            layer_id += 1
            
            #self.min_cnn_channels.append(cnn_channel[i+1])

        #calculate channel for every cnn layers
        fc_channel = list()
        
        # 输出x的长宽 (4 x  4) , 第一层fc层需要flatten成 
        # num_final_cnn_channels * (4 x 4) = num_final_cnn_channels * 16          
        # 第一个fc 的输出结点数 恰好是 16
        
        # 最后一层conv2d的通道数* 第一层 fc层的结点数
#        self.num_flat_features = len(genome.layer[self.num_cnn_layer - 1][1]) * \
#                                 len(genome.layer[self.num_cnn_layer][1])

        # nn.AvgPool(4)(x) 全局平均池化
        self.num_flat_features = len(genome.layer[self.num_cnn_layer-1][1])                                          
        
        # 每一层 fc层的结点数量
        fc_channel.append(self.num_flat_features)
                
        
        for i in range(self.num_cnn_layer, self.num_layer):
            fc_channel.append(len(genome.layer[i][1]))

        #add fc layer
        for i in range(self.num_layer - self.num_cnn_layer):
            self.add_module(str(layer_id), nn.Linear(fc_channel[i], fc_channel[i+1]))
            layer_id += 1

        #set all weight and bias to zero
        for module in self.children():
            module.weight.data.fill_(0.0)
            module.bias.data.fill_(0.0)

    def set_parameters(self, genome: neat1.genome.DefaultGenome):

        layer = list(self.children())#make sure change layer can affect parameters in cnn

        nodes = {}


        #add the input node to nodes dict
        order = 0 # order 是 结点在当前层layer的下标
        for i in range(-self.num_inputs, 0):
            position = [-1, order]   #-1 means input node
            nodes.update({i: position})
            order += 1

#        print('genome.nodes[15].in_nodes' , genome.nodes[301].in_nodes)
        #add every layers to nodes dict
        for i in range(self.num_layer):
            #print('i', i)
            l = list(genome.layer[i][1]) # l是 结点 id 的 list
            l.sort()
            order = 0
            for j in range(len(l)):
             #   print('j', j, 'l[j] ', l[j])
                #add node (id, [layer, order in layer])
                position = [i, j]
                # l[j]: id of the jth node in ith layer
                nodes.update({l[j]: position})

                #add conv kernel and bias to pytorch module
                if i < self.num_cnn_layer:                   
                    a = np.array(self.old_nodes[l[j]].kernal)
#                    if i == 0:
#                        print('i ===   0')
#                        print(a)
                    #print('gg')
                    # TODO: 为什么是 2*i+1层？？
                    # 卷积核是 二维的？？ 
                    # 只有 一个输入通道只对应一个输出通道
                    # 即 (data[j] = data[j][0])才能这么做，
                    # 否则，data[j][:] 该输出通道 所有输入通道的 w 都一样
                    #layer[i * 2 + 1].weight.data[j] = torch.FloatTensor(a.reshape(3, 3))
                    # TODO: 卷积层  len(a) == 1 ？
                    if (len(self.old_nodes[l[j]].in_nodes) == 0 or len(self.old_nodes[l[j]].in_nodes) == len(a)) == False:
                        print('layer %d , %d -th ,node_id %d '%(i, j ,l[j]))
                        print(len(self.old_nodes[l[j]].in_nodes), ' ',len(a) )
                    assert len(self.old_nodes[l[j]].in_nodes) == 0 or len(self.old_nodes[l[j]].in_nodes) == len(a)
#                    if i == 0:
#                        print('i == 0', l[j],self.old_nodes[l[j]].in_nodes)
                    if i == 0 and len(self.old_nodes[l[j]].in_nodes) == 0:
                        print(str(l[j]) , ' no_input to first cnn layer')
                        #raise RuntimeError( str(l[j]) + ' no_input to first cnn layer')
                    #else:   print(str(l[j]))
                    if len(self.old_nodes[l[j]].in_nodes) > 0:
                        if i == 0:
                            pass
                            #print('i ===   0')
                            #print(a)
                        for k in range(len(a)):
#                            print('k', k, self.old_nodes[l[j]].in_nodes, genome.nodes[l[j]].in_nodes)
                            #print('k', k, self.old_nodes[l[j]].in_nodes)
                            #print(nodes[ self.old_nodes[l[j]].in_nodes[k]])
                            layer[i * 1 + 0 ].weight.data[j][nodes[ self.old_nodes[l[j]].in_nodes[k]][1] ] = \
                               torch.FloatTensor(a[k].reshape(3, 3))
                    
                    #print(len(a), len(a[0]))
                    #print(layer[i * 2 + 1].weight.data.size())
                    #print('hh')
                    b = self.old_nodes[l[j]].bias
                    #print(b.size())
#                    if j >=  layer[i * 2 + 1].bias.data.size()[0]:
#                        print(layer[i * 2 + 1].bias.data.size())
#                    print(j, layer[i + 0 ].bias.data.size())
                    layer[i + 0 ].bias.data[j] = torch.FloatTensor([b])
                    #print('kk')
                else:
                    b = self.old_nodes[l[j]].bias
                    layer[i + 0 ].bias.data[j] = torch.FloatTensor([b])                    
        
        self.nodes = nodes.copy()
        
        #print('hh')
        for in_node, out_node in genome.connections:
            # 输出结点所在的层layer                
            c = nodes[out_node][0] #layer number
            #print('kk')
            if c < self.num_cnn_layer: #cnn layer
                pass
                # TODO: 层数为什么要 *2 ？？？
                # 2c层 卷积核为 1x1, bias 全为 0
#                layer_num = 2 *c
                # nodes[out_node] 取出 out_node 的位置 [层数i,结点在该层的位置j]
                # nodes[out_node][1] 取出 结点位置j
#                weight_tensor_num = nodes[out_node][1]
#                weight_num = nodes[in_node][1]
                
#                (layer[layer_num].weight.data[weight_tensor_num])[weight_num] = \
#                    torch.FloatTensor([genome.connections[(in_node, out_node)].weight])
            
            elif c != self.num_cnn_layer:
                layer_num = 0 + c
                #print('qq')
                weight_tensor_num = nodes[out_node][1]
                weight_num = nodes[in_node][1]
                
                (layer[layer_num].weight.data[weight_tensor_num])[weight_num] = \
                    torch.FloatTensor([genome.connections[(in_node, out_node)].weight])
                #print('ww')                    
            else:
                #print('ll')
                #print(len(layer[6].weight[0]))
                layer_num = 0 + c
                
                weight_tensor_num = nodes[out_node][1]
                # weight_num 表示的是权重对应的下标，不是数量？？？                
                #weight_num = nodes[in_node][1] * self.num_first_fc_layer_node + nodes[out_node][1]
                weight_num  = nodes[in_node][1]
                #print(weight_num)
                (layer[layer_num].weight.data[weight_tensor_num])[weight_num] = \
                    torch.FloatTensor([genome.connections[(in_node, out_node)].weight])
                #print('mm')                        

    def write_back_parameters(self, genome: neat1.genome.DefaultGenome):

        layer = list(self.children())#make sure change layer can affect parameters in cnn

        nodes = {}        
        order = 0
        for i in range(-self.num_inputs, 0):
            position = [-1, order]   #-1 means input node
            nodes.update({i: position})
            order += 1
        first = True            
        for i in range(self.num_layer):
            l = list(genome.layer[i][1])
            l.sort()
            for j in range(len(l)):
                # (id, [layer, order in layer] )
                position = [i, j]
                nodes.update({l[j]: position})
                                
                if i < self.num_cnn_layer:
                    
                    a1 = np.array(self.old_nodes[l[j]].kernal)                    
                    a = np.array(layer[ i ].weight.data[j].cpu())                            
                    
                    #genome.nodes[l[j]].kernal = a.reshape(9)
                          
#                    print('layer %d , %d -th ,node_id %d '%(i, j ,l[j]))
#                    print(len(self.old_nodes[l[j]].in_nodes), ' ',len(a), len(a1) )
                    
                    assert len(self.old_nodes[l[j]].in_nodes) == 0 or len(self.old_nodes[l[j]].in_nodes) == len(a1)

                    if i == 0 and len(self.old_nodes[l[j]].in_nodes) == 0:
                        print(str(l[j]) , ' no_input to first cnn layer')

                    if len(self.old_nodes[l[j]].in_nodes) > 0:  
                        
                        for k in range(len(a1)):
                            
                            genome.nodes[l[j]].kernal[k] = (a[ nodes[ self.old_nodes[l[j]].in_nodes[k] ][1] ]).reshape(9)
                            
#                            layer[i * 1 + 0 ].weight.data[j][nodes[ self.old_nodes[l[j]].in_nodes[k]][1] ] = \
#                               torch.FloatTensor(a[k].reshape(3, 3))
                    
                    #print(len(a), len(a[0]))
                    #if i == 0 and first:
                        #first = False
                        #print(j,' ', l[j] ,' ', genome.nodes[l[j]].bias)                        
                        #print(layer[i].bias.data[j].item()  )
                        #print(layer[i].bias.data, '\n')
                        #genome.nodes[l[j]].bias = layer[i].bias.data[j].item()                
                        #print(genome.nodes[l[j]].bias)
                    
                    genome.nodes[l[j]].bias = layer[i].bias.data[j].item()                
                    #if i == 0: 
                        #print(j,' ', l[j] ,' ', genome.nodes[l[j]].bias)                        
                        #print(layer[i].bias.data[j].item()  )
                        #print(layer[i].bias.data, '\n')
                        #genome.nodes[l[j]].bias = layer[i].bias.data[j].item()                
                        #print(genome.nodes[l[j]].bias)
                else:
                    genome.nodes[l[j]].bias = layer[i].bias.data[j].item()

        #TODO: add write back
        for in_node, out_node in genome.connections:

            c = nodes[out_node][0] #layer number
            if c < self.num_cnn_layer: #cnn layer
                pass

            elif c != self.num_cnn_layer:
                layer_num = 0 + c
                weight_tensor_num = nodes[out_node][1]
                weight_num = nodes[in_node][1]
                genome.connections[(in_node, out_node)].weight = \
                    (layer[layer_num].weight.data[weight_tensor_num])[weight_num].item()
            else:
                
                layer_num = 0 + c
                
                weight_tensor_num = nodes[out_node][1]
            
                #weight_num = nodes[in_node][1] * self.num_first_fc_layer_node + nodes[out_node][1]
                weight_num  = nodes[in_node][1]
                #print(weight_num)
                genome.connections[(in_node, out_node)].weight = \
                (layer[layer_num].weight.data[weight_tensor_num])[weight_num].item()
                               