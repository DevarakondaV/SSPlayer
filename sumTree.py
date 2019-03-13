import numpy as np
import sys
import time

class sumTree:

    def __init__(self,cap):
        """
            Function initializes a sumtree data structure
            args:
                cap. Int. Total number of elements
                batch_size. Int. Size of mini batch
        """
        self.capacity = cap
        self.data_i = 0

        self.data = np.zeros([self.capacity],dtype=np.object)

        # self.tree = np.zeros([2*self.capacity-1],dtype=np.float16)
        self.tree = np.zeros([2*self.capacity-1])
        # Divide equally into k range [0 to p_total]
        # Uniformally samples from each range.
        # Finally the transittions that correspond to each of these sampled values are retrieved from the tree. OVerhead is similar to rank-based pri.
        
        return
    
    def add(self,p_val,data):
        """
            Adds priority for new transition

            args:
                p_val: float16.
                data: (Tuple of transitions)
        """
        
        self.data[self.data_i] = data
        tree_i = self.capacity-1+self.data_i
        
        self.update(tree_i,p_val)
        if (self.data_i == self.capacity-1):
            self.data_i = 0
        else:
            self.data_i += 1

        return


    def update(self,tree_i,p_val):
        """
            Function recursively updates the tree
            
            args:
                tree_i: int. Index from which to update tree
        """
        delta = p_val-self.tree[tree_i]
        self.tree[tree_i] = p_val


        while (tree_i != 0):
            tree_i = (tree_i - 1) // 2
            self.tree[tree_i] += delta


    def set_priority(self,leaf_i,p_val):
        """
            Function updates priority in tree and updates all
        
            args:
                leaf_i: int. Index of tree being modified
                p_val:  new priority
        """


        self.update(leaf_i,p_val)

        return

    def get_leaf(self,p_val,p=False):
        """

            Function returns leaf index, priority and data 
            where p_val
        """

        node_idx = 0

        while True:
            left_child_node = 2*node_idx+1
            right_child_node = left_child_node+1
            if (p):
                print([p_val,self.tree[node_idx],self.tree[left_child_node],self.tree[right_child_node]],node_idx)

            if (left_child_node >= len(self.tree)):
                leaf_idx = node_idx
                break
            else :
                if ( p_val <= self.tree[left_child_node]):
                    node_idx = left_child_node
                else:
                    p_val -= self.tree[left_child_node]
                    node_idx = right_child_node
            
        data_i = leaf_idx+1-self.capacity
        return leaf_idx, self.tree[leaf_idx],self.data[data_i]

    
    def get_total_priority(self):
        return self.tree[0]


