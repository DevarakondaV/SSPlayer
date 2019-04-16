from sumTree import sumTree
import numpy as np
import sys
class experience:

    def __init__(self,capacity):
        
        self.beta = 0.4
        self.alpha = 0.6
        self.epsilon = 0.001
        self.max =  1
        self.total_exp = 0
        self.tree = sumTree(capacity)
    
    def __len__(self):
        return self.tree.data_i
    
    
    def store(self,error,experience):
        """
            Adds new experience to the tree
            
            args:
                experience: SARSA tuple
        """
        #New experience needs maximum priority

        # priority_max = np.minimum(np.max(self.tree.tree[-self.tree.capacity:]),1)
        error = np.abs(error)
        self.total_exp +=1
        error += self.epsilon
        clipped_e = np.clip(error,0,self.max)
        p_val = np.power(clipped_e,self.alpha)
        self.tree.add(p_val,experience)

        return

    def sample(self,batch_size):
        """
            sample a batch of experiences

        """

        self.beta = np.min([1.0,self.beta+.001])

        p_seg = self.tree.get_total_priority()/batch_size
        IS_weights = []
        
        img_1 = []
        img_2 = []
        action = []
        reward = []

        tree_idx = []

        if (self.total_exp > self.tree.capacity):
            min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.get_total_priority()
        else:
            min_prob = np.min(self.tree.tree[-self.tree.capacity:self.tree.capacity-1+self.tree.data_i]) / self.tree.get_total_priority()

        maxwi = np.power(self.tree.capacity*min_prob,-self.beta)

        for i in range(0,batch_size):
            
            try:
                rand_p = np.random.uniform(p_seg*i,p_seg*(i+1),1)#.astype(np.float16)[0]
            except OverflowError:
                print(p_seg*i,p_seg*(i+1))
                print(self.tree.get_total_priority())
            leaf_idx,priority_i,exp_i = self.tree.get_leaf(rand_p)

            try:
                img_1.append(exp_i[0])
                action.append(exp_i[1])
                reward.append(exp_i[2])
                img_2.append(exp_i[3])
            except TypeError:
                leaf_idx,priority_i,exp_i = self.tree.get_leaf(rand_p,p=True)
                print(rand_p)

            tree_idx.append(leaf_idx)



            probability_i = (priority_i/self.tree.get_total_priority())
            IS_weights.append(self.tree.capacity*probability_i)

        IS_weights = np.vstack(IS_weights)
        IS_weights = np.power(IS_weights,-self.beta) / maxwi
        img_1 = np.asarray(img_1).squeeze()
        img_2 = np.asarray(img_2).squeeze()
        reward = np.reshape(np.asarray(reward),(batch_size,1))
        action = np.reshape(np.asarray(action),(batch_size,1))
        return tree_idx,IS_weights,(img_1,action,reward,img_2)

    def update(self,idx,error):
        error = np.abs(error)
        error += self.epsilon
    
        clipped_e = np.clip(error,0,self.max)
        clipped_e = np.power(clipped_e,self.alpha)
        for i,p_val in zip(idx,clipped_e):
            self.tree.set_priority(i,p_val)
