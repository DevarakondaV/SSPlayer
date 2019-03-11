from sumTree import sumTree
import numpy as np

class experience:

    def __init__(self,capacity):
        
        self.beta = 0.4
        self.epsilon = 0.01
        self.tree = sumTree(capacity)
    
    def __len__(self):
        return self.tree.data_i
    
    def store(self,experience):
        """
            Adds new experience to the tree
            
            args:
                experience: SARSA tuple
        """
        #New experience needs maximum priority

        priority_max = np.max(self.tree.tree[-self.tree.capacity:])

        #When tree is completely empty
        if (priority_max == 0):
            priority_max = 1

        self.tree.add(priority_max,experience)

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
        
        for i in range(0,batch_size):
            #print(p_seg*i,p_seg*(i+1))
            rand_p = np.random.uniform(p_seg*i,p_seg*(i+1),1).astype(np.float32)[0]
            leaf_idx,priority_i,exp_i = self.tree.get_leaf(rand_p)
            # print(self.tree.get_total_priority(),rand_p)
            # print(leaf_idx,self.tree.data_i)

            try:
                img_1.append(exp_i[0])
                action.append(exp_i[1])
                reward.append(exp_i[2])
                img_2.append(exp_i[3])
            except TypeError:
                print(rand_p)
                print(self.tree.tree)

            tree_idx.append(leaf_idx)

            probability_i = (priority_i+self.epsilon)/self.tree.get_total_priority()

            IS_weights.append(((1/self.tree.capacity)*(1/probability_i))**self.beta)
        
        IS_weights /= max(IS_weights)

        img_1 = np.asarray(img_1)
        img_2 = np.asarray(img_2)
        reward = np.reshape(np.asarray(reward),(batch_size,1))
        action = np.reshape(np.asarray(action),(batch_size,1))

        return tree_idx,np.asarray(IS_weights),(img_1,action,reward,img_2)

    def update(self,idx,priority):
        priorities = np.abs(priority) #+self.epsilon

        clipped_priorities = np.minimum(priorities,1)
        for i,p_val in zip(idx,clipped_priorities):
            self.tree.set_priority(i,p_val)
