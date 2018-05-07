import numpy as np
import tensorflow as tf
import time
import os


transfer_function = tf.nn.relu # activation function applied to the sequence convolutions
alphabet = list('ACGT')        # the only valid sequence symbols (case-sensitive!)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('motif', 'ACGTAC',
                    'Sequence motif. May only contain A, C, G or T as letters. Must be shorter than sequence_length.')

flags.DEFINE_integer('filter_length', -1,
                     'Length of convolutional filter.')

flags.DEFINE_float('learning_rate', 0.1, 
                   'Learning rate for Adam Optimizer.')

flags.DEFINE_integer('sequence_length', 40,
                     'Length of training sequences. Must be longer than motif.')

flags.DEFINE_integer('training_steps', 3000,
                     'Number of training steps.')

flags.DEFINE_integer('display_steps', 500,
                     'Display accuracy on training data every display_steps steps. Nothing is displayed if set to 0.')

flags.DEFINE_string('gpus', '',
                    'String indicating which GPUs to use. For example, 1,2 indicates that GPUs 1 and 2 shall be used.') 

flags.DEFINE_integer('num_positive', 100,
                     'Number of positive training examples.')

flags.DEFINE_integer('num_negative', 10000,
                     'Number of negative training examples.')

flags.DEFINE_integer('batch_size', 100,
                     'Batch size. Half of the batch are positive examples, the other half are negative examples.')

flags.DEFINE_float('regul', 0.01, 'L2 regularization strength.')


def check_args(flags):
    assert (flags.num_positive > 0), 'Number of positive examples must be greater than 0.'
    assert (flags.num_negative > 0), 'Number of negative examples must be greater than 0.'   
    assert (flags.sequence_length > 0), 'Sequence length must be greater than 0.'          
    assert (len(flags.motif) > 1), 'Motif length must be greater than 1.'     
    assert (sum([1 for letter in flags.motif if letter in alphabet])==len(flags.motif)), 'Motif must only contain upper case letters A, C, G or T'
    assert (flags.batch_size < flags.num_negative), 'Batch size must be smaller than number of negative training examples.'      
    assert (flags.batch_size % 2)==0, 'Batch size must be divisible by 2.'           
    assert ((flags.num_negative % flags.num_positive) ==0), 'Number of negative examples must be a multiple of the number of positive examples.'
    assert (len(flags.motif) < flags.sequence_length), 'Motif cannot be longer than sequence'
    
    # if not filter length has been specified, use a filer that is as long as the motif ...
    if flags.filter_length == -1:
        flags.filter_length = len(flags.motif)


check_args(FLAGS)
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpus



# ------------------------------------------------------
# Converts a string 'ACGT' into one-hot coding
# ------------------------------------------------------

def get_motif_from_string(motif):
    
    onehot_dict = dict(zip(list('ACGT'),np.arange(4)))
    
    v = np.array([onehot_dict[key] for key in motif]).reshape([-1,1])
    h = np.arange(4).reshape([1,-1])

    return np.equal(v,h)*1





# ------------------------------------------------------
# Function for generating data. 
# It places a specified motif at a random position in 
# each positive training sequence. Then, it randomly 
# shuffles the nucleotide order of each positive 
# sequence to generate negative training examples.
# ------------------------------------------------------


def generate_data(num_pos, num_neg, seq_len, motif):
    
    motif_len = len(motif)
    pos_data = (np.random.randint(0,4,[num_pos, 1,seq_len, 1]) == np.reshape(np.arange(4),[1,1,1,4]))*1.0 # background data
    for s in range(num_pos):
        q = np.random.randint(seq_len-motif_len+1)
        pos_data[s,0,q:q+motif_len,:] = get_motif_from_string(motif)

    neg_data = pos_data[np.random.randint(0, num_pos, num_neg),:,:] # randomly sample from positive data
    idx = np.arange(seq_len)
    for b in range(num_neg):           
        np.random.shuffle(idx)
        seq = neg_data[b,:,:,:]
        neg_data[b,:,:,:] = seq[:,idx,:]   
        
       
    
    multiplier = num_neg // num_pos
    
    pos_data = np.concatenate([pos_data]*multiplier)

    if len(pos_data) != len(neg_data):
        raise('number of positive examples is not the same as the number of negative examples.')

    return pos_data, neg_data   





# ------------------------------------------------------
# Data set class. Contains the data as well as a method 
# for generating examples batches for training.
# TODO: replace with tf.dataset
# ------------------------------------------------------


class mydata(object):
    
    def __init__(self, num_pos_train, num_neg_train, seq_len, motif):
        
        self.motif_len = len(motif)
        [pos_data_train, neg_data_train] = generate_data(num_pos_train, num_neg_train, seq_len, motif)

        self.pos_data_train = pos_data_train
        self.neg_data_train = neg_data_train

        self.num_pos_train  = num_pos_train
        self.num_neg_train  = num_neg_train
        self.train_idx      = 0

    def next_batch(self, bs):
  
        x_batch  = np.concatenate([self.pos_data_train[self.train_idx:self.train_idx+bs//2],
                                   self.neg_data_train[self.train_idx:self.train_idx+bs//2]],axis=0)

        if self.train_idx + bs >= self.num_neg_train:
            self.train_idx = 0
            np.random.shuffle(self.neg_data_train)
            np.random.shuffle(self.pos_data_train)
        else:
            self.train_idx += bs//2
        
        return x_batch, np.concatenate([np.ones([bs//2]), np.zeros([bs//2])])
        





# ------------------------------------------------------
# The following is done to generate circular filters.
# Example:
# "weights" is a 1x6x4x1 array. We want to take out 6 slices
# of size 1x1x4x1 and arrange them 
# according to ind = [0,1,2,3,4,5] etc.
# This is done by creating a list "L" of 1x1x4x1 slices.
# The 6 correctly arranged filter slices in this list are then 
# concatenated into an 1x6x4x1 filter.
# This is done 6 times, and all filters are concatenated 
# into a 1x6x4x6 kernel.
# ------------------------------------------------------

def circular_kernel(W, weight_indices):
    W_split = tf.split(W, num_or_size_splits=FLAGS.filter_length, axis=1)
    
    kernel_list = []                    # List of all circular filters    
    for ind in weight_indices:          # go through all filter indices
        L = [W_split[i] for i in ind]   # list of filter slice arrays
        temp = tf.concat(L,axis=1)      # concatenate list to filter of size 1 x length x 4 x 1
        kernel_list.append(temp)        

    kernel = tf.concat(kernel_list,3)   # concatenate elements of "kernel_list" 
    return kernel




def cnn_with_circular_fiters(x, weights, weight_indices):

    conv = transfer_function(tf.nn.conv2d(x, circular_kernel(weights['w_filter'],weight_indices), [1, 1, 1, 1], padding='VALID'))
    pool = tf.reduce_max(conv,reduction_indices = 2,keepdims=True) # None x 1 x 1 x num_filters
    act  = tf.reshape(pool,[-1,FLAGS.filter_length], name='reshape_pool1')

    out  = tf.add(tf.matmul(act, weights['w_out']), weights['b_out']) 
    out  = tf.matmul(tf.reshape(out,[-1,1],name='reshape_out'), tf.constant([1.0,-1.0], dtype=tf.float32, shape=[1,2]))

    return out, act




def regular_cnn(x, weights):
    
    conv = transfer_function(tf.nn.conv2d(x,  weights['w_filter'], [1, 1, 1, 1], padding='VALID'))
    pool = tf.reduce_max(conv,reduction_indices = 2,keepdims=True) 
    act  = tf.reshape(pool,[-1,1], name='reshape_pool1')
    
    out  = tf.add(tf.matmul(act, weights['w_out']), weights['b_out'])
    out  = tf.matmul(tf.reshape(out,[-1,1],name='reshape_out'), tf.constant([1.0,-1.0], dtype=tf.float32, shape=[1,2]))

    return out, act





# ------------------------------------------------------
# Build graph and train the model based on "data".
# Depending on the value of "use_circular_filters", 
# circular filters will be used or not.
# ------------------------------------------------------


def build_and_train(data, use_circular_filters):

    g = tf.Graph()
    with g.as_default():

        
        # ------------ set up graph ------------
        # Define not trainable Network variables
        x           = tf.placeholder(tf.float32, shape=[None,1,FLAGS.sequence_length,4]) # data
        y           = tf.placeholder(tf.int64,   shape=[None])             # labels
        

        
        
        # ------------ set up network ------------
        if use_circular_filters:
            
            # weight_indices is a list of index-lists; 
            # each index-list contains information on the order of the circular filters.
            weight_indices=[]
            orig=list(range(FLAGS.filter_length))
            for n in range(FLAGS.filter_length):
                orig.insert(0,orig.pop(-1))
                weight_indices.append(orig[:])    

            weights = {
                'w_filter':  tf.Variable(tf.truncated_normal([1, FLAGS.filter_length, 4, 1],stddev=FLAGS.learning_rate),name='trainable_var_1',trainable=True),
                'w_out':     tf.Variable(tf.truncated_normal([FLAGS.filter_length, 1],stddev=FLAGS.learning_rate), name='trainable_var_2',trainable=True),
                'b_out':     tf.Variable(tf.constant(0.0,shape=[1]),name='trainable_var_3',trainable=True)    
            }          
            yhat, act   = cnn_with_circular_fiters(x, weights, weight_indices) # predictions & activations

        else:
            weights = {
                'w_filter':  tf.Variable(tf.truncated_normal([1, FLAGS.filter_length, 4, 1],stddev=FLAGS.learning_rate),name='trainable_var_1',trainable=True),
                'w_out':     tf.Variable(tf.truncated_normal([1, 1],stddev=FLAGS.learning_rate),name='trainable_var_2',trainable=True),
                'b_out':     tf.Variable(tf.constant(0.0,shape=[1]),name='trainable_var_3',trainable=True)    
            }            
            yhat, act   = regular_cnn(x,weights) # predictions & activations


            
            
        # ------------ set up optimizers etc. ------------
        ye          = tf.one_hot(indices=y, depth=2, axis=1)
        loss        = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=yhat, labels=ye)) + tf.nn.l2_loss(weights['w_filter'])*FLAGS.regul
        accuracy    = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(yhat,1),tf.argmax(ye,1)), tf.float32))    

        optimizer   = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        grads       = optimizer.compute_gradients(loss, var_list=tf.trainable_variables())   
        train_op    = optimizer.apply_gradients(grads)    

        init_op     = tf.global_variables_initializer()

        
        
        
        # ------------ set up session ------------
        config = tf.ConfigProto()
        config.allow_soft_placement     = True
        config.gpu_options.allow_growth = True  

        sess = tf.Session(config=config)
        sess.run(init_op) # re-initialize all variables



        # ------------ training------------
        acc_train_ema  = 0    # exponentially moving average of training accuracy
        emafactor      = 0.01 # update factor for exponentially moving average of training accuracy        
        
        start_time = time.time()
        for step in range(FLAGS.training_steps):

            [x_batch,y_batch] = data.next_batch(FLAGS.batch_size)
            _,accuracy_np     = sess.run([train_op,accuracy], feed_dict={x:x_batch, y:y_batch})
            acc_train_ema     = acc_train_ema*(1-emafactor) + accuracy_np*emafactor
            
            if (FLAGS.display_steps>0) and (step % FLAGS.display_steps==0):
                print('step %5d // training accuracy: %.3f' % (step, acc_train_ema))
        comp_time = time.time() - start_time
        
                
                
            
    
        # ------------ obtain inferred filter ------------
        w_out, w_filter = sess.run([weights['w_out'], weights['w_filter']])
        unsorted_filter_indices = np.argmax(np.squeeze(w_filter),axis=1)
        
        # when circular filters are used, it needs to be figures out 
        # which of the circular filters contains the unshifted motif.
        if use_circular_filters:
            inferred_filter_order = weight_indices[np.argmin(w_out)]
            sorted_filter_indices = unsorted_filter_indices[inferred_filter_order]
            inferred_filter       = ''.join([alphabet[i] for i in sorted_filter_indices])   # turn indices into nucleotides

        else:
            inferred_filter       = ''.join([alphabet[i] for i in unsorted_filter_indices]) # turn indices into nucleotides
    
    return inferred_filter





def main(argv):



    print('Creating data ...')
    data = mydata(FLAGS.num_positive, FLAGS.num_negative, FLAGS.sequence_length, FLAGS.motif)
    
    print('\n\n---------------------------------------------')
    print('Start training CNN with circular filters ... ')
    inferred_motif = build_and_train(data, True)
    print('... training done.')
    print('Original motif: %s' % FLAGS.motif)
    print('Inferred motif: %s' % inferred_motif)
    print('---------------------------------------------')
    
    print('\n\n---------------------------------------------')
    print('Start training regular CNN ... ')
    inferred_motif = build_and_train(data, False)
    print('... training done.')
    print('Original motif: %s' % FLAGS.motif)
    print('Inferred motif: %s' % inferred_motif)
    print('---------------------------------------------')



if __name__ == '__main__':
    tf.app.run(main)

