import tensorflow as tf
import os.path as osp

from HER.ddpg.models import Model
from HER.deepq.models import _mlp

class classifier:
    def __init__(self, name, in_shape, out_shape, sess=None, log_dir="/tmp"):
        # properties
        self.in_shape = in_shape
        self.out_shape = out_shape

        # tf
        self.in_tensor = tf.placeholder(tf.float32, shape=(None,) + (in_shape,), name='state_goal')
        self.out_tensor = _mlp(inpt=self.in_tensor, hiddens=[50,10],scope="suc_pred_model", num_actions= out_shape,layer_norm=True)
        self.pred = tf.sigmoid(self.out_tensor) > 0.5
        self.target_tensor = tf.placeholder(tf.float32, shape=(None,) + (out_shape,), name='final_state')
        
        self.sess = sess
        self.log_dir = log_dir
        # loss function 
        self.loss = tf.losses.sigmoid_cross_entropy(self.target_tensor, self.out_tensor)
        self.accuracy, _ = tf.metrics.accuracy(self.target_tensor, self.pred)

        # summary
        tf.summary.histogram("input", self.in_tensor)
        tf.summary.histogram("output", self.out_tensor)
        tf.summary.histogram("outputVstarget", self.target_tensor - self.out_tensor)
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accuracy", self.accuracy)
        self.sum = tf.summary.merge_all()

        # optim
        self.lr = tf.placeholder(tf.float32,(),'learning_rate')
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.optim = tf.train.AdamOptimizer(self.lr) \
                          .minimize(self.loss, var_list=(self.vars))

        # saver
        self.saver = tf.train.Saver(var_list = self.vars, max_to_keep=5, 
                                        name="%s_suc_pred_net"%name, 
                                        keep_checkpoint_every_n_hours=1,
                                        save_relative_paths=True)

        self.writer_t = tf.summary.FileWriter(osp.join(self.log_dir ,'tfsum' ,'train'), sess.graph)
        self.writer_v = tf.summary.FileWriter(osp.join(self.log_dir , 'tfsum', 'test'), sess.graph)


    def train(self, num_epochs, batch_size, lr=1e-3, train_dataset = None, test_dataset=None, test_freq = 1, save_freq=10, log=True):
        feed_dict = {self.lr :lr}
        
        for curr_epoch in range(num_epochs):

            # train_feats, train_label = self.sess.run(train_dataset)
            curr_train_data = train_dataset.sample(batch_size, axis=0).as_matrix()
            
            feed_dict[self.in_tensor] = curr_train_data[:, :self.in_shape]
            feed_dict[self.target_tensor] = curr_train_data[:,[-1]]

            train_summary, train_loss, _ = self.sess.run([self.sum, self.loss, self.optim], feed_dict)
            if log:
                print("epoch:%d, loss:%.5f"%(curr_epoch+1, train_loss))

                self.writer_t.add_summary(train_summary,curr_epoch)


            # test
            if (curr_epoch+1)%test_freq == 0:
                test_summary = self.test(test_dataset)
                if log:
                    self.writer_v.add_summary(test_summary,curr_epoch)


            if (curr_epoch+1)%save_freq == 0:
                self.save()

            # if (curr_epoch+1)%10:
            #     print("epoch:%d, loss:%.5f"%(curr_epoch+1, train_loss))

        print(num_epochs)

    def save(self, path=None):
        if path is None:
            path = osp.join(self.log_dir, "suc_pred_model")
        self.saver.save(self.sess, path)

    def test(self, test_dataset):
        test_feats, test_label = test_dataset
        test_summary = self.sess.run(self.sum, feed_dict={
                                        self.in_tensor: test_feats,
                                        self.target_tensor: test_label
                                        })
        return test_summary

