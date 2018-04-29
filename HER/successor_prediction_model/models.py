import tensorflow as tf
import os.path as osp

from HER.ddpg.models import Model
from HER.deepq.models import _mlp

class regressor:
    def __init__(self, name, in_shape, out_shape, sess=None, log_dir="/tmp"):
        self.in_tensor = tf.placeholder(tf.float32, shape=(None,) + (in_shape,), name='state_goal')
        self.out_tensor = _mlp(inpt=self.in_tensor, hiddens=[64,64],scope="suc_pred_model", num_actions= out_shape)
        self.target_tensor = tf.placeholder(tf.float32, shape=(None,) + (out_shape,), name='final_state')
        
        self.sess = sess
        self.log_dir = log_dir
        # loss function 
        self.loss = tf.reduce_sum(tf.reduce_sum(tf.square(self.target_tensor - self.out_tensor), axis=1))

        # summary
        tf.summary.scalar("sqrt loss", tf.sqrt(self.loss))
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
            train_feats, train_label = sess.run(train_dataset)
            feed_dict[self.in_tensor] = train_feats
            feed_dict[self.target_tensor] = train_label

            train_summary, _ = sess.run([self.sum, self.optim], feed_dict)

            if log:
                writer_t.add_summary(train_summary,curr_epoch)

            # test
            if (curr_epoch+1)%test_freq:
                test_summary = self.test(test_dataset)
                if log:
                    writer_v.add_summary(test_summary,curr_epoch)


            if (curr_epoch+1)%save_freq:
                self.save()

            print("epoch:%d, loss:%.5f"%(curr_epoch+1, loss))

    def save(path=None):
        if path is None:
            path = osp.join(self.log_dir, "suc_pred_model")
        self.saver.save(self.sess, path)

    def test(self, test_dataset):
        test_feats, test_label = self.sess.run(test_dataset)
        test_summary = self.sess.run(self.sum, feed_dict={
                                    self.in_tensor: test_feats,
                                    self.target_tensor: test_label
                                    })

        return test_summary

