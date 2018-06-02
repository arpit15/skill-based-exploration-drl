import tensorflow as tf
import numpy as np 

# Python Custom Gradient Function
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    """
    PyFunc defined as given by Tensorflow
    :param func: Custom Function
    :param inp: Function Inputs
    :param Tout: Ouput Type of out Custom Function
    :param stateful: Calculate Gradients when stateful is True
    :param name: Name of the PyFunction
    :param grad: Custom Gradient Function
    :return:
    """
    # Generate Random Gradient name in order to avoid conflicts with inbuilt names
    rnd_name = 'PyFuncGrad' + 'ABC@a1b2c3'

    # Register Tensorflow Gradient
    tf.RegisterGradient(rnd_name)(grad)

    # Get current graph
    g = tf.get_default_graph()

    # Add gradient override map
    with g.gradient_override_map({"PyFunc": rnd_name, "PyFuncStateless": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)



def my_identity_func(action, select_vec):
    # python func
    return np.multiply(action, select_vec)


def _custom_identity_grad(op, grad):
    action = op.inputs[0]
    select_vec = op.inputs[1]
    return [tf.multiply(grad, select_vec), tf.zeros_like(grad)]
                

if __name__ == "__main__":
    with tf.Session() as sess:
        # c = tf.constant([[-2.0, 0.0], [0.5, 2.0]])
        # s = tf.constant([[1.0, 0.0], [0.0, 1.0]])

        c = tf.constant([[-2.0, 0.5]])
        s = tf.constant([[1.0, 0.0]])
        s3 = -tf.reduce_mean(tf.multiply(c,s))

        s1 = py_func(my_identity_func, [c, s], c.dtype, name="MyIdentity", grad=_custom_identity_grad)
        s2 = -tf.reduce_mean(s1)
        
        dels1 = tf.gradients(s2, s1)[0]
        dels = tf.gradients(s2, s)[0]
        delc = tf.gradients(s2, c)[0]
        dels3 = tf.gradients(s3,c)[0]

        init = tf.global_variables_initializer()
        sess.run(init)

        # expected output: [[-2.   0.5]] 1.0
        print(c.eval(), s2.eval())

        print(dels1, dels, delc)

        # expected output : [[-0.5 -0.5]]
        print(dels1.eval())
        
        # expected output : [[0., 0.]]
        print(dels.eval())
        
        # expected output : [[-0.5 -0.]]
        print(delc.eval())
        
        # expected output : [[-0.5 -0.]]
        print(dels3.eval())
