import tensorflow as tf


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
    rnd_name = 'ParamSelectorGrad' + 'ABC@a1b2c3'

    # Register Tensorflow Gradient
    tf.RegisterGradient(rnd_name)(grad)

    # Get current graph
    g = tf.get_default_graph()

    # Add gradient override map
    with g.gradient_override_map({"ParamSelector": rnd_name, "ParamSelectorStateless": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)



def my_identity_func(action, select_vec):
	return action


# @tf.RegisterGradient("MyIdentity")
def _custom_identity_grad(op, grad):
	action = op.inputs[0]
	select_vec = op.inputs[1]
	return tf.multiply(grad, select_vec), tf.zeros_like(grad)
                

if __name__ == "__main__":
	with tf.Session() as sess:
		c = tf.constant([[-2.0, 0.0], [0.5, 2.0]])
		s1 = py_func(my_identity_func, [c, -1., 1.], c.dtype, name="MyIdentity", grad=_custom_identity_grad)
		s2 = -tf.reduce_mean(s1*2)
		init = tf.global_variables_initializer()
		sess.run(init)
		print(c.eval(), s2.eval())

		## expected result
		# [[ 0.25  -0.25 ]
 		# [-0.375 -0.75 ]]
		print(tf.gradients(s2, c)[0].eval())
