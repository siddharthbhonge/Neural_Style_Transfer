import os
import sys

import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from ns_utils import *
import numpy as np
import tensorflow as tf

model = load_vgg_model("model/imagenet-vgg-verydeep-19.mat")
print(model)

content_image = scipy.misc.imread("images/louvre.jpg")
imshow(content_image)

def compute_content_cost(a_C, a_G):
	# Reshape a_C and a_G (≈2 lines)
	m, n_H, n_W, n_C = a_G.get_shape().as_list()
	a_C = tf.reshape(tf.transpose(a_C, perm = [0, 3, 1, 2]), [1, n_C, n_H * n_W])
	#print(a_C_unrolled.get_shape())
	a_G = tf.reshape(tf.transpose(a_G, perm = [0, 3, 1, 2]), [1, n_C, n_H * n_W])
	# compute the cost with tensorflow (≈1 line)
	J_content = 1.0 / (4.0 * n_C * n_H * n_W) * tf.reduce_sum(tf.square(tf.subtract(a_C, a_G))) 
	### END CODE HERE ###
	return J_content
	
	
	
style_image = scipy.misc.imread("images/monet_800600.jpg")
imshow(style_image)

def gram_matrix(A):
	GA = tf.matmul(A, tf.transpose(A))

	return GA	
	
def compute_layer_style_cost(a_S, a_G):
	m, n_H, n_W, n_C = a_G.get_shape().as_list()

	# Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
	a_S = tf.reshape(tf.transpose(a_S, perm = [0, 3, 1, 2]), [n_C, n_H * n_W])
	a_G = tf.reshape(tf.transpose(a_G, perm = [0, 3, 1, 2]), [n_C, n_H * n_W])

	# Computing gram_matrices for both images S and G (≈2 lines)
	GS = gram_matrix(a_S)
	GG = gram_matrix(a_G)

	# Computing the loss (≈1 line)
	J_style_layer = 1.0 / (4.0 * n_C**2 * n_H**2 * n_W**2) * tf.reduce_sum(tf.square(tf.subtract(GS, GG)))

	return J_style_layer
	
STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]
	
def compute_style_cost(model, STYLE_LAYERS):
	J_style = 0

	for layer_name, coeff in STYLE_LAYERS:

		# Select the output tensor of the currently selected layer
		out = model[layer_name]

		# Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
		a_S = sess.run(out)
		a_G = out

		# Compute style_cost for the current layer
		J_style_layer = compute_layer_style_cost(a_S, a_G)

		# Add coeff * J_style_layer of this layer to overall style cost
		J_style += coeff * J_style_layer

	return J_style
	
def total_cost(J_content, J_style, alpha = 10, beta = 40):
    
	### START CODE HERE ### (≈1 line)
	J = alpha * J_content + beta * J_style
	### END CODE HERE ###

	return J
	
# Reset the graph
tf.reset_default_graph()

# Start interactive session
sess = tf.InteractiveSession()


content_image = scipy.misc.imread("images/statue.jpg")
content_image = reshape_and_normalize_image(content_image)

style_image = scipy.misc.imread("images/picasso.jpg")
style_image = reshape_and_normalize_image(style_image)

generated_image = generate_noise_image(content_image)
imshow(generated_image[0])

model = load_vgg_model("model/imagenet-vgg-verydeep-19.mat")
sess.run(model['input'].assign(content_image))
out = model['conv4_2']
a_C = sess.run(out)
a_G = out
J_content = compute_content_cost(a_C, a_G)

# Assign the input of the model to be the "style" image 
sess.run(model['input'].assign(style_image))

# Compute the style cost
J_style = compute_style_cost(model, STYLE_LAYERS)
J = total_cost(J_content, J_style)
optimizer = tf.train.AdamOptimizer(2.0)

# define train_step (1 line)
train_step = optimizer.minimize(J)

def model_nn(sess, input_image, num_iterations = 200):
	sess.run(tf.global_variables_initializer()) 
	sess.run(model['input'].assign(input_image))

	for i in range(num_iterations):

		sess.run(train_step)
		generated_image = sess.run(model['input'])
		if i%20 == 0:
			Jt, Jc, Js = sess.run([J, J_content, J_style])
			print("Iteration " + str(i) + " :")
			print("total cost = " + str(Jt))
			print("content cost = " + str(Jc))
			print("style cost = " + str(Js))
			save_image("output/" + str(i) + ".png", generated_image)
	save_image('output/generated_image.jpg', generated_image)
    
	return generated_image
	
model_nn(sess, generated_image)


