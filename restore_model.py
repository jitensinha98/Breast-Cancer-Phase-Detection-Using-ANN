'''
The saved model is restored to perform some tests and classification
'''
# Importing required modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf

# Model save path
model_path="saved_model/model.ckpt"

def prep_dataset(dataset):
	
	dataset=dataset.drop(dataset.columns[dataset.columns.str.contains('Unnamed',case = False)],axis = 1)
	dataset = dataset.drop(dataset.columns[[0]], axis=1)
 
	x=dataset[dataset.columns[1:(dataset.shape[1]-1)]].values
	y=dataset[dataset.columns[0]]

	return (x,y)

def model_neural_network(data):
	hidden_layer_1={'weights':tf.Variable(tf.random_normal([n_cols,hl1_nodes])),'biases':tf.Variable(tf.random_normal([hl1_nodes]))}
	hidden_layer_2={'weights':tf.Variable(tf.random_normal([hl1_nodes,hl2_nodes])),'biases':tf.Variable(tf.random_normal([hl2_nodes]))}
	hidden_layer_3={'weights':tf.Variable(tf.random_normal([hl2_nodes,hl3_nodes])),'biases':tf.Variable(tf.random_normal([hl3_nodes]))}
	hidden_layer_4={'weights':tf.Variable(tf.random_normal([hl3_nodes,hl4_nodes])),'biases':tf.Variable(tf.random_normal([hl4_nodes]))}
	hidden_layer_5={'weights':tf.Variable(tf.random_normal([hl4_nodes,hl5_nodes])),'biases':tf.Variable(tf.random_normal([hl5_nodes]))}
	output_layer={'weights':tf.Variable(tf.random_normal([hl5_nodes,n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}
	

	l1=tf.add(tf.matmul(data,hidden_layer_1['weights']),hidden_layer_1['biases'])
	tf.nn.relu(l1)

	l2=tf.add(tf.matmul(l1,hidden_layer_2['weights']),hidden_layer_2['biases'])
	tf.nn.relu(l2)

	l3=tf.add(tf.matmul(l2,hidden_layer_3['weights']),hidden_layer_3['biases'])
	tf.nn.relu(l3)

	l4=tf.add(tf.matmul(l3,hidden_layer_4['weights']),hidden_layer_4['biases'])
	tf.nn.relu(l4)

	l5=tf.add(tf.matmul(l4,hidden_layer_5['weights']),hidden_layer_5['biases'])
	tf.nn.relu(l2)

	output=tf.add(tf.matmul(l5,output_layer['weights']),output_layer['biases'])

	return output

def train_neural_network(X):	
	prediction = model_neural_network(X)	
	# Creating a saver object
	saver=tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		# Restoring saved model		
		saver.restore(sess, model_path)
		prediction_ = tf.argmax(prediction,1)
		# Testing the similarity between predicted and original class
		print("TESTING ORIGINAL DATASET CLASSES WITH PREDICTED CLASSES UPTO 200 ROWS")
		print("--------------------------------------------------------")
		for i in range(1,200):
			# Testing the similarity between predicted and original class
			prediction_run=sess.run(prediction_,feed_dict={X:original_x[i].reshape(1,n_cols)})
			print("Original Class = ",original_y[i]," Predicted Class = ",prediction_run)


dataset=pd.read_csv("breast-cancer.csv")

original_x,original_y=prep_dataset(dataset)

n_cols=original_x.shape[1]

hl1_nodes=100
hl2_nodes=100
hl3_nodes=100
hl4_nodes=100
hl5_nodes=100

n_classes=2
hm_epochs=200

X=tf.placeholder('float',[None,n_cols])
Y=tf.placeholder('float')


train_neural_network(X)

