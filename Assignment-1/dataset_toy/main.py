import numpy as np 
import pandas as pd 
from scipy.stats import mode
import time
import matplotlib.pyplot as plt
import sys

#Function to calculate squared loss on the split groups
def squared_loss(groups):
	num_examples = float(sum([len(group) for group in groups]))
	loss = 0
	for group in groups:
		size = float(len(group))
		if size == 0:
			continue
		score = 0
		value = np.mean(group[:,-1])
		score = np.sum((group[:,-1]-value)**2)
		loss+=score*(size/num_examples)
	return loss

#Function to calculate absolute loss on the split groups
def absolute_loss(groups):
	num_examples = float(sum([len(group) for group in groups]))
	loss = 0
	for group in groups:
		size = float(len(group))
		if size == 0:
			continue
		score = 0
		value = np.mean(group[:,-1])
		score = np.sum(abs(group[:,-1]-value))
		loss+=score*(size/num_examples)
	return loss

#This function compares the groups(examples) with the split_value at the given index for a particular attribute and divdes the dataset into left and right subgroups.
def test_split(index, split_value, data, groups):
	left, right = groups
	for row in right:
		if row[index] < split_value and right.shape[0]>1:
			left = np.vstack((left, row))
			right = np.delete(right, (0), axis=0)
		elif right.shape[0]>=2 and left.shape[0] == 0:
			left = np.vstack((left, row))
			right = np.delete(right, (0), axis=0)			
		else:
			break

	return left, right

#This function performs the split at each attribute value of each attribute and calculates the error for that split. 
#To optimise the code, the dataset is sorted based on the attribute whose attribute values are being used for performing the split.
def get_split(data, loss_choice):
	best_index, best_value, best_score, best_groups = 999,999,999999999999999,None
	for index in range(data.shape[1]-1):
		dataset = data[data[:,index].argsort()] #Sort the dataset based on the attribute at index
		right,left = np.array([]).reshape(0, data.shape[1]), np.array([]).reshape(0, data.shape[1])
		right = np.vstack((dataset,right))
		groups = left,right
		for row in data:
			groups = test_split(index, row[index], dataset, groups)

			if loss_choice == '--absolute':
				loss = absolute_loss(groups)
			elif loss_choice == '--square':
				loss = squared_loss(groups)

			if loss < best_score:
				best_score = loss
				best_index = index
				best_value = row[index]
				best_groups = groups

	output={}
	output["index"] = best_index
	output["groups"] = best_groups
	output["value"] = best_value

	return output

#Building the tree structure

#Define the terminal node for stopping the tree construction at a particular depth or if min_leaf_examples are reached
#For stopping the tree, simple return the mean of the remaining examples as the output
def terminal_node(group):
	outcomes = np.mean(group[:,-1])
	return outcomes

#Splitting function that recursively splits the tree at each node based on the get_split function
def split(node, max_depth, min_size, depth, loss_choice):
	if node['groups'] == None:
		return
	else:
		left, right = node['groups'] # Split the groups at the node
		del(node['groups'])

	if depth >= max_depth: # Check for early stopping
		node['left'], node['right'] = terminal_node(left), terminal_node(right)
		return

	# Check if the number of examples at that group are less then the minimum examples we want at the leaf
	if len(left) <= min_size: 
		node['left'] = terminal_node(left)
	else:
		node['left'] = get_split(left, loss_choice)
		split(node['left'], max_depth, min_size, depth+1, loss_choice)

	# Check if the number of examples at that group are less then the minimum examples we want at the leaf
	if len(right) <= min_size:
		node['right'] = terminal_node(right)
	else:
		node['right'] = get_split(right, loss_choice)
		split(node['right'], max_depth, min_size, depth+1, loss_choice)

#Call the split function and build the tree
def build_tree(train, max_depth, min_size, loss_choice):
	root = get_split(train, loss_choice)
	split(root, max_depth, min_size, 1, loss_choice)
	return root

#REcursively traverse the tree structure to make the prediction for a test example
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

# Accuracy is taken as least square of the error from training data output
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		correct += (actual[i]-predicted[i])**2
	return correct / float(len(actual))

depth = 2 # Max number of nodes
min_leaf_size = 1
loss_choice = '--absolute'

data = np.loadtxt('toy_dataset.csv', delimiter=',', skiprows=1)

start = time.time()
tree = build_tree(data,depth,min_leaf_size, loss_choice)
end = time.time()

prediction = []
for row in data:
	prediction.append(predict(tree, row))
error = np.sum( (prediction-data[:,-1])**2 )/(len(prediction))

train_time = end-start

#The commented code below was used to imporve kaggle score by randomly shuffling the dataset
#The same code was modified for plotting the graphs in the report

# best_error = 999
# error_vals = []
# node_vals = []

# for i in range(100):
# 	print(i)
# 	#np.random.shuffle(data)
# 	data_split = int(0.9*data.shape[0])
# 	train_data = data[0:data_split, :]
# 	test_data = data[data_split:, :]
# 	tree = build_tree(data,20,min_leaf_size)
# 	best_tree = tree
# 	prediction = []
# 	for row in data:
# 		prediction.append(predict(tree, row))
# 	prediction = np.array(prediction)
# 	error = np.sum( (prediction-data[:,-1])**2 )/(len(prediction))
# 	error_vals.append(error)
# 	node_vals.append(i)

# 	print(error)
# 	if error < best_error:
# 		best_error = error
# 		best_tree = tree
# 		best_index = i


# plt.plot(node_vals,error_vals)
# plt.title("Variation of Absolute training loss with number of nodes")
# plt.xlabel("Number of Nodes")
# plt.ylabel("Absolute Training Loss")
# plt.show()

# test_data = np.loadtxt(test_path, delimiter=',', skiprows=1)
# prediction = []
# start = time.time()
for row in data:
	prediction.append(predict(tree, row))
# end = time.time()
# test_time = end - start

print("Training Time:", train_time)
print("Training Error", error)

Id = np.arange(1,len(prediction)+1)
output = {'Id': Id , 'output': prediction}

df = pd.DataFrame(data=output)
df.to_csv("output.csv", sep = ',', index=False)




