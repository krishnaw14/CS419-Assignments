import numpy as np 
import pandas as pd 
from scipy.stats import mode

def squared_loss(groups, classes):
	n_instances = float(sum([len(group) for group in groups]))
	loss = 0
	for group in groups:
		size = float(len(group))
		if size == 0:
			continue
		score = 0
		#value = mode(group[:,-1])[0][0]
		value = np.mean(group[:,-1])
		score = np.sum(abs(group[:,-1]-value))
		loss+=score*(size/n_instances)
	return loss

def absolute_loss(groups, classes):
	n_instances = float(sum([len(group) for group in groups]))
	loss = 0
	for group in groups:
		size = float(len(group))
		if size == 0:
			continue
		score = 0
		#value = mode(group[:,-1])[0][0]
		value = np.mean(group[:,-1])
		score = np.sum(abs(group[:,-1]-value))
		loss+=score*(size/n_instances)
	return loss

def test_split(index, split_value, data):
	left, right = np.array([]).reshape(0, data.shape[-1]), np.array([]).reshape(0, data.shape[-1])
	for row in data:
		if row[index] < split_value:
			left = np.vstack((left, row))
		else:
			right = np.vstack((right, row))

	return left, right

def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(dataset.shape[0]/n_fold)
	for i in range(n_folds):
		fold = list()
		while len(fold)<fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	dataset_split = np.array(dataset_split)
	return dataset_split


def get_split(data):
	class_values = list(set(row[-1] for row in data))
	b_index, b_value, b_score, b_groups = 999,999,999,None

	for index in range(data.shape[1]-1):
		for row in data:
			groups = test_split(index, row[index], data)
			loss = squared_loss(groups, class_values)
			if loss < b_score:
				b_score = loss
				b_index = index
				b_value = row[index]
				b_groups = groups

	output={}
	output["index"] = b_index
	output["groups"] = b_groups
	output["value"] = b_value

	return output

#Building the tree structure
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	#outcomes = mode(group[:,-1])[0][0]
	return max(set(outcomes), key = outcomes.count)
	#return outcomes

def split(node, max_depth, min_size, depth):
	# try:
	left, right = node['groups']
	del(node['groups'])
	# except TypeError:
	# 	print(node['groups'])
	# 	print(type(node['groups']))
	# 	return

	# if not left or not right:
	# 	node['left'] = node['right'] = to_terminal(left + right)
	# 	return

	if not isinstance(left,np.ndarray) or not isinstance(right,np.ndarray): # if the left_node and right_node are indeed numpy arrays
		node['left'] = node['right'] = to_terminal(left + right)
		return

	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return

	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)

	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)

def build_tree(train, max_depth, min_size):
	root = get_split(train)
	split(root, max_depth, min_size, 1)
	return root

def print_tree(node, depth=0):
	if isinstance(node, dict):
		print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
		print_tree(node['left'], depth+1)
		print_tree(node['right'], depth+1)
	else:
		print('%s[%s]' % ((depth*' ', node)))

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

def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		correct += (actual[i]-predicted[i])**2
	return correct / float(len(actual))

# for row in dataset:
# 	prediction = predict(stump, row)
# 	print(' Got=%d' % ( prediction))

data = np.loadtxt('Data2/train.csv', delimiter=',', skiprows=1)
np.random.shuffle(data)

n_fold = 5
max_depth = 20
min_size = 10

tree = build_tree(data,30,3)
print("Tree", tree, type(tree) )
print_tree(tree)

test_data = np.loadtxt('Data2/test.csv', delimiter=',', skiprows=1)
prediction = []
for row in test_data:
	prediction.append(predict(tree, row))

Id = np.arange(1,len(prediction)+1)
output = {'Id': Id , 'output': prediction}

#print(accuracy_metric(data[:,-1], prediction))
df = pd.DataFrame(data=output)
df.to_csv("out.csv", sep = ',', index=False)




