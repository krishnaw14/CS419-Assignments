import pandas as pd 
import numpy as np 

def squared_error(attribute, label):
	split_value = np.mean(attribute)
	#print("split_value", split_value)
	right , left = np.array([]), np.array([]) 
	for i in range(len(attribute)):
		if attribute[i] > split_value:
			right = np.append(right, label[i])
		else:
			left = np.append(left, label[i])

	if len(right)!=0:
		right_value = int(sum(right)/len(right))
		right_error = np.mean((right - right_value)**2)
	else:
		right_error = 0

	if len(left)!=0:
		left_value = int(sum(left)/len(left))
		left_error = np.mean((left - left_value)**2)
	else:
		left_error = 0

	error = left_error+right_error
	error = error/(len(left)+len(right))

	return error, split_value

def absolute_error(attribute, label):
	split_value = np.mean(attribute)
	print("split_value", split_value)
	label_error_l, label_error_r = 0,0
	right , left = np.array([]), np.array([]) 
	for i in range(len(attribute)):
		if attribute[i] > split_value:
			right = np.append(right, label[i])
		else:
			left = np.append(left, label[i])
	right_value = sum(right)/len(right)
	print("right_value", right_value)
	left_value = sum(left)/len(left)
	print("left_value", left_value)

	left_error = np.mean(abs(left - left_value));
	right_error = np.mean(abs(right - right_value));

	error = left_error+right_error

	return error

def split(attribute_id, split_value, data):
	print("Split function")
	Xi_left = np.array([]).reshape(0, data.shape[-1])
	Xi_right = np.array([]).reshape(0, data.shape[-1])

	for i in range(data.shape[0]):

		if data[i,attribute_id] <= split_value:
			Xi_left = np.vstack((Xi_left, data[i,:]))
		else:
			Xi_right = np.vstack((Xi_right, data[i,:]))

	return Xi_left, Xi_right

def best_split(data):
	best_attribute, best_split_value, best_error, best_groups = 1000, 1000, 1000, None
	print("best_split function")
	print(data)
	print(type(data))
	print(data.shape)
	label = data[:,-1]
	for attribute_id in range(data.shape[1]-1):
		attribute = data[:,attribute_id]
		error, split_value = squared_error(attribute, label)
		if error < best_error:
			best_groups = split(attribute_id, split_value, data)
			best_error = error
			best_split_value = split_value
			best_attribute = attribute_id
	output = {}
	output["attribute_id"] = attribute_id
	output["groups"] = best_groups
	output["split_value"] = best_split_value

	return output

def terminal_node(group):
	mean_value = np.mean(group[:,-1])
	return mean_value

def split_branch(node, max_depth, min_num_sample, depth):
	left_node, right_node = node['groups']
	del(node['groups'])

	if depth >= max_depth:
		node['left'] = terminal_node(left_node)
		node['right'] = terminal_node(right_node)
		return
	
	if len(left_node) <= min_num_sample:
		node['left'] = terminal_node(left_node)
	else:
		node['left'] = best_split(left_node)
		split_branch(node['left'], max_depth, min_num_sample, depth+1)

	if len(right_node) <= min_num_sample:
		node['right'] = terminal_node(right_node)
	else:
		node['right'] = best_split(right_node)
		split_branch(node['right'], max_depth, min_num_sample, depth+1)	


def build_tree(data, max_depth, min_num_sample):
	root = best_split(data)
	split_branch(root, max_depth, min_num_sample, 1)
	return root

data = np.loadtxt('Data1/train.csv', delimiter=',', skiprows=1)
label = data[:,-1]
attributes = data[:, 0:-1]

tree = build_tree(data, 10, 20)

def predict_sample(node,sample):
    print(node)
    if sample[node["attribute_id"]] < node["split_value"]:
        if isinstance(node['left'],dict):
            return predict_sample(node['left'],sample)
        else:
            return node['left']
    else:
        if isinstance(node['right'],dict):
            return predict_sample(node['right'],sample)
        else:
            return node['right']

def predict(X):
    y_pred = np.array([])
    for i in X:
        y_pred = np.append(y_pred,predict_sample(tree,i))
    return y_pred

test_data = np.loadtxt('Data1/test.csv', delimiter=',', skiprows=1)
#test_label = test_data[:,-1]
# y_pred = predict(test_data)

# loss = 0.0
# vals = []
# for i in range(test_data.shape[0]):
# 		vals.append(y_pred[i])
# 		#loss+=(y_pred[i] - label[i])**2

# #loss/=test_label.shape[0]
# print("Loss = ", loss)
# values = np.array(vals)
# df = pd.DataFrame(values)
# df.to_csv("output.csv", sep = ',')

prediction = []
for row in test_data:
	prediction.append(predict(row))

Id = np.arange(1,len(prediction)+1)
output = {'Id': Id , 'output': prediction}

#print(accuracy_metric(data[:,-1], prediction))
df = pd.DataFrame(data=output)
df.to_csv("out.csv", sep = ',')


