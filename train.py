import pandas as pd 
import numpy as np 

def squared_error(attribute, label):
	split_value = np.mean(attribute)
	label_error_l, label_error_r = 0,0
	right , left = np.array([]), np.array([]) 
	for i in range(len(attribute)):
		if attribute[i] > split_value:
			right = np.append(right, label[i])
		else:
			left = np.append(left, label[i])
	right_value = sum(right)/len(right)
	left_value = sum(left)/len(left)

	left_error = np.mean((left - left_value)**2);
	right_error = np.mean((right - right_value)**2);

	error = left_error+right_error

	return error

def absolute_error(attribute, label):
	split_value = np.mean(attribute)
	label_error_l, label_error_r = 0,0
	right , left = np.array([]), np.array([]) 
	for i in range(len(attribute)):
		if attribute[i] > split_value:
			right = np.append(right, label[i])
		else:
			left = np.append(left, label[i])
	right_value = sum(right)/len(right)
	left_value = sum(left)/len(left)

	left_error = np.mean(abs(left - left_value));
	right_error = np.mean(abs(right - right_value));

	error = left_error+right_error

	return error

data = pd.read_csv('toy_dataset.csv')
label = data.iloc[:,-1]
attributes = data.iloc[:, 0:-1]

print(squared_error(attributes.iloc[:,0], label))
print(squared_error(attributes.iloc[:,1], label))
print(absolute_error(attributes.iloc[:,0], label))
print(absolute_error(attributes.iloc[:,1], label))
