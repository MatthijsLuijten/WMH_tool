import numpy as np
import parameters
from sklearn.model_selection import train_test_split

def load_data():
	cases_2011 = load_cases(parameters.path_2011_cases)
	cases_2015 = load_cases(parameters.path_2015_cases)
	cases = np.concatenate((cases_2011, cases_2015))
	np.random.shuffle(cases)
	train_cases, test_cases = cases[80:], cases[:80]
	return train_cases, test_cases


def load_cases(filePath):
	result = list()
	file = open(filePath)
	i = 0
	for line in file:
		if line[:-1]!="" and line!="":
			result.append(line[:-1])
		i = i+1
	return result