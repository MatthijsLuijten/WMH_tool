import numpy as np
import parameters
from sklearn.model_selection import train_test_split

def load_data():
	cases_2011 = load_cases(parameters.path_2011_cases)
	cases_2015 = load_cases(parameters.path_2015_cases)
	cases = cases_2011 + cases_2015
	train_cases, test_cases = train_test_split(cases, test_size=0.2, shuffle=True)
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