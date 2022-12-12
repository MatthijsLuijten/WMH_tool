import parameters
from sklearn.model_selection import train_test_split

def load_data():
	cases = load_cases(parameters.path_good_cases)
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

def load_pm_data(path):
	cases = load_cases(path)
	return cases