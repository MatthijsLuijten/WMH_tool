import parameters

def load_data():
	train_cases = load_cases(parameters.path_trainingcases)
	# valid_cases = load_cases(parameters.path_validationcases)
	test_cases = load_cases(parameters.path_testcases)
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