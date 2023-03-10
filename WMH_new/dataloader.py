import parameters
from sklearn.model_selection import train_test_split


def load_data():
    cases = load_cases(parameters.path_good_cases)
    return train_test_split(cases, test_size=0.2, shuffle=True)


def load_all_data():
    return load_cases(parameters.path_good_cases)


def load_cases(file_path):
    with open(file_path) as file:
        return [line.strip() for line in file if line.strip()]


def load_pm_data(file_path):
    return load_cases(file_path)
    # return train_test_split(load_cases(file_path), test_size=0.2, shuffle=True)