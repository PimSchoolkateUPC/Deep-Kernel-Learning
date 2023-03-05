from torch import Tensor
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import regex as re
import numpy as np

def load_dat_file(path, sep=",", y_pos=-1, x_transform=None, y_transform=None):
    """
    Reads in a data file located at the specified `path` and returns the data as a tuple of two lists: `x_data` and `y_data`.
    
    Args:
    - path (str): The path to the data file.
    - sep (str, optional): The separator used in the data file. Defaults to `,`.
    - y_pos (int, optional): The position of the label (y-value) in each row of data. Set to `None` if the data does not include labels. Defaults to -1.
    - x_transform (function, optional): A function to transform each x-value before appending it to `x_data`. If `None`, no transformation is performed. Defaults to `None`.
    - y_transform (function, optional): A function to transform each y-value before appending it to `y_data`. If `None`, no transformation is performed. Defaults to `None`.

    Returns:
    - tuple: A tuple of two lists: `x_data` and `y_data`.

    """
    # Initialize empty lists to hold the x and y data
    x_data = []
    y_data = []

    # Define default x and y transformation functions
    if x_transform is None:
        def x_transform(x):
            return x

    if y_transform is None:
        def y_transform(y):
            return y

    # Open the data file and read in each line
    with open(path, "r") as file:
        for line in file:
            # Skip any line that starts with "@"
            if "@" in line:
                continue
            # Remove any trailing newline characters
            line = line.replace('\n', '')
            # Split the line into elements using the specified separator
            elements = str2list(line, sep)
            # If the y_pos parameter is specified, append the y-value to y_data
            if y_pos is not None:
                y_data.append(y_transform(elements.pop(y_pos)))
            # Append the transformed x-values to x_data
            x_data.append(x_transform(elements))
    # Return the x and y data as a tuple
    return x_data, y_data


def str2list(string, sep=","):
    return string.split(sep)

def australian_transform_x(x):
    isnumber = re.compile(r'\d+(?:,\d*)?')
    return [float(ele) if isnumber.search(ele) else ele for ele in x]

def australian_transform_y(y):
    return int(y)

def balance_transform_x(x):
    return [float(ele) for ele in x]

def balance_transform_y(y):
    if y == " B":
        return 0
    elif y == " R":
        return 1
    elif y == " L":
        return 2

def magic_transform_x(x):
    return [float(ele) for ele in x]

def magic_transform_y(y):
    if y == "g":
        return 0
    elif y == "h":
        return 1

def monk_transform_x(x):
    return [float(ele) for ele in x]

def monk_transform_y(y):
    return int(y)

def sonar_transform_x(x):
    return [float(ele) for ele in x]

def sonar_transform_y(y):
    if y == " R":
        return 0
    elif y == " M":
        return 1

def spambase_transform_x(x):
    return [float(ele) for ele in x]

def spambase_transform_y(y):
    return int(y)

def titanic_transform_x(x):
    return [float(ele) for ele in x]

def titanic_transform_y(y):
    if y=="-1.0":
        return 0
    elif y=="1.0":
        return 1

def CNAE_transform_x(x):
    return [float(ele) for ele in x]

def CNAE_transform_y(y):
    return int(y) - 1

def covertype_transform_x(x):
    return [float(ele) for ele in x]

def covertype_transform_y(y):
    return int(y)

def madelon_transform_x(x):
    return [float(ele) for ele in x[:-1]]

def madelon_transform_y(y):
    assert len(y) == 1
    if y[0] == "-1":
        return 0
    elif y[0] == "1":
        return 1

def unique_in_list(l):
    u = []
    for item in l:
        if item not in u:
            u.append(item)
    return u


class BaseDataset(Dataset):
    def __init__(self, x: list, y: list):
        self.x = Tensor(x)
        self.y = Tensor(y).long()
                
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class MLRepositoryDatasetGenerator():

    """FOR FUTURE IMPROVEMNT: BUILD A CLASS/FUNC THAT AUTOMATICALLY CREATES TEST/TRAIN SET ON DISK INSTEAD OF IN MEMORY TO IMPROVE EFFICIENCY."""

    def __init__(self, name: str, data_dir: str, test_size=0.2, seed=42, verbose=True) -> None:
        self._load_dataset(name, data_dir)
        if verbose:
            print(f"The explanatory data contains {len(self.x[0])} features")
            print(f"There are {len(unique_in_list(self.y))} target classes")
        self._instantiate_train_test(test_size, seed)

    def __call__(self, mode: str):
        if mode == "train":
            return BaseDataset(self.train_x, self.train_y)
        elif mode == "test":
            return BaseDataset(self.test_x, self.test_y)
        else:
            raise ValueError(f"Either 'train' or 'test' can be passed as valid modes. Got '{mode}' instead")
    
    def get_data_as_arrays(self):
        return np.array(self.train_x), np.array(self.train_y), np.array(self.test_x), np.array(self.test_y)

    def _instantiate_train_test(self, test_size, seed):
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.x, self.y, test_size=test_size, random_state=seed)
        del self.x, self.y

    def _load_dataset(self, name: str, data_dir: str):
        if name == "australian":
            self.x, self.y = load_dat_file(data_dir + "/spambase/spambase.dat", x_transform=spambase_transform_x, y_transform=spambase_transform_y)
        elif name == "balance":
            self.x, self.y = load_dat_file(data_dir + "/balance/balance.dat", x_transform=balance_transform_x, y_transform=balance_transform_y)
        elif name == "magic":
            self.x, self.y = load_dat_file(data_dir + "/magic/magic.dat", x_transform=magic_transform_x, y_transform=magic_transform_y)
        elif name == "monk-2":
            self.x, self.y = load_dat_file(data_dir + "/monk-2/monk-2.dat", x_transform=monk_transform_x, y_transform=monk_transform_y)
        elif name == "sonar":
            self.x, self.y = load_dat_file(data_dir + "/sonar/sonar.dat", x_transform=sonar_transform_x, y_transform=sonar_transform_y)
        elif name == "titanic":
            self.x, self.y = load_dat_file(data_dir + "/titanic/titanic.dat", x_transform=titanic_transform_x, y_transform=titanic_transform_y)
        elif name == "spambase":
            self.x, self.y = load_dat_file(data_dir + "/spambase/spambase.dat", x_transform=spambase_transform_x, y_transform=spambase_transform_y)
        elif name == "CNAE-9":
            self.x, self.y = load_dat_file(data_dir + "/CNAE-9/CNAE-9.data", y_pos=0, x_transform=CNAE_transform_x, y_transform=CNAE_transform_y)
        elif name == "covertype":
            self.x, self.y = load_dat_file(data_dir + "/covertype/covtype.data", x_transform=covertype_transform_x, y_transform=covertype_transform_y)
        elif name == "madelon":
            self.x, _ = load_dat_file(data_dir + "/madelon/madelon_train.data", y_pos=None, x_transform=madelon_transform_x, sep=" ")
            self.y, _ = load_dat_file(data_dir + "/madelon/madelon_train.labels", y_pos=None, x_transform=madelon_transform_y)
        else:
            raise ValueError(f"The dataset {name} has not been implemented yet or does not exist")  