from sklearn.model_selection import train_test_split

class DataSplit:
    def __init__(self, data):
        self.data = data

    def split_data(self, test_size=0.2, random_state=42):
        train, test = train_test_split(self.data, test_size=test_size, random_state=random_state)
        return train, test