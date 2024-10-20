import gensim.downloader as api

# List all available datasets
print(list(api.info()['models'].keys()))  # This lists all available datasets

# Load the text8 dataset
dataset = api.load('text8')

# Convert the dataset to a list
dataset_list = list(dataset)

# Now you can access the first 10 elements
print(dataset_list[:10])