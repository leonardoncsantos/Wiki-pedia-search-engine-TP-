import pickle

# Load the dictionary file
with open('tokInfo.dict', 'rb') as f:  # Replace with your dictionary file
    data = pickle.load(f)

# Print the contents of the dictionary
print(data)
