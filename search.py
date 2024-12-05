from itertools import chain
from collections import defaultdict
import numpy
import pickle
import sys
import copy

# Load the necessary files
with open("tfidf.dict",'rb') as f:
    tfidf = pickle.load(f)
    
with open("tokInfo.dict",'rb') as f:
    tokInfo = pickle.load(f)

with open("pageRank.dict",'rb') as f:
    pageRankDict = pickle.load(f)


print("Normalizing tf idf...", end="")
tfidfNorm = copy.deepcopy(tfidf)

# Normalizing tf-idf: We need to calculate the norm (magnitude) of each document's tf-idf vector and divide each term's value by that norm
for doc, terms in tfidf.items():
    norm = numpy.sqrt(sum(value ** 2 for value in terms.values()))  # Calculate the Euclidean norm of the tf-idf vector
    if norm > 0:
        for term in terms:
            tfidfNorm[doc][term] /= norm  # Normalize each term's tf-idf value by the norm
print("done.")


# Returns the topN documents by token relevance (vector model)
def getBestResults(queryStr, topN, tfidfMatrix):
    query = queryStr.split(" ")
    res = defaultdict(float)
    
    # Compute the relevance for each document based on the query and the TF-IDF matrix
    for term in query:
        if term in tokInfo:  # Only consider terms that exist in the token info
            for doc in tfidfMatrix:
                if term in tfidfMatrix[doc]:
                    res[doc] += tfidfMatrix[doc][term] * tokInfo[term]  # Dot product of query and document tf-idf vectors
    
    # Sort documents by relevance (highest first)
    sortedResults = sorted(res.items(), key=lambda item: item[1], reverse=True)
    
    # Return the topN results
    return [doc for doc, _ in sortedResults[:topN]]


# Sorts a list of results according to their pageRank
def rankResults(results):
    # Sort the results by their pageRank in descending order
    return sorted(results, key=lambda doc: pageRankDict.get(doc, 0), reverse=True)


def printResults(rankedResults):
    for idx, page in enumerate(rankedResults):
        print(f"{idx + 1}. {page}")


# Get the search query from command line arguments
if len(sys.argv) > 1:
    query = sys.argv[1]  # Capture the first argument as the search query
else:
    print("Please provide a search query.")
    sys.exit(1)  # Exit if no query is provided

top = 15  # Number of results to show

print(f"Results for '{query}'\n===========")
results = getBestResults(query, top, tfidf)
printResults(results)

print("\n\nResults after normalization for ", query, "\n===========")
results = getBestResults(query, top, tfidfNorm)
printResults(results)

print("\n\nResults after ranking for ", query, "\n===========")
# First, get the best results, then rank them by their page rank
rankedResults = rankResults(results)
printResults(rankedResults)
