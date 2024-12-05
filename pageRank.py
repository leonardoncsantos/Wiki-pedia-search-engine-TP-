from itertools import chain
import numpy
import pickle

CONVERGENCE_LIMIT = 0.0000001

# Load the link information
with open("links.dict", 'rb') as f:
    links = pickle.load(f)

# List of page titles
allPages = list(set().union(chain(*links.values()), links.keys()))
# For simplicity of coding, we give an index to each of the pages.
linksIdx = [[allPages.index(target) for target in links.get(source, list())] for source in allPages]

# Remove redundant links (i.e. same link in the document)
for l in links:
    links[l] = list(set(links[l]))

# One click step in the "random surfer model"
# origin = probability distribution of the presence of the surfer (list of numbers) on each of the page
def surfStep(origin, links):
    dest = [0.0] * len(origin)
    for idx, proba in enumerate(origin):
        if len(links[idx]) > 0:
            w = 1.0 / len(links[idx])  # uniform distribution to each outgoing link
        else:
            w = 0.0
            
        for link in links[idx]:
            dest[link] += proba * w
    return dest  # proba distribution after a click

# Ensure that the list of allPages is not empty to avoid division by zero
if len(allPages) > 0:
    pageRanks = [1.0 / len(allPages)] * len(allPages)  # will contain the page ranks
    sourceVector = [1.0 / len(allPages)] * len(allPages)  # Default source vector
else:
    # If there are no pages, set the pageRanks and sourceVector to empty lists
    pageRanks = []
    sourceVector = []

# Initialize delta to a large value for the convergence check
delta = float("inf")

# Or use a personalized source vector:
# Example: if you want to personalize, say, starting with a higher probability for a specific set of pages
# Example for personalization, you can modify sourceVector like this:
# personalizedPages = ['Page1', 'Page2', 'Page3']
# personalizedSourceVector = [0.1 if page in personalizedPages else 0.0 for page in allPages]
# sourceVector = personalizedSourceVector

# Iterating until convergence
while delta > CONVERGENCE_LIMIT:
    print("Convergence delta:", delta, sum(pageRanks), len(pageRanks))
    
    # Apply the random surfer model: calculate next page ranks
    pageRanksNew = surfStep(pageRanks, linksIdx)  # compute the next step in the algorithm
    
    # Calculate the "jump probability" (the effect of personalization or damping factor)
    jumpProba = sum(pageRanks) - sum(pageRanksNew)  # what effect is detected here?
    if jumpProba < 0:  # Technical artifact due to numerical errors
        jumpProba = 0
    
    # Correct for this effect: Apply the jump probability correction
    pageRanksNew = [pageRank + jump for pageRank, jump in zip(pageRanksNew, (p * jumpProba for p in sourceVector))]

    # Compute the delta (the difference between the old and new page ranks)
    delta = numpy.linalg.norm(numpy.array(pageRanksNew) - numpy.array(pageRanks), 1)  # L1 norm

    # Update page ranks
    pageRanks = pageRanksNew

# Get the top 20 pages with highest ranks
bestPages = [allPages[i] for i in numpy.argsort(pageRanks)[-20:]]
bestPageRanks = [pageRanks[i] for i in numpy.argsort(pageRanks)[-20:]]

# Name the entries of the pageRank vector
pageRankDict = dict()
for idx, pageName in enumerate(allPages):
    pageRankDict[pageName] = pageRanks[idx]

# Rank of some pages (for example, 'Page1', 'Page2', etc.)
# Add any specific pages you're interested in ranking
somePages = ['Page1', 'Page2', 'Page3']  # Example pages to rank
for page in somePages:
    if page in pageRankDict:
        print(f"Page rank of {page}: {pageRankDict[page]}")
    else:
        print(f"Page {page} not found in page ranks")

# Save the ranks as pickle object
with open("pageRank.dict", 'wb') as fileout:
    pickle.dump(pageRankDict, fileout, protocol=pickle.HIGHEST_PROTOCOL)
