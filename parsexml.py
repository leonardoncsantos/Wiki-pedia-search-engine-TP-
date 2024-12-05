import sys
import xml.etree.ElementTree
from collections import defaultdict
import numpy
import re
import pickle
import glob
from itertools import chain

xmlFiles = list(chain(*[ glob.glob(globName)  for globName in sys.argv[1:] ]))
print("Files as input:", xmlFiles)

docs = dict()

##############################
print("Parsing XML...")
##############################
for xmlFile in xmlFiles:
    pages = xml.etree.ElementTree.parse(xmlFile).getroot()

    for page in pages.findall("{http://www.mediawiki.org/xml/export-0.10/}page"):
        titles = page.findall("{http://www.mediawiki.org/xml/export-0.10/}title")
        revisions = page.findall("{http://www.mediawiki.org/xml/export-0.10/}revision")
    
        if titles and revisions:
            revision = revisions[0]  # last revision
            contents = revision.findall("{http://www.mediawiki.org/xml/export-0.10/}text")
            if contents:
                docs[titles[0].text] = contents[0].text 


# Some regEx for parsing
cleanExtLinks = r"(?i)\[https?://[^\s\]]+"  # Clean external links (simple pattern)
linkRe = r"\[\[([^\|\]]+)(?:\|[^\]]*)?\]\]"  # Internal links pattern
removeLinkRe = r"\[\[[^\]]+\|([^\|\]]+)\]\]"  # Remove links with descriptions
removeLink2Re = r"\[\[([^\|\]]+)\]\]"  # Remove links without descriptions
wordRe = r"[a-zA-Z\-]+"  # Pattern for words, including hyphens in words
stopWords = ["-"]  # Stop words list, can be expanded based on requirement


# Number 4.5 

print("Extracting links, transforming links in text, tokenizing, and filling a tok-doc matrix...")
links = dict()
doctok = dict()

for idx, doc in enumerate(docs):
    if idx % (len(docs) // 20) == 0:
        print("Progress " + str(int(idx * 100 / len(docs))) + "%")
    
    links[doc] = list()  # Store the links for the document
    
    for link in re.finditer(linkRe, docs[doc]):
        target = link.group(1).split('|')[0]  # The first part of the link (target)
        display_text = link.group(2) if link.group(2) else target  # The display text, if it exists, otherwise the target
        
        if target in docs.keys():  # Only add the link if the target exists in the documents
            links[doc].append((target, display_text))  # Append both target and display text as a tuple
    
    cleanDoc = re.sub(cleanExtLinks, "", docs[doc])  # Remove external links

    # Transform links to text (remove internal link formatting)
    docs[doc] = re.sub(removeLinkRe, r"\1", cleanDoc)
    docs[doc] = re.sub(removeLink2Re, r"\1", docs[doc])

    # Tokenizing the document and filling the doctok matrix
    doctok[doc] = list()
    for wordre in re.finditer(wordRe, cleanDoc):
        word = wordre.group(0).lower()
        if word not in stopWords:
            doctok[doc].append(word)

print("done.")


print("Building tf-idf table...")
docList = list(doctok.keys())
Ndocs = len(docList)

tokInfo = defaultdict(float)  # tokInfo[tok] contains the information in bits of the token
tf = defaultdict(lambda: defaultdict(int))  # tf[doc][tok] contains the frequency of the token tok in document doc

# Calculate term frequencies (TF)
for doc, words in doctok.items():
    for word in words:
        tf[doc][word] += 1

# Calculate inverse document frequency (IDF) and store in tokInfo
for doc in docList:
    for word in tf[doc]:
        tokInfo[word] += 1

# Calculate IDF (log(N / df) + 1) where df is the document frequency
for word, df in tokInfo.items():
    tokInfo[word] = numpy.log((Ndocs + 1) / (df + 1)) + 1

print("done.")

print("creating tf-idf...", end="")
tfidf = defaultdict(dict)  # this should be in reverse sparse format

# Calculate tf-idf values
for doc, words in doctok.items():
    for word in words:
        tfidf[doc][word] = tf[doc][word] * tokInfo[word]

print("done.")

print("Saving the links and the tfidf as pickle objects...")
with open("links.dict", 'wb') as fileout:
    pickle.dump(links, fileout, protocol=pickle.HIGHEST_PROTOCOL)

with open("tfidf.dict", 'wb') as fileout:
    pickle.dump(tfidf, fileout, protocol=pickle.HIGHEST_PROTOCOL)

with open("tokInfo.dict", 'wb') as fileout:
    pickle.dump(tokInfo, fileout, protocol=pickle.HIGHEST_PROTOCOL)
