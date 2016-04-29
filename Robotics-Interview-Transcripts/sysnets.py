from nltk.corpus import wordnet as wn
climb= wn.synsets("car", wn.NOUN)
traverse=wn.synsets("automobile",wn.NOUN)

for version1 in list(climb):
    for version2 in traverse:
        print("Word 1"+ str(version1))
        print("Word 2" + str(version2))
        print("Path similarity")
        print(str(version1.path_similarity(version2)))
        print("Path distance.")
        print(version1.shortest_path_distance(version2))
        print(str(version1.lowest_common_hypernyms(version2)))
        print("\n")