import numpy as np
import matplotlib.pyplot as plt


def initializeCentroids(Data, k):
    noOfRows = Data.shape[0]
    centroids = []
    # select 3 points randomly from 0 to 3929 and three are indices of 3 points in rows
    indices = np.random.randint(0, noOfRows, k)
    for i in range(k):
        centroids.append(Data[indices[i]])
    return centroids


def jaccardDistance(tweet1, tweet2):
    return 1 - (len(set(tweet1).intersection(tweet2)) / len(set().union(tweet1, tweet2)))


def findClosestCentroids(Data, centroids, k):
    noOfRows = Data.shape[0]
    clusters = dict()
    clusters.clear()
    for i in range(noOfRows):
        minDistance = np.math.inf
        indexOfCluster = -1
        for j in range(k):  # 0 1 2
            if centroids[j] == Data[i]:
                minDistance = 0
                indexOfCluster = j
                break
            tempDistance = jaccardDistance(centroids[j], Data[i])
            if tempDistance < minDistance:
                minDistance = tempDistance
                indexOfCluster = j

        if minDistance == 1:
            indexOfCluster = np.random.randint(0, k)

        clusters.setdefault(indexOfCluster, []).append([Data[i]])
        # add the tweet distance from its closest centroid to compute SSE in the end
        indexOfTweet = len(clusters.setdefault(indexOfCluster, [])) - 1
        clusters.setdefault(indexOfCluster, [])[indexOfTweet].append(minDistance)
    return clusters


def computeCentroids(clusters, k):  # check if true or not
    newCentroid = []
    newCentroid.clear()
    # iterate each cluster and check for tempSSE tweet with the closest distance sum with
    # all other tweets in the same cluster

    # select that tweet as the centroid for the cluster
    for c in range(k):
        sumOfMinDistances = np.math.inf
        indexOfCentroid = -1
        for t1 in range(len(clusters[c])):
            sumOfDistances = 0
            # get distances sum for every of tweet t1 with every tweet t2 in tempSSE same cluster
            for t2 in range(len(clusters[c])):
                dis = jaccardDistance(clusters[c][t1][0], clusters[c][t2][0])
                sumOfDistances += dis

            # select the tweet with the minimum distances sum as the centroid for the cluster
            if sumOfDistances < sumOfMinDistances:
                sumOfMinDistances = sumOfDistances
                indexOfCentroid = t1
        newCentroid.append(clusters[c][indexOfCentroid][0])
    return newCentroid


def fit(Data, maxIterations, k, tempBool=True):
    centroids = initializeCentroids(Data, k)
    SSE = []
    SSE.clear()
    previousCentroids = []
    tempSSE = 0
    for i in range(maxIterations):
        if didItChange(previousCentroids, centroids) is False:
            clusters = findClosestCentroids(Data, centroids, k)
            previousCentroids = centroids
            if tempBool is True:
                tempSSE = SSECalculation(clusters, k)
                SSE.append(tempSSE)
                print("Iteration = ", (i + 1), "\tSSE = ", tempSSE)
            else:
                print("Iteration = ", i)
            centroids = computeCentroids(clusters, k)
            clusters = findClosestCentroids(Data, centroids, k)
        else:
            if tempBool is True:
                tempSSE = SSECalculation(clusters, k)
                SSE.append(tempSSE)
                print("Iteration = ", (i + 1), "\tSSE = ", tempSSE)
            else:
                print("Iteration = ", (i + 1))
        printSizes(clusters)
    tempSSE = SSECalculation(clusters, k)
    print("The Last Centroids are = ", centroids)
    return SSE, tempSSE, clusters


def didItChange(previousCentroids, newCentroids):
    if len(previousCentroids) != len(newCentroids):
        return False
    for c in range(len(newCentroids)):
        if " ".join(newCentroids[c]) != " ".join(previousCentroids[c]):
            return False
    return True


def SSECalculation(clusters, k):
    SSE = 0
    # iterate every cluster 'c', compute SSE as the sum of square of distances of the tweet from its centroid
    for c in range(k):
        for t in range(len(clusters[c])):
            SSE = SSE + (clusters[c][t][1]) ** 2

    return SSE


def printSizes(clusters):
    for c in range(len(clusters)):
        print("Cluster " + str(c + 1) + ": " + str(len(clusters[c])) + " tweets")


def plotSSE(SSE):
    x_ax = np.arange(1, len(SSE)+1, 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use('ggplot')
    ax.set_title('SSE Over Iterations')
    ax.set_ylabel('SSE')
    ax.set_xlabel('Iteration')
    ax.plot(x_ax, SSE, marker='*', color='b', label='SSE (sum of squared error)')
    ax.legend(fontsize=9, loc='upper right')


def plotWCSS(WCSS):
    x_ax = np.arange(1, 7, 1)
    figs, asx = plt.subplots(figsize=(10, 6))
    asx.set_ylim(np.min(WCSS) - 50, np.max(WCSS) + 50)
    plt.style.use('ggplot')
    asx.set_ylabel('SSE')
    asx.set_xlabel('Number of Clusters')
    asx.set_title('Elbow Method')
    asx.plot(x_ax, WCSS, color='r', marker='^', label='WCSS')
    asx.legend(fontsize=9, loc='upper right')


path = "Health-Tweets\\bbchealth.txt"
maxIterations = 5
textChoice = input("Please enter y to change default file:\n")
if textChoice == 'y':
    fileName = input("Please enter the name of the file you want:\n")
    path = "Health-Tweets\\"+fileName+".txt"

experimentsChoice = input("Please enter y to change default number of experiments:\n")
if experimentsChoice == 'y':
    noOfExp = input("Please enter the number of maximum experiments:\n")
    maxIterations = int(noOfExp)

d = open(path, 'r', encoding='UTF-8')
tempData = []
tempDataConcatenate = []

for line in d:
    h = (line[50:-1].split("http")[0]).lower()
    h = ' '.join([word for word in h.split() if not word.startswith('@')])
    h = h.replace('#', '')
    tempDataConcatenate.append(h)
    h = h.split(' ')
    tempData.append(h)

Data = np.array(tempData, dtype="object")
DataConcatenate = np.array(tempDataConcatenate, dtype="object")
print(DataConcatenate[0:10])
SSE, temp, clusters = fit(Data, maxIterations, 3)
plotSSE(SSE)

print("\t\t\t\t\t\t*****************************************************************\n")

WCSS = []
for i in range(1, 7):
    print("No.Centroid = ", i, "\n")
    k = 1
    c, tempDistance, clusters = fit(Data, k, i, False)
    WCSS.append(tempDistance/i)
    print("\t\t\t\t\t\t---------------------------------------------\n")
plotWCSS(WCSS)
plt.show()
