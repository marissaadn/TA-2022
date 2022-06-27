import operator

class distanceMetrics:

#Class untuk menghitung jarak

    def __init__(self):
        
        #Inisialisasi/Constructor function

        pass
        
    def euclideanDistance(self, vector1, vector2):
        
        #Function untuk menghitung jarak Euclidean
                
        self.vectorA, self.vectorB = vector1, vector2
        if len(self.vectorA) != len(self.vectorB):
            raise ValueError("Panjang vektor tidak sama")
        distance = 0.0
        for i in range(len(self.vectorA)-1):
            distance += (self.vectorA[i] - self.vectorB[i])**2
        return (distance)**0.5

class kNNClassifier:
    
#Class untuk membangun model KNN
    
    def __init__(self):
        
        #KNN constructor
        
        pass
    
    def fit(self, xTrain, yTrain):
        
        #Train KNN model dengan x Data
        
        assert len(xTrain) == len(yTrain)
        self.trainData = xTrain
        self.trainLabels = yTrain

    def getNeighbors(self, testRow):
        
        #Train KNN model dengan x Data dan menghitung jarak dengan memanggil function Euclidean

        calcDM = distanceMetrics()
        distances = []
        for i, trainRow in enumerate(self.trainData):
            distances.append([trainRow, calcDM.euclideanDistance(testRow, trainRow), self.trainLabels[i]])
            distances.sort(key=operator.itemgetter(1))

        neighbors = []
        for index in range(self.k):
            neighbors.append(distances[index])
        return neighbors
        
    def predict(self, xTest, k, distanceMetric='euclidean'):
            
        #Menggunakan model KNN pada data testing
        
        self.testData = xTest
        self.k = k
        self.distanceMetric = distanceMetric
        predictions = []
        
        for i, testCase in enumerate(self.testData):
            neighbors = self.getNeighbors(testCase)
            output= [row[-1] for row in neighbors]
            prediction = max(set(output), key=output.count)
            predictions.append(prediction)
        
        return predictions