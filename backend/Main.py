import pandas as pd
from LinearRegression import LinearRegression
import datetime
from SocketConnection import SocketConnection
import eventlet

class Main:
    requestedPrice = None
    socket = None

    def __init__(self, socket):
        self.socket = socket

    def __main__(self):
        df = pd.read_csv('csv/houses-mini.csv')
        trainingSlice = df[0:150]
        testingSlice = df[150:450]

        features = trainingSlice['sqft_living'].astype('float32')
        labels = trainingSlice['price'].astype('float32')

        testingFeatures = testingSlice['sqft_living'].astype('float32')
        testingLabels = testingSlice['price'].astype('float32')

        self.regression = LinearRegression(features, labels, {
            'learningRate': 0.05,
            'iterations': 1,
            'batchSize': 50,
        })

        self.trainingTimes = []
        self.R2 = 0
        self.mse = 0
        self.iterations = 0

        self.socket.server.emit('data', {
            'features': list(features),
            'labels': list(labels)
        })

        def think():
            startDate = datetime.datetime.now()
            self.regression.train()
            self.trainingTimes.append((datetime.datetime.now() - startDate).total_seconds() * 1000)

            self.R2 = self.regression.test(testingFeatures, testingLabels)
            self.mse = self.regression.mseHistory[0]
            
            observations = list(features)
            self.socket.server.emit('update', {
                'r2': self.R2,
                'mse': self.mse.item(),
                'iterations': self.iterations,
                'mseHistory': list(map(lambda x: x.item(), self.regression.mseHistory)),
                'trainingTimes': self.trainingTimes,
                'predictions': self.regression.predict(observations).tolist()
            })

            if self.requestedPrice is not None:
                prediction = self.regression.predict([self.requestedPrice])
                self.socket.server.emit('prediction', { 'price': prediction.tolist()[0][0] })
            
            self.socket.server.sleep(0.1)

            if self.iterations >= 250:
                return

            self.iterations = self.iterations + 1
            think()

        think()

    def setRequestedPrice(self, price):
        self.requestedPrice = price

eventlet.monkey_patch()

socket = SocketConnection()
main = Main(socket)
socket.set_main(main)
main.socket.open()
