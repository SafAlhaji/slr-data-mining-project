import tensorflow as tf
from tensorflow import Tensor

class LinearRegression:
    def __init__(self, features, labels, options):
        self.g = tf.Graph()

        with self.g.as_default():
            self.features = tf.constant(features)
            self.labels = tf.constant(labels)
            self.labels = tf.reshape(self.labels, [self.labels.get_shape().as_list()[0], 1])
            self.mseHistory = []
            self.mean = None
            self.variance = None
                    
            self.features = self.processFeatures(self.features)

            self.options = {
                'learningRate': 0.1,
                'iterations': 1000,
                'batchSize': 10,
                **options,
            }

            self.weights = tf.zeros([self.features.get_shape().as_list()[1], 1])


    def gradientDescent(self, features: Tensor, labels):
        with self.g.as_default():
            differences = tf.subtract(
                tf.linalg.matmul(features, self.weights),
                labels
            )

            slopes = tf.divide(
                tf.linalg.matmul(tf.transpose(features), differences),
                features.get_shape().as_list()[0]
            )

            return tf.subtract(
                self.weights,
                tf.multiply(slopes, self.options['learningRate'])
            )


    def train(self):
        with self.g.as_default():
            batchQuantity = round(self.features.get_shape().as_list()[0] / self.options['batchSize'])
            for i in range(0, self.options['iterations']):
                for j in range(0, batchQuantity):
                    startIndex = j * self.options['batchSize']

                    with tf.compat.v1.Session():
                        featureSlice = tf.slice(self.features, [startIndex, 0], [self.options['batchSize'], -1])
                        labelSlice = tf.slice(self.labels, [startIndex, 0], [self.options['batchSize'], -1])

                        self.weights = self.gradientDescent(featureSlice, labelSlice)
                
                    self.recordMSE()


    def test(self, testFeatures, testLabels):
        with tf.compat.v1.Session(graph=self.g):
            testFeatures = tf.constant(testFeatures)
            testLabels = tf.constant(testLabels)
            testLabels = tf.reshape(testLabels, [testLabels.get_shape().as_list()[0], 1])

            testFeatures = self.processFeatures(testFeatures)

            predictions = tf.linalg.matmul(testFeatures, self.weights)

            SSres = tf.reduce_sum(
                tf.square(
                    tf.subtract(testLabels, predictions)
                )
            ).eval()

            SStot = tf.reduce_sum(
                tf.square(
                    tf.subtract(
                        testLabels,
                        tf.reduce_mean(testLabels)
                    )
                )
            ).eval()

            return 1 - (SSres / SStot)


    def processFeatures(self, features):
        if self.mean is not None and self.variance is not None:
            features = tf.subtract(features, self.mean)
            features = tf.divide(features, tf.sqrt(self.variance))
        else:
            features = self.standardize(features)

        size = features.get_shape().as_list()[0]
        features = tf.concat([tf.ones([size, 1]), tf.reshape(features, [size, 1])], 1)
        
        return features


    def standardize(self, features):
        mean, variance = tf.nn.moments(features, 0)

        self.mean = mean
        self.variance = variance

        return tf.divide(
            tf.subtract(features, mean),
            tf.sqrt(variance)
        )


    def recordMSE(self):
        with tf.compat.v1.Session(graph=self.g):
            mse = tf.divide(
                tf.reduce_sum(
                    tf.square(
                        tf.subtract(
                            tf.linalg.matmul(
                                self.features,
                                self.weights
                            ),
                            self.labels
                        )
                    )
                ),
                self.features.get_shape().as_list()[0]
            )

            self.mseHistory.insert(0, mse.eval())

    def predict(self, observations):
        with tf.compat.v1.Session(graph=self.g):
            observations = tf.constant(observations)
            observations = self.processFeatures(observations)

            return tf.matmul(observations, self.weights).eval()
