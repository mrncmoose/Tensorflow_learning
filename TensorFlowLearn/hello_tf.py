'''
Created on Sep 16, 2020

@author: Fred T. Dunaway
*******  There are SSL errors downloading the dataset. ***************
*** Manually loading the data shows a very very poor data structure design that is not worth fixing.  *********
'''

import tensorflow as tf
import numpy


from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

#The tensor flow keras model class
class Tf_k(Model):
  def __init__(self):
    super(Tf_k, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10)

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)


if __name__ == '__main__':
    
    # load and prepare the MINST data set.  See:  http://yann.lecun.com/exdb/mnist/
    path = 'data_sets/mnist.npz'
    data = numpy.load(path)
    
    (x_train, y_train), (x_test, y_test) = data
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Add a channels dimension
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    #shuffle and batch the dataset
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(32)
    
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
    
    # Create an instance of the model
    model = Tf_k()
    
    #Choose optimizer and loss function
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    #Set metrics and calculate accuracy for the training set.
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    
    #Now train the model
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            train_loss(loss)
            train_accuracy(labels, predictions)
    
    #Test the model
    @tf.function
    def test_step(images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)        
        test_loss(t_loss)
        test_accuracy(labels, predictions)
        
    EPOCHS = 5

    for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
    
        for images, labels in train_ds:
            train_step(images, labels)
    
        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)
    
        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                            train_loss.result(),
                            train_accuracy.result() * 100,
                            test_loss.result(),
                            test_accuracy.result() * 100))
        