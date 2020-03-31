from keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers

# Get the training data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Create and compile the network.
model = models.Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Get training and testing sets
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=128)

# Get test elements
test_digits = test_images[0:10]

# Predict those test elements
predictions = model.predict(test_digits)

# The first element predicted according with our model
first_obtained = predictions[0].argmax()

# The expected result for the first element
first_expected = test_labels[0]

print("First element prediction: " + str(predictions[0]))

print("The element predicted as a " + str(first_obtained))

print("The element is: " + str(first_expected))

# Evaluate the result with a test set and show the average precision.
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Average test set prediction accuracy:', test_acc)