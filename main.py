import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(px_train, classification_train), (px_test, classification_test) = mnist.load_data()

px_train= tf.keras.utils.normalize(px_train, axis=1)
px_test = tf.keras.utils.normalize(px_test, axis=1)
#
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# model.add(tf.keras.layers.Dense(128,activation='relu'))
# model.add(tf.keras.layers.Dense(128,activation='relu'))
# model.add(tf.keras.layers.Dense(10,activation='softmax'))
#
# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#
# model.fit(px_train,classification_train,epochs=3)
# model.save('handwritten.model')
#
model = tf.keras.models.load_model('handwritten.model')

loss,accuracy = model.evaluate(px_test, classification_test)
print(loss)
print(accuracy)

image_number=0
while os.path.isfile(f"Digits/digit{image_number}.png"):
     try:
            img = cv2.imread(f"Digits/digit{image_number}.png")[:,:,0]
            img = np.invert(np.array([img]))
            prediction = model.predict(img)
            print(f"This digit is probaly a {np.argmax(prediction)}")
            plt.imshow(img[0],cmap=plt.cm.binary)
            plt.show()
     except:
         print("Error")
     finally:
         image_number+=1