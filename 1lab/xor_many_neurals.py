import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras import Model


# Создание модели из 7 нейронов
def createNeural(input_shape):
    input_x = Input(shape=(input_shape, ))
    z1 = Dense(6)(input_x)
    p1 = Activation('relu')(z1)
    z2 = Dense(1)(p1)
    p2 = Activation('sigmoid')(z2)
    return Model(inputs=input_x, outputs=p2)


# Обучение модели
def trainModel(X, Y):
    m = createNeural(2)
    m.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    m.fit(x=X, y=Y, epochs=5000)
    return m


# Тестирование обученной модели
def testModel(model, X_test):
    result = model.predict(X_test)
    if result >= 0.5:
        return 1
    elif result < 0.5:
        return 0
    

# Формирование обучающей выборки
X = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=tf.float32)
Y = tf.constant([[0], [1], [1], [0]], dtype=tf.float32)
# Работа с нейроном
model = trainModel(X, Y)
print(testModel(model, tf.constant([[0, 0]])))
print(testModel(model, tf.constant([[0, 1]])))
print(testModel(model, tf.constant([[1, 0]])))
print(testModel(model, tf.constant([[1, 1]])))
