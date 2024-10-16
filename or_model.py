import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras import Model


# Создание модели нейрона
def createNeural(input_shape):
    input_x = Input(shape=(input_shape, ))
    z = Dense(1)(input_x)
    p = Activation('sigmoid')(z)
    return Model(inputs=input_x, outputs=p)


# Обучение модели
def trainModel(X, Y):
    m = createNeural(2)
    m.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    m.fit(x=X, y=Y, epochs=2000)
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
Y = tf.constant([[0], [1], [1], [1]], dtype=tf.float32)
# Работа с нейроном
model = trainModel(X, Y)
print(testModel(model, tf.constant([[1, 0]])))
