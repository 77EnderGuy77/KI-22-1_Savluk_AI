import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist # type: ignore
from keras.models import Sequential # type: ignore
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout # type: ignore
from keras.utils import to_categorical # type: ignore
# Файли mnist_test.csv та mnist_train.csv за послиланням https://mega.nz/folder/UaNgwI6J#VDO5A-aMuyyaPFLsKOesfg

# Завантаження тренувальних та тестових даних
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Перетворення даних
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255

# Перетворення міток на категорії
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Створення моделі
model = Sequential()

# Додаємо шари до моделі
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Випадкова відкидання для запобігання перенавчанню
model.add(Dense(10, activation='softmax'))

# Компіляція моделі
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Навчання моделі
model.fit(X_train, y_train, batch_size=128, epochs=10, validation_split=0.2)

# Оцінка моделі
score = model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {score[0]}')
print(f'Test accuracy: {score[1]}')

# Додатково: Візуалізація результатів
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Відображення деяких результатів
for i in range(10):
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f'Predicted: {predicted_classes[i]}, Actual: {np.argmax(y_test[i])}')
    plt.axis('off')
    plt.show()
