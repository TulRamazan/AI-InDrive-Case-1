import argparse
import yaml
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np

def train_model(data_path, epochs):
    """
    Обучает модель для классификации состояния автомобиля.

    Args:
        data_path (str): Путь к YAML-файлу с настройками данных.
        epochs (int): Количество эпох для обучения.
    """
    print("Загрузка данных и их подготовка...")
    
    with open(data_path, 'r') as file:
        config = yaml.safe_load(file)

    train_dir = config['train_dir']
    val_dir = config['val_dir']
    image_size = config['image_size']
    batch_size = config['batch_size']
    num_classes = config['num_classes']

    # Аугментация данных для тренировочного набора
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Аугментация данных для валидационного набора
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    print("Создание модели...")
    # Используем предобученную модель EfficientNetB0 (transfer learning)
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(image_size, image_size, 3)
    )

    # Добавляем свои слои для классификации
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Замораживаем веса базовой модели
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Обучение модели...")
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size
    )

    print("Сохранение модели...")
    model.save('model.h5')
    print("Обучение завершено. Модель сохранена как model.h5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Обучение модели классификации автомобилей.')
    parser.add_argument('--data', type=str, required=True, help='Путь к YAML-файлу с настройками данных.')
    parser.add_argument('--epochs', type=int, default=20, help='Количество эпох для обучения.')
    
    args = parser.parse_args()
    train_model(args.data, args.epochs)
