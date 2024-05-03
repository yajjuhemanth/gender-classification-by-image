import tensorflow as tf

# Define data paths
train_data_dir = 'D:/photos/magic/TRAIN'
validation_data_dir = 'D:/photos/magic/IFHE/BBA/Batch'

# Define image parameters
img_width, img_height = 224, 224
batch_size = 32

# Define data generators with augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Flow training images in batches using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# Flow validation images in batches using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# Load pre-trained MobileNetV2 model
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)

# Fine-tune last few layers of the base model
for layer in base_model.layers[:-5]:
    layer.trainable = False

# Add custom classification layers
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)  # Adding dropout for regularization
predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# Combine base model with custom layers
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# Compile the model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=2, min_lr=0.00001, monitor='val_loss')

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=20,  # Increase epochs to allow more training
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[early_stopping, reduce_lr])

# Save the model for future use
model.save('gender_classification_model_v2.h5')
