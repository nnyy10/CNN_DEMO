import datetime
import os

from keras.layers import Dense
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

############################  Load data  ############################

train_dir = "data/1-train"
test_dir = "data/2-test"
val_dir = "data/3-validate"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    brightness_range=[0.8, 1.2],
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

test_val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

test_generator = test_val_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)

val_generator = test_val_datagen.flow_from_directory(
    val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)


############################  Load model  ############################

base_model = load_model("models/facenet_pretrained.h5")

x = base_model.output
x = Dense(1024, activation="relu")(x)
predictions = Dense(7, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False


model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

############################  Train  ############################
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=10,
)


for layer in base_model.layers:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=50,  # Additional epochs
)

############################  Evaluate  ############################
test_loss, test_acc = model.evaluate(
    test_generator, steps=test_generator.samples // BATCH_SIZE
)
print(f"Test accuracy: {test_acc:.2f}")

############################  Saving final model  ############################

models_dir = "models"
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_filename = f"xception_model_{current_time}.h5"
model_save_path = os.path.join(models_dir, model_filename)

# After model training
model.save(model_save_path)
print(f"Model saved to {model_save_path}")
