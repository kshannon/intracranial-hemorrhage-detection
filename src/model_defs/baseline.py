# baseline model

num_chan_in = 2
height = 512
width = 512
num_classes = 6

inputs = K.layers.Input([height, width, num_chan_in], name="DICOM")

params = dict(kernel_size=(3, 3), activation="relu",
                      padding="same",
                      kernel_initializer="he_uniform")

convA = K.layers.Conv2D(name="convAa", filters=32, **params)(inputs)
convA = K.layers.Conv2D(name="convAb", filters=32, **params)(convA)
poolA = K.layers.MaxPooling2D(name="poolA", pool_size=(2, 2))(convA)

convB = K.layers.Conv2D(name="convBa", filters=64, **params)(poolA)
convB = K.layers.Conv2D(
    name="convBb", filters=64, **params)(convB)
poolB = K.layers.MaxPooling2D(name="poolB", pool_size=(2, 2))(convB)

convC = K.layers.Conv2D(name="convCa", filters=32, **params)(poolB)
convC = K.layers.Conv2D(
    name="convCb", filters=32, **params)(convC)
poolC = K.layers.MaxPooling2D(name="poolC", pool_size=(2, 2))(convC)

flat = K.layers.Flatten()(poolC)

drop = K.layers.Dropout(0.5)(flat)

dense1 = K.layers.Dense(128, activation="relu")(drop)

dense2 = K.layers.Dense(num_classes, activation="sigmoid")(dense1)

model = K.models.Model(inputs=[inputs], outputs=[dense2])

opt = K.optimizers.Adam()

model.compile(loss=K.losses.categorical_crossentropy,
              optimizer=opt,
              metrics=[tf.keras.metrics.CategoricalCrossentropy()])

# loss = tf.nn.weighted_cross_entropy_with_logits()
# https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits

                                                                 
model.fit_generator(TRAINING_DATA, 
                    validation_data=VALIDATION_DATA, 
                    callbacks=[checkpoint, tb_logs],
                    epochs=EPOCHS)