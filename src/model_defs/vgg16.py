############## Model Definition goes here ##############
num_chan_in = 2
height = 224
width = 224
num_classes = 6

inputs = K.layers.Input([height, width, num_chan_in], name="DICOM")

params = dict(kernel_size=(3, 3), activation="relu",
                      padding="same",
                      kernel_initializer="he_uniform")


convA = K.layers.Conv2D(name="convAa", filters=64, **params)(inputs)
convA = K.layers.Conv2D(name="convAb", filters=64, **params)(convA) #stride defaults to (1,1) for k.layers.conv
poolA = K.layers.MaxPooling2D(name="poolA", pool_size=(2, 2))(convA) #stride defaults to pool size

convB = K.layers.Conv2D(name="convBa", filters=128, **params)(poolA)
convB = K.layers.Conv2D(name="convBb", filters=128, **params)(convB)
poolB = K.layers.MaxPooling2D(name="poolB", pool_size=(2, 2))(convB)


convC = K.layers.Conv2D(name="convCa", filters=256, **params)(poolB)
convC = K.layers.Conv2D(name="convCb", filters=256, **params)(convC)
poolC = K.layers.MaxPooling2D(name="poolC", pool_size=(2, 2))(convC)

convD = K.layers.Conv2D(name="convDa", filters=512, **params)(poolC)
convD = K.layers.Conv2D(name="convDb", filters=512, **params)(convD)
poolD = K.layers.MaxPooling2D(name="poolD", pool_size=(2, 2))(convD)

convE = K.layers.Conv2D(name="convEa", filters=512, **params)(poolD)
convE = K.layers.Conv2D(name="convEb", filters=512, **params)(convE)
poolE = K.layers.MaxPooling2D(name="poolE", pool_size=(2, 2))(convE)

# img size after 5 MaxPooling (2,2) = 16*16*512 = 131072 post flattening layer
flat = K.layers.Flatten()(poolC) # 131072

dense1 = K.layers.Dense(4096, activation="relu")(flat) #VGG paper used dim which was a factor of 6.125, we followed similar strategy
dense2 = K.layers.Dense(4096, activation="relu")(dense1)
dense3 = K.layers.Dense(num_classes, activation="sigmoid")(dense2) 


model = K.models.Model(inputs=[inputs], outputs=[dense3])
opt = K.optimizers.Adam(learning_rate=0.01)
model.compile(loss=K.losses.categorical_crossentropy,
              optimizer=opt,
              metrics=['accuracy'])

