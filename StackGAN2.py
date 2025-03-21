import os
import pickle
import random
import time
import PIL
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import Input, Model, layers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, ReLU, Reshape, UpSampling2D, Conv2D, Activation, concatenate, Flatten, Lambda, Concatenate, ZeroPadding2D
from tensorflow.keras.layers import add
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt

# Set up directory structure based on your project root
project_dir = "d:\\Masters GIKI\\Course Work\\Generative AI\\Assignments\\Assignment 3\\Stack-GAN-Implementation"
data_dir = os.path.join(project_dir, "data", "coco")
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
coco_dataset_dir = os.path.join(project_dir, "data", "coco2014")
models_dir = os.path.join(project_dir, "model_weights", "stage1")
results_dir = os.path.join(project_dir, "data", "results")

# Create results directory if it doesn't exist
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Define file paths for COCO subset
embeddings_file_path_train = os.path.join(train_dir, "char-CNN-RNN-embeddings.pickle")
embeddings_file_path_val = os.path.join(val_dir, "char-CNN-RNN-embeddings.pickle")
filenames_file_path_train = os.path.join(train_dir, "filenames.pickle")
filenames_file_path_val = os.path.join(val_dir, "filenames.pickle")

# TensorBoard log directory
log_dir = os.path.join(project_dir, "logs", str(int(time.time())))
summary_writer = tf.summary.create_file_writer(log_dir)


class RandomNormalLayer(layers.Layer):
    def call(self, inputs):
        return tf.random.normal(tf.shape(inputs))


class Reparam(layers.Layer):
    """Reparameterization trick layer to sample from the latent space."""
    def call(self, inputs):
        mu, logvar = inputs
        epsilon = tf.random.normal(tf.shape(mu))
        return mu + tf.exp(logvar / 2) * epsilon


class ReparmLayer(layers.Layer):
    """Reparameterization trick as a layer."""
    def call(self, inputs):
        mu, logvar = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return mu + tf.exp(logvar / 2) * epsilon


def build_ca_model(embedding_dim=1024, condition_dim=128):
    """Get conditioning augmentation model."""
    input_layer = Input(shape=(embedding_dim,))
    x = Dense(256, activation="relu")(input_layer)
    mu = Dense(condition_dim)(x)
    logvar = Dense(condition_dim)(x)
    model = Model(inputs=[input_layer], outputs=[mu, logvar])
    return model


def build_embedding_compressor_model():
    """Build embedding compressor model."""
    input_layer = Input(shape=(1024,))
    x = Dense(128)(input_layer)
    x = ReLU()(x)
    model = Model(inputs=[input_layer], outputs=[x])
    return model


def generate_c(x):
    mean = x[:, :128]
    log_sigma = x[:, 128:]
    stddev = K.exp(log_sigma)
    epsilon = K.random_normal(shape=(K.shape(mean)[1],))
    c = stddev * epsilon + mean
    return c


def build_stage1_generator(z_dim=100, condition_dim=128):
    input_layer = layers.Input(shape=(z_dim + condition_dim,))
    
    x = layers.Dense(4*4*256, use_bias=False)(input_layer)
    x = layers.Reshape((4, 4, 256))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding="same", use_bias=False)(x)  # (8, 8, 128)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding="same", use_bias=False)(x)   # (16, 16, 64)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(32, (4,4), strides=(2,2), padding="same", use_bias=False)(x)   # (32, 32, 32)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(3, (4,4), strides=(2,2), padding="same", use_bias=False)(x)    # (64, 64, 3)
    x = layers.Activation("tanh")(x)

    model = keras.Model(inputs=input_layer, outputs=x)
    return model


def residual_block(input_tensor):
    """Residual block in the generator network."""
    x = Conv2D(128 * 4, kernel_size=(3, 3), padding='same', strides=1)(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(128 * 4, kernel_size=(3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = add([x, input_tensor])
    x = ReLU()(x)
    return x


def joint_block(inputs):
    c, x = inputs
    c = K.expand_dims(c, axis=1)
    c = K.expand_dims(c, axis=1)
    c = K.tile(c, [1, 16, 16, 1])
    return K.concatenate([c, x], axis=3)


def build_stage2_generator():
    """Build Stage-II generator."""
    input_layer = Input(shape=(1024,))
    input_lr_images = Input(shape=(64, 64, 3))

    ca = Dense(256)(input_layer)
    # Use negative_slope instead of alpha
    mean_logsigma = LeakyReLU(negative_slope=0.2)(ca)
    c = Lambda(generate_c)(mean_logsigma)

    x = ZeroPadding2D(padding=(1, 1))(input_lr_images)
    x = Conv2D(128, kernel_size=(3, 3), strides=1, use_bias=False)(x)
    x = ReLU()(x)

    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(256, kernel_size=(4, 4), strides=2, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(512, kernel_size=(4, 4), strides=2, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    c_code = Lambda(joint_block)([c, x])

    x = ZeroPadding2D(padding=(1, 1))(c_code)
    x = Conv2D(512, kernel_size=(3, 3), strides=1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    for _ in range(4):
        x = residual_block(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(512, kernel_size=3, padding="same", strides=1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(256, kernel_size=3, padding="same", strides=1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, kernel_size=3, padding="same", strides=1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, kernel_size=3, padding="same", strides=1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(3, kernel_size=3, padding="same", strides=1, use_bias=False)(x)
    x = Activation('tanh')(x)

    model = Model(inputs=[input_layer, input_lr_images], outputs=[x, mean_logsigma])
    return model


def build_stage2_discriminator():
    """Create Stage-II discriminator network."""
    input_layer = Input(shape=(256, 256, 3))

    x = Conv2D(64, (4, 4), padding='same', strides=2, use_bias=False)(input_layer)
    x = LeakyReLU(negative_slope=0.2)(x)

    x = Conv2D(128, (4, 4), padding='same', strides=2, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)

    x = Conv2D(256, (4, 4), padding='same', strides=2, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)

    x = Conv2D(512, (4, 4), padding='same', strides=2, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)

    x = Conv2D(1024, (4, 4), padding='same', strides=2, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)

    x = Conv2D(2048, (4, 4), padding='same', strides=2, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)

    x = Conv2D(1024, (1, 1), padding='same', strides=1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)

    x = Conv2D(512, (1, 1), padding='same', strides=1, use_bias=False)(x)
    x = BatchNormalization()(x)

    x2 = Conv2D(128, (1, 1), padding='same', strides=1, use_bias=False)(x)
    x2 = BatchNormalization()(x2)
    x2 = LeakyReLU(negative_slope=0.2)(x2)

    x2 = Conv2D(128, (3, 3), padding='same', strides=1, use_bias=False)(x2)
    x2 = BatchNormalization()(x2)
    x2 = LeakyReLU(negative_slope=0.2)(x2)

    x2 = Conv2D(512, (3, 3), padding='same', strides=1, use_bias=False)(x2)
    x2 = BatchNormalization()(x2)

    added_x = add([x, x2])
    added_x = LeakyReLU(negative_slope=0.2)(added_x)

    input_layer2 = Input(shape=(4, 4, 128))
    merged_input = concatenate([added_x, input_layer2])

    x3 = Conv2D(64 * 8, kernel_size=1, padding="same", strides=1)(merged_input)
    x3 = BatchNormalization()(x3)
    x3 = LeakyReLU(negative_slope=0.2)(x3)
    x3 = Flatten()(x3)
    x3 = Dense(1)(x3)
    x3 = Activation('sigmoid')(x3)

    stage2_dis = Model(inputs=[input_layer, input_layer2], outputs=[x3])
    return stage2_dis


def build_adversarial_model(gen_model2, dis_model, gen_model1):
    """Create adversarial model."""
    embeddings_input_layer = Input(shape=(1024,))
    noise_input_layer = Input(shape=(100,))
    compressed_embedding_input_layer = Input(shape=(4, 4, 128))

    gen_model1.trainable = False
    dis_model.trainable = False

    # Compute c using the single CA model instance
    mu, logvar = ca_model(embeddings_input_layer)
    
    # Use our custom layer for reparameterization
    c = ReparmLayer()([mu, logvar])

    # Concatenate noise and c for stage1_gen
    gen_input = Concatenate(axis=1)([noise_input_layer, c])
    lr_images = gen_model1(gen_input)

    hr_images, mean_logsigma2 = gen_model2([embeddings_input_layer, lr_images])
    valid = dis_model([hr_images, compressed_embedding_input_layer])

    model = Model(
        inputs=[embeddings_input_layer, noise_input_layer, compressed_embedding_input_layer],
        outputs=[valid, mean_logsigma2]
    )
    return model


def load_filenames(filenames_file_path):
    """Load filenames from filenames.pickle file."""
    with open(filenames_file_path, 'rb') as f:
        filenames = pickle.load(f, encoding='latin1')
    return filenames


def load_embeddings(embeddings_file_path):
    """Load embeddings."""
    with open(embeddings_file_path, 'rb') as f:
        embeddings = pickle.load(f, encoding='latin1')
        embeddings = np.array(embeddings)
        print('Embeddings shape:', embeddings.shape)
    return embeddings


def get_img(img_path, image_size):
    """Load and resize images."""
    img = Image.open(img_path).convert('RGB')
    img = img.resize(image_size, PIL.Image.BILINEAR)
    return np.array(img)


def load_dataset(filenames_file_path, coco_dataset_dir, embeddings_file_path, image_size):
    filenames = load_filenames(filenames_file_path)
    all_embeddings = load_embeddings(embeddings_file_path)

    X, embeddings = [], []
    print("All embeddings shape:", all_embeddings.shape)

    for index, filename in enumerate(filenames):
        try:
            img_name = os.path.join(
                coco_dataset_dir, 
                "train2014" if "train" in filenames_file_path else "val2014", 
                filename
            )
            img = get_img(img_name, image_size)
            all_embeddings1 = all_embeddings[index, :, :]
            embedding_ix = random.randint(0, all_embeddings1.shape[0] - 1)
            embedding = all_embeddings1[embedding_ix, 0, :]  # Extract (1024,) instead of (1, 1024)
            X.append(img)
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    X = np.array(X)
    embeddings = np.array(embeddings)
    return X, embeddings


def KL_loss(y_true, y_pred):
    mean = y_pred[:, :128]
    logsigma = y_pred[:, 128:]
    loss = -logsigma + 0.5 * (-1 + K.exp(2. * logsigma) + K.square(mean))
    loss = K.mean(loss)
    return loss


def custom_generator_loss(y_true, y_pred):
    return K.binary_crossentropy(y_true, y_pred)


def write_log(writer, name, value, step):
    """Write training summary to TensorBoard."""
    with writer.as_default():
        tf.summary.scalar(name, value, step=step)
        writer.flush()


def save_rgb_img(img, path):
    """Save an RGB image."""
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Image")
    plt.savefig(path)
    plt.close()


# Instantiate the CA model once and reuse it throughout training.
ca_model = build_ca_model()

if __name__ == '__main__':
    hr_image_size = (256, 256)
    lr_image_size = (64, 64)
    batch_size = 4
    z_dim = 100
    stage1_generator_lr = 0.0002
    stage1_discriminator_lr = 0.0002
    epochs = 2
    condition_dim = 128
    stage1_discriminator_lr = 0.0002
    stage1_generator_lr = 0.0002

    dis_optimizer = Adam(learning_rate=stage1_discriminator_lr, beta_1=0.5, beta_2=0.999)
    gen_optimizer = Adam(learning_rate=stage1_generator_lr, beta_1=0.5, beta_2=0.999)

    # Load datasets
    X_hr_train, embeddings_train = load_dataset(
        filenames_file_path=filenames_file_path_train,
        coco_dataset_dir=coco_dataset_dir,
        embeddings_file_path=embeddings_file_path_train,
        image_size=hr_image_size
    )

    X_hr_val, embeddings_val = load_dataset(
        filenames_file_path=filenames_file_path_val,
        coco_dataset_dir=coco_dataset_dir,
        embeddings_file_path=embeddings_file_path_val,
        image_size=hr_image_size
    )

    X_lr_train, _ = load_dataset(
        filenames_file_path=filenames_file_path_train,
        coco_dataset_dir=coco_dataset_dir,
        embeddings_file_path=embeddings_file_path_train,
        image_size=lr_image_size
    )

    X_lr_val, _ = load_dataset(
        filenames_file_path=filenames_file_path_val,
        coco_dataset_dir=coco_dataset_dir,
        embeddings_file_path=embeddings_file_path_val,
        image_size=lr_image_size
    )

    # Build and compile models
    stage2_dis = build_stage2_discriminator()
    stage2_dis.compile(loss='binary_crossentropy', optimizer=dis_optimizer)

    stage1_gen = build_stage1_generator()
    stage1_gen.compile(loss="binary_crossentropy", optimizer=gen_optimizer)
    stage1_gen.load_weights(os.path.join(models_dir, "stage1_gen_final.weights.h5"))

    stage2_gen = build_stage2_generator()
    stage2_gen.compile(loss="binary_crossentropy", optimizer=gen_optimizer)

    embedding_compressor_model = build_embedding_compressor_model()
    embedding_compressor_model.compile(loss='binary_crossentropy', optimizer='adam')

    adversarial_model = build_adversarial_model(stage2_gen, stage2_dis, stage1_gen)
    adversarial_model.compile(
        loss=['binary_crossentropy', KL_loss],
        loss_weights=[1.0, 2.0],
        optimizer=gen_optimizer,
        metrics=None
    )

    real_labels = np.ones((batch_size, 1), dtype=float) * 0.9
    fake_labels = np.zeros((batch_size, 1), dtype=float) * 0.1

    for epoch in range(epochs):
        print("========================================")
        print(f"Epoch is: {epoch}")

        gen_losses = []
        dis_losses = []

        number_of_batches = int(X_hr_train.shape[0] / batch_size)
        print(f"Number of batches: {number_of_batches}")
        for index in range(number_of_batches):
            print(f"Batch: {index + 1}")

            z_noise = np.random.normal(0, 1, size=(batch_size, z_dim))
            X_hr_train_batch = X_hr_train[index * batch_size:(index + 1) * batch_size]
            embedding_batch = embeddings_train[index * batch_size:(index + 1) * batch_size]
            X_hr_train_batch = (X_hr_train_batch - 127.5) / 127.5

            # Use the single, pre-built CA model. Wrap the input in a list.
            mu, logvar = ca_model.predict_on_batch([embedding_batch])
            epsilon = np.random.normal(0, 1, size=(batch_size, condition_dim))
            c = mu + np.exp(logvar / 2) * epsilon

            # Concatenate z_noise and c
            gen_input = np.concatenate([z_noise, c], axis=1)

            # Generate low-res images using stage1_gen
            lr_fake_images = stage1_gen.predict_on_batch(gen_input)
            hr_fake_images, _ = stage2_gen.predict([embedding_batch, lr_fake_images], verbose=3)

            compressed_embedding = embedding_compressor_model.predict_on_batch(embedding_batch)
            compressed_embedding = np.reshape(compressed_embedding, (-1, 1, 1, condition_dim))
            compressed_embedding = np.tile(compressed_embedding, (1, 4, 4, 1))

            dis_loss_real = stage2_dis.train_on_batch(
                [X_hr_train_batch, compressed_embedding],
                np.reshape(real_labels, (batch_size, 1))
            )
            dis_loss_fake = stage2_dis.train_on_batch(
                [hr_fake_images, compressed_embedding],
                np.reshape(fake_labels, (batch_size, 1))
            )
            dis_loss_wrong = stage2_dis.train_on_batch(
                [X_hr_train_batch[:(batch_size - 1)], compressed_embedding[1:]],
                np.reshape(fake_labels[1:], (batch_size - 1, 1))
            )
            d_loss = 0.5 * np.add(dis_loss_real, 0.5 * np.add(dis_loss_wrong, dis_loss_fake))
            print(f"d_loss: {d_loss}")

            g_loss = adversarial_model.train_on_batch(
                [embedding_batch, z_noise, compressed_embedding],
                [K.ones((batch_size, 1)) * 0.9, K.ones((batch_size, 256)) * 0.9]
            )
            print(f"g_loss: {g_loss}")

            dis_losses.append(d_loss)
            gen_losses.append(g_loss)

        write_log(summary_writer, 'discriminator_loss', np.mean(dis_losses), epoch)
        write_log(summary_writer, 'generator_loss', np.mean(gen_losses)[0], epoch)

        if epoch % 2 == 0:
            z_noise2 = np.random.normal(0, 1, size=(batch_size, z_dim))
            embedding_batch = embeddings_val[0:batch_size]

            # Compute c for validation using the single CA model.
            mu, logvar = ca_model.predict_on_batch([embedding_batch])
            epsilon = np.random.normal(0, 1, size=(batch_size, condition_dim))
            c = mu + np.exp(logvar / 2) * epsilon
            gen_input = np.concatenate([z_noise2, c], axis=1)

            lr_fake_images = stage1_gen.predict_on_batch(gen_input)
            hr_fake_images, _ = stage2_gen.predict([embedding_batch, lr_fake_images], verbose=3)

            for i, img in enumerate(hr_fake_images[:10]):
                save_rgb_img(img, os.path.join(results_dir, f"gen_{epoch}_{i}.png"))

    # Save the models
    stage2_gen.save_weights(os.path.join(models_dir, "stage2_gen.h5"))
    stage2_dis.save_weights(os.path.join(models_dir, "stage2_dis.h5"))
