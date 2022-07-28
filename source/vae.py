import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
                                    Flatten, Dense, Reshape, Conv2DTranspose, \
                                    Activation, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

import os
import pickle

import tensorflow as tf
tf.compat.v1.disable_eager_execution()


class VAE:
    """
    VAE represents a deep convolutional VAE architecture 
    with mirrored encoder and decoder components.
    """
    
    def __init__(self,
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim):
        # review CNN notes to understand these inputs
        self.input_shape = input_shape    # [28, 28, 1]
        self.conv_filters = conv_filters  # [2, 4, 8]
        self.conv_kernels = conv_kernels  # kernel size at each layer: [3, 5, 3]
        self.conv_strides = conv_strides  # [1, 2, 2]
        self.latent_space_dim = latent_space_dim  # 2 
        self.reconstruction_loss_weight = 1000.
        
        self.encoder = None
        self.decoder = None
        self.model = None
        
        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None
        
        self._build()
        
    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()
        
    def compile(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, 
                           loss=self._calculate_combined_loss,
                           metrics=[self._calculate_reconstruction_loss,
                                    self._calculate_KL_loss])
    
    def train(self, x_train, batch_size, num_epochs):
        self.model.fit(x_train,  # input 
                       x_train,  # output
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=True)

    def save(self, save_folder="."):
        self._create_folder_if_needed(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def reconstruct(self, images):
        latent_representations = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations

    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = VAE(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder

    def _calculate_combined_loss(self, y_target, y_predicted):
        reconstruction_loss = self._calculate_reconstruction_loss(y_target, y_predicted)
        kl_loss = self._calculate_KL_loss(y_target, y_predicted)
        combined_loss = self.reconstruction_loss_weight * reconstruction_loss + kl_loss
        return combined_loss

    def _calculate_reconstruction_loss(self, y_target, y_predicted):
        error = y_target - y_predicted
        reconstruction_loss = K.mean(K.square(error), axis=[1, 2, 3])
        return reconstruction_loss

    def _calculate_KL_loss(self, y_target, y_predicted):
        kl_loss = -0.5 * K.sum(1 + self.log_var - K.square(self.mu) - K.exp(self.log_var), axis=1)
        return kl_loss

    def _create_folder_if_needed(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]

        save_path = os.path.join(folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, folder):
        save_path = os.path.join(folder, "weights.h5")
        self.model.save_weights(save_path)


    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()
    
    # AUTOENCODER ===========================================================
    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name="autoencoder")
    
    # DECODER ===============================================================
    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")
    
    def _add_decoder_input(self):
        return Input(shape=self.latent_space_dim, name="decoder_input")
    
    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck)   # [1, 2, 4] --> 8 (flatten)
        dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
        return dense_layer
    
    def _add_reshape_layer(self, dense_layer):
        return Reshape(target_shape=self._shape_before_bottleneck)(dense_layer)
    
    def _add_conv_transpose_layers(self, x):
        """
        Add convolutional transpose blocks.
        """
        # loop through all the conv layers in the reverse order and stop at the first layer
        # (i.e., mirror the conv layers of the encoder)
        for layer_index in reversed(range(1, self._num_conv_layers)):
            # Assume three conv layers in the encoder [0, 1, 2] --> decoder [2, 1, 0]
            # Goal: [1, 2] --> [2, 1]
            x = self._add_conv_transpose_layer(layer_index, x)
        return x
    
    def _add_conv_transpose_layer(self, layer_index, x):
        layer_num = self._num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_num}"
        )
        x = conv_transpose_layer(x)
        x = ReLU(name=f"decoder_relu_{layer_num}")(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
        return x
    
    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=1,  # [24, 24, 1] image shape with 1 channel
            kernel_size=self.conv_kernels[0], 
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )
        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name="sigmoid_layer")(x)
        return output_layer
    
    # ENCODER ===============================================================
    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="encoder")
    
    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_input")
    
    def _add_conv_layers(self, encoder_input):
        """
        Creates all convolutional blocks in the encoder.
        """
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x
    
    def _add_conv_layer(self, layer_index, x):
        """
        Adds a convolutional block to a graph of layers consisting of
        conv2D + ReLU + batch normalization.
        """
        layer_number = layer_index + 1
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"encoder_conv_layer_{layer_number}"
        )
        x = conv_layer(x)
        x = ReLU(name=f"encoder_relu_{layer_number}")(x)
        x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
        return x
    
    def _add_bottleneck(self, x):
        """
        Flatten data and add bottleneck layer (Dense layer) with Gaussian sampling.
        """
        self._shape_before_bottleneck = K.int_shape(x)[1:]   # e.g., [2, 7, 7, 32]
        x = Flatten()(x)
        
        # new for VAE
        self.mu = Dense(self.latent_space_dim, name="mu")(x)
        self.log_var = Dense(self.latent_space_dim, name="log_var")(x)

        def sample_point_from_normal_dist(args):
            mu, log_var = args
            eps = K.random_normal(shape=K.shape(self.mu), 
                                  mean=0.0,
                                  stddev=1.0)
            sampled_point = mu + K.exp(log_var / 2) * eps
            return sampled_point

        x = Lambda(sample_point_from_normal_dist, 
                   name="encoder_output")([self.mu, self.log_var])

        return x


if __name__ == "__main__":
    # example of how to create an Autoencoder instance and print the summary
    autoencoder = VAE(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2
    )
    autoencoder.summary()
