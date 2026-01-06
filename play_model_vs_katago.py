import os
import glob
import numpy as np
from sgfmill import sgf, boards, ascii_boards
import os
import copy
import random
import pickle
import re
from scipy.special import softmax
import ctypes
from ctypes import *
import pandas as pd
import itertools
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Layer

def read_sgf_until_first_pass(filename):
    with open(filename, "rb") as f:
        content = f.read()

    game = sgf.Sgf_game.from_bytes(content)
    board_size = game.get_size()
    if board_size != 19:
        return 0
    main_sequence = game.get_main_sequence()

    # Read through the game and store moves before the first pass
    moves = []

    for i, node in enumerate(main_sequence):
        if i == 0:
            continue

        color, move = node.get_move()
        if move is None:
            break

        moves.append((color, move))

    # Select a random move before the first pass
    target_move_index = random.randint(0, len(moves) - 1)
    target_move_index = len(moves) - 1
    target_move = moves[target_move_index]

    print(target_move_index)
    print("target_move = ", target_move)

    # Create an empty board and play through the game up to the selected move
    board = boards.Board(board_size)
    last_moves = [(-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1)]

    for i, (color, move) in enumerate(moves):
        if i > target_move_index:
            break

        row, col = move
        try:
            board.play(row, col, color)
            last_moves.append((row, col))
            if len(last_moves) > 5:
                last_moves.pop(0)
        except ValueError:
            pass


    return board, last_moves, target_move[1]

def sgfmill_board_to_numpy(board):
    board_size = board.side
    numpy_board = np.zeros((board_size, board_size), dtype=int)

    for row in range(board_size):
        for col in range(board_size):
            stone = board.get(row, col)
            if stone == 'b':
                numpy_board[row, col] = 1
            elif stone == 'w':
                numpy_board[row, col] = 2

    return numpy_board

import gc

config = {
  "version":10,
  "support_japanese_rules":True,
  "use_fixup":True,
  "use_scoremean_as_lead":False,
  "use_initial_conv_3":True,
  "use_fixed_sbscaling":True,
  "trunk_num_channels":96,
  "mid_num_channels":96,
  "regular_num_channels":64,
  "dilated_num_channels":32,
  "gpool_num_channels":32,
  "block_kind": [
    ["rconv1","regular"],
    ["rconv2","regular"],
    ["rconv3","gpool"],
    ["rconv4","regular"],
    ["rconv5","gpool"],
    ["rconv6","regular"]
  ],
  "p1_num_channels":32,
  "g1_num_channels":32,
  "v1_num_channels":32,
  "sbv2_num_channels":48,
  "v2_size":64,
  "v3_size":64
}

def get_dll_data3(board, last_move, last_move_was_a_capture):

    if last_move[0] >= 0:
        if board[last_move[0], last_move[1]] > 2 or board[last_move[0], last_move[1]] < 1:
            print("This is now the dll part")
            print("error")
            print("Here are the last moves: ", last_moves)
            board[last_move[0], last_move[1]] = board[last_move[0], last_move[1]] + 3
            print("Here is where that is on the board: ", board[last_move[0], last_move[1]])
            print(board)
            print("last_move_was_a_capture = ", last_move_was_a_capture, flush = True)
            board[last_move[0], last_move[1]] = board[last_move[0], last_move[1]] - 3
            # a = 1/0
    # load the DLL
    lib = ctypes.CDLL('./InputLabel.dll')

    # lib = ctypes.CDLL('./libInputLabel.so')

    # Define ctypes types
    IntArr19x19 = ((ctypes.c_int * 19) * 19)
    IntArr10x19x19 = (((ctypes.c_int * 19) * 19) * 12)

    # Adjust the argtypes
    lib.create_inputs.argtypes = [
        IntArr19x19,  # board
        IntArr10x19x19,  # data
        ctypes.c_int,  # last_move_x
        ctypes.c_int,  # last_move_y
        ctypes.c_int,  # last_move_was_a_capture
    ]

    # Construct the ctypes array more explicitly
    board_ct = IntArr19x19()
    for i in range(19):
        for j in range(19):
            board_ct[i][j] = board[i, j]

    data = IntArr10x19x19()

    # Call the shared library function
    lib.create_inputs(board_ct, data, last_move[0], last_move[1], int(last_move_was_a_capture))

    # Convert the returned data to a numpy array for easier handling
    data_np = np.ctypeslib.as_array(data)
    return data_np

def create_input_data_individual3(board, last_moves, global_data):
    side_to_move = global_data[6]
    last_move_was_a_capture = global_data[7]
    komi = global_data[4]

    input_data = np.zeros((19, 19, 23), dtype=float)
    move_history = np.zeros((19, 19, 5), dtype=int)

    if True:
        for i in range(0, 5):
            if i < len(last_moves):
                r, c = last_moves[i]
                if r >= 0:
                    move_history[r, c, i] = 1


    additional_inputs = get_dll_data3(board.astype(int), last_moves[4], last_move_was_a_capture)
    legal_moves = additional_inputs[8]

    if side_to_move == 1:
        input_data[:, :, 21] = additional_inputs[10]
        input_data[:, :, 22] = additional_inputs[11]

    else:
        input_data[:, :, 21] = additional_inputs[11]
        input_data[:, :, 22] = additional_inputs[10]

    failed_ladders_capped = np.where(additional_inputs[3] < 4, 0, additional_inputs[3])

    # Clip values
    additional_inputs[6] = np.clip(additional_inputs[6], None, 20) # Number of liberties of each group
    input_data[:, :, 10] = np.clip(additional_inputs[7], None, 50) # Number of stones in each group
    input_data[:, :, 21] = np.clip(input_data[:, :, 21], None, 50) # Number of black stones that would connect if play here
    input_data[:, :, 22] = np.clip(input_data[:, :, 22], None, 50) # Number of white stones that would connect if play here

    # Apply log(x + 1) transformation
    input_data[:, :, 10] = np.log(input_data[:, :, 10] + 1)
    input_data[:, :, 21] = np.log(input_data[:, :, 21] + 1)
    input_data[:, :, 22] = np.log(input_data[:, :, 22] + 1)

    # Normalize by dividing by log(51)
    input_data[:, :, 10] = input_data[:, :, 10] / np.log(51)
    input_data[:, :, 21] = input_data[:, :, 21] / np.log(51)
    input_data[:, :, 22] = input_data[:, :, 22] / np.log(51)

    group_labels = additional_inputs[4].copy()
    group_labels[group_labels >= 1000] -= 1000
    additional_inputs[4] = (additional_inputs[4] - group_labels)/1000

    input_data[board == side_to_move, 0] = 1                # My stones one hot
    input_data[board == (3 - side_to_move), 1] = 1          # Opponent stones one hot
    input_data[:, :, 2] = additional_inputs[4]              # Matrix showing the illegal move due to ko (if there is one)
    input_data[:, :, 3] = additional_inputs[5]              # Matrix showing legal moves that would start a ko
    input_data[:, :, 4] = additional_inputs[6] / 20         # number of liberties of each group
    input_data[:, :, 5] = (additional_inputs[6]  == 1)      # Groups with one liberty
    input_data[:, :, 6] = (additional_inputs[6]  == 2)      # Groups with two liberties
    input_data[:, :, 7] = (additional_inputs[6]  == 3)      # Groups with three liberties
    input_data[:, :, 8] = (additional_inputs[6]  == 4)      # Groups with four liberties
    input_data[:, :, 9] = (additional_inputs[6]  == 5)      # Groups with five liberties
    # input_data[:, :, 10] = additional_inputs[7]             # Number of stones in each group
    input_data[:, :, 11] = additional_inputs[0]             # Ladder matrix where attacker has infinite ko threats
    input_data[:, :, 12] = additional_inputs[1]             # Ladder matrix where defender has infinite ko threats
    input_data[:, :, 13] = additional_inputs[2] / 50        # The max depth of working ladders
    input_data[:, :, 14] = failed_ladders_capped / 50       # The max depth of failed ladders
    input_data[:, :, 15] = additional_inputs[9]             # Groups that have two eyes
    input_data[:, :,16:21] = move_history                   # Move history of last five moves

    global_input = np.zeros((5), dtype=float)
    global_input[0] = komi / 10
    global_input[1] = np.sum(board > 0) / 360.0     # This roughly gives the move number
    global_input[2] = (np.sum(board == side_to_move) - np.sum(board == (3 - side_to_move))) / 20    # How many captures there are
    global_input[3] = np.sum(input_data[:, :, 2])
    global_input[4] = np.sign(komi)

    return input_data, legal_moves, global_input, group_labels

class GroupModel(tf.keras.Model):
    def __init__(self, config):
        super(GroupModel, self).__init__()
        self.config = config

        initial_learning_rate = (16e-4) / 2000000000000
        self.lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[50000, 100000, 150000],
            values=[initial_learning_rate, initial_learning_rate / 2, initial_learning_rate / 4, initial_learning_rate / 8]
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)

        # self.value_head_dense1 = tf.keras.layers.Dense(units=config["v2_size"], activation='tanh', kernel_initializer=tf.keras.initializers.HeNormal())
        # self.value_head_dense3 = tf.keras.layers.Dense(units=1, activation='tanh', kernel_initializer=tf.keras.initializers.HeNormal())

        self.sb_head_dense1 = tf.keras.layers.Dense(units=config["v2_size"], activation='tanh', kernel_initializer=tf.keras.initializers.HeNormal())
        self.sb_head_dense3 = tf.keras.layers.Dense(units=1, activation='tanh', kernel_initializer=tf.keras.initializers.HeNormal())
        self.input_conv = tf.keras.layers.Conv2D(filters=config["trunk_num_channels"], kernel_size=5, padding='same', kernel_initializer=tf.keras.initializers.HeNormal())
        self.global_input_dense = tf.keras.layers.Dense(units=config["trunk_num_channels"], kernel_initializer=tf.keras.initializers.HeNormal())

        self.num_blocks = len(config["block_kind"])
        self.res_conv1 = [tf.keras.layers.Conv2D(filters=config["trunk_num_channels"], kernel_size=3, padding='same', kernel_initializer=tf.keras.initializers.HeNormal()) for _ in range(self.num_blocks)]
        self.res_bn1 = [tf.keras.layers.BatchNormalization() for _ in range(self.num_blocks)]
        self.res_conv2 = [tf.keras.layers.Conv2D(filters=config["trunk_num_channels"], kernel_size=3, padding='same', kernel_initializer=tf.keras.initializers.HeNormal()) for _ in range(self.num_blocks)]
        self.res_bn2 = [tf.keras.layers.BatchNormalization() for _ in range(2)]

        self.global_conv = [tf.keras.layers.Conv2D(filters=config["gpool_num_channels"], kernel_size=3, padding='same', kernel_initializer=tf.keras.initializers.HeNormal()) for block_name, block_type in config["block_kind"] if block_type == "gpool"]
        self.global_fc = [tf.keras.layers.Dense(units=config["trunk_num_channels"], kernel_initializer=tf.keras.initializers.HeNormal()) for block_name, block_type in config["block_kind"] if block_type == "gpool"]

        self.conv = tf.keras.layers.Conv2D(filters=config["p1_num_channels"], kernel_size=1, padding='same', kernel_initializer=tf.keras.initializers.HeNormal())

        self.g1_conv = tf.keras.layers.Conv2D(filters=config["g1_num_channels"], kernel_size=1, padding='same', kernel_initializer=tf.keras.initializers.HeNormal())
        self.g1_fc = tf.keras.layers.Dense(units=config["p1_num_channels"], kernel_initializer=tf.keras.initializers.HeNormal())
        self.policy_conv = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='same', kernel_initializer=tf.keras.initializers.HeNormal())

        self.ownership_conv = tf.keras.layers.Conv2D(1, kernel_size=1, padding='same', kernel_initializer=tf.keras.initializers.HeNormal())


        # self.sb_head_dense1 = tf.keras.layers.Dense(units=config["v2_size"], activation='relu', use_bias=True, kernel_initializer=tf.keras.initializers.HeNormal())
        # self.sb_head_dense3 = tf.keras.layers.Dense(units=1, use_bias=True, kernel_initializer=tf.keras.initializers.GlorotNormal())

        self.value_head_conv = tf.keras.layers.Conv2D(filters=config["v1_num_channels"], kernel_size=1, use_bias=True, kernel_initializer=tf.keras.initializers.HeNormal())
        self.value_head_dense1 = tf.keras.layers.Dense(units=config["v2_size"], activation='relu', use_bias=True, kernel_initializer=tf.keras.initializers.HeNormal())
        self.value_head_dense3 = tf.keras.layers.Dense(units=1, use_bias=True, kernel_initializer=tf.keras.initializers.GlorotNormal())

        self.offset_error = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.winrate_size_loss = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.sb_loss = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.value_loss = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.unmodified_value_loss = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.policy_loss = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.cross_ent_loss = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.false_pos_loss = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.ownership_loss = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.L2_loss = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.global_counter = tf.Variable(0.0, dtype=tf.float32, trainable=False)

        self.basic_sb_loss = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.basic_value_loss = tf.Variable(0.0, dtype=tf.float32, trainable=False)

    def reinitialize_value_layers(self):
        # Assuming the shapes of the weights and biases are known
        dense1_kernel_shape = (128, 64)  # Replace with the actual shape
        dense1_bias_shape = (64,)       # Replace with the actual shape
        dense3_kernel_shape = (64, 1)   # Replace with the actual shape
        dense3_bias_shape = (1,)        # Replace with the actual shape

        # Reinitialize value_head_dense1
        new_kernel1 = tf.keras.initializers.HeNormal()(shape=dense1_kernel_shape)
        new_bias1 = tf.zeros(shape=dense1_bias_shape)
        self.value_head_dense1.set_weights([new_kernel1, new_bias1])

        # Reinitialize value_head_dense3
        new_kernel3 = tf.keras.initializers.GlorotNormal()(shape=dense3_kernel_shape)
        new_bias3 = tf.zeros(shape=dense3_bias_shape)
        self.value_head_dense3.set_weights([new_kernel3, new_bias3])

    def call(self, input_data, global_input, group_labels_batch, training=False):
        label_range = tf.range(1, 129, dtype=group_labels_batch.dtype)
        label_range = label_range[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :]

        # Expand labels for broadcasting
        labels = group_labels_batch[:, :, :, tf.newaxis, tf.newaxis]

        # Broadcasting comparison
        masks = tf.equal(labels, label_range)
        masks = tf.cast(masks, dtype=tf.float32)


        mean_mask = input_data[:, :, :, 0] + input_data[:, :, :, 1] + tf.cast(tf.equal(group_labels_batch, 0), tf.float32) - 1
        mean_mask = tf.reshape(mean_mask, (input_data.shape[0], 19, 19, 1))
        # sum_mask = mean_mask * group_sizes / 10

        input_conv_output = self.input_conv(input_data, training=training)

        global_input_processed = self.global_input_dense(global_input)
        res_block_output = input_conv_output + tf.reshape(global_input_processed, [-1, 1, 1, global_input_processed.shape[-1]])

        j = 0
        for i in range(self.num_blocks):
            block_name, block_type = self.config["block_kind"][i]
            if block_type == "regular":
                res_block_output = self.res_conv_block(res_block_output, i, training=training)
            elif block_type == "gpool":
                res_block_output = self.global_res_conv_block(res_block_output, i, j, training=training)
                j += 1

            if i == 3 or i == 5:
                sum_slice = self.group_link_sum(res_block_output[:, :, :, 0:16], masks) + res_block_output[:, :, :, 0:16] * mean_mask/10
                mean_slice = self.group_link_mean(res_block_output[:, :, :, 16:32], masks) + res_block_output[:, :, :, 16:32] * mean_mask
                res_block_output = tf.concat([sum_slice, mean_slice, res_block_output[:, :, :, 32:]], axis=-1)

                # sum_slice = self.group_link_sum(res_block_output[:, :, :, 0:16], masks)
                # res_block_output = tf.concat([sum_slice, res_block_output[:, :, :, 16:]], axis=-1)

            if False and i == 5:
                board = input_data[:, :, :, 0] + 2 * input_data[:, :, :, 1]

                board = board.numpy().reshape(19,19)
                black_positions = np.argwhere(board == 1)
                white_positions = np.argwhere(board == 2)
                # input_global = 10*input_global_batch.numpy()
                # print(input_global)
                # komi = str(input_global[0, 0])
                # komi_2 = str(-global_data[0, 4])
                def plot_heatmap(data, ax, title):
                    cax = ax.matshow(data, cmap='viridis')
                    ax.scatter(black_positions[:, 1], black_positions[:, 0], c='black', marker='o', s=25, label="Black Stones")
                    ax.scatter(white_positions[:, 1], white_positions[:, 0], c='white', marker='o', s=25, label="White Stones", edgecolor='black')
                    ax.set_title(title)
                    plt.colorbar(cax, ax=ax)

                fig, axs = plt.subplots(3, 4, figsize=(18,10))

                for row in range(3):
                    for col in range(4):
                        plot_heatmap(res_block_output[0, :, :, row + col * 3].numpy(), axs[row, col], "Target")
                # plot_heatmap(np.log10(softmaxed_policy_output.numpy()).reshape(19,19), axs[1], "Ouput")
                # plot_heatmap(mask.numpy().reshape(19,19), axs[2], "Kata Top 3 = " +str(kld_2.numpy()))

                plt.tight_layout()
                plt.show()

                fig, axs = plt.subplots(3, 4, figsize=(18,10))

                for row in range(3):
                    for col in range(4):
                        plot_heatmap(res_block_output[0, :, :, 16 + row + col * 3].numpy(), axs[row, col], "Target")
                # plot_heatmap(np.log10(softmaxed_policy_output.numpy()).reshape(19,19), axs[1], "Ouput")
                # plot_heatmap(mask.numpy().reshape(19,19), axs[2], "Kata Top 3 = " +str(kld_2.numpy()))

                plt.tight_layout()
                plt.show()


        res_block_output = tf.nn.relu(res_block_output)
        policy_head = self.conv(res_block_output, training=training)
        global_policy_head = self.g1_conv(res_block_output, training=training)
        gp_pooled = self.global_pool(global_policy_head)
        gp_dense = self.g1_fc(gp_pooled)
        policy_head = policy_head + gp_dense
        policy_head = tf.nn.relu(policy_head)
        policy_output = self.policy_conv(policy_head)

        value_head = self.value_head_conv(res_block_output, training=training)
        value_head_pooled = self.value_pool(value_head)    # This takes two values from each of the 32 channels outputted by value_head_conv. It takes the average and the maximum of each channel. This leads to it outputting a shape of (batchsize, 64)
        value_head_dense_0 = self.value_head_dense1(value_head_pooled, training=training)
        value_output = self.value_head_dense3(value_head_dense_0, training=training)
        value_output = tf.squeeze(value_output)

        ownership_output = self.ownership_conv(value_head, training=training)

        return policy_output, value_output, ownership_output

    def group_link_sum(self, in_layer, masks):
        expanded_in_layer = in_layer[:, :, :, :, tf.newaxis] * masks

        # Compute sums
        sums = tf.reduce_sum(expanded_in_layer, axis=[1, 2])  # Shape: (batch_size, 4, 63)
        sum_masked = sums[:, tf.newaxis, tf.newaxis, :, :] * masks
        sum_output = tf.reduce_sum(sum_masked, axis=-1)  # Shape: (batch_size, 19, 19, 4)

        # Reshape and concatenate sum_output and mean_output
        sum_output = tf.reshape(sum_output, [sum_output.shape[0], 19, 19, 16])

        return sum_output / 10

    def group_link_mean(self, in_layer, masks):
        expanded_in_layer = in_layer[:, :, :, :, tf.newaxis] * masks

        sums = tf.reduce_sum(expanded_in_layer, axis=[1, 2])  # Shape: (batch_size, 4, 63)
        sum_masked = sums[:, tf.newaxis, tf.newaxis, :, :] * masks
        sum_output = tf.reduce_sum(sum_masked, axis=-1)  # Shape: (batch_size, 19, 19, 4)

        # Compute means
        means = sums / (tf.reduce_sum(masks, axis=[1, 2]) + 1e-10)  # Shape: (batch_size, 4, 63)
        mean_masked = means[:, tf.newaxis, tf.newaxis, :, :] * masks
        mean_output = tf.reduce_sum(mean_masked, axis=-1)  # Shape: (batch_size, 19, 19, 4)

        # Reshape and concatenate sum_output and mean_output
        mean_output = tf.reshape(mean_output, [mean_output.shape[0], 19, 19, 16])

        return mean_output

    def res_conv_block(self, in_layer, block_idx, training=False):
        relu_layer_0 = tf.nn.relu(in_layer)
        conv_layer_0 = self.res_conv1[block_idx](relu_layer_0, training=training)
        bn_layer_0 = self.res_bn1[block_idx](conv_layer_0, training=training)
        relu_layer_1 = tf.nn.relu(bn_layer_0)
        output = self.res_conv2[block_idx](relu_layer_1, training=training)

        return output + in_layer

    def global_res_conv_block(self, in_layer, block_idx, global_block_idx, training=False):
        relu_layer_0 = tf.nn.relu(in_layer)

        conv_layer = self.res_conv1[block_idx](relu_layer_0, training=training)
        conv_layer_global = self.global_conv[global_block_idx](relu_layer_0, training=training)

        bn_layer_0 = self.res_bn1[block_idx](conv_layer_global, training=training)
        # relu_layer_1 = tf.nn.relu(bn_layer_0)
        pooled_layer = self.global_pool(bn_layer_0)
        fc_layer = self.global_fc[global_block_idx](pooled_layer)

        combined_layer = conv_layer + fc_layer

        bn_layer_1 = self.res_bn2[global_block_idx](combined_layer, training=training)
        relu_layer_2 = tf.nn.relu(bn_layer_1)
        output = self.res_conv2[block_idx](relu_layer_2, training=training)

        return output + in_layer

    def global_pool(self, input_tensor):
        # Compute mean and max pooling along height and width dimensions
        mean_pool = tf.reduce_mean(input_tensor, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(input_tensor, axis=[1, 2], keepdims=True)
        mean_relu = tf.reduce_mean(tf.nn.relu(input_tensor), axis=[1, 2], keepdims=True)

        # Concatenate mean and max tensors along the last dimension (channels)
        pooled_output = tf.concat([mean_pool, max_pool, mean_relu], axis=-1)

        return pooled_output

    def value_pool(self, input_tensor):
        input_tensor = tf.math.tanh(input_tensor)
        # Compute mean and max pooling along height and width dimensions
        mean_pool_positive = tf.reduce_mean(tf.nn.relu(input_tensor), axis=[1, 2], keepdims=True)
        mean_pool_negative = tf.reduce_mean(tf.nn.relu(-input_tensor), axis=[1, 2], keepdims=True)
        square_pool_positive = tf.reduce_mean(tf.square(tf.nn.relu(input_tensor)), axis=[1, 2], keepdims=True)
        square_pool_negative = tf.reduce_mean(tf.square(tf.nn.relu(-input_tensor)), axis=[1, 2], keepdims=True)

        pooled_output = tf.concat([mean_pool_positive, mean_pool_negative, square_pool_positive, square_pool_negative], axis=-1)

        return pooled_output

def softmax_2d(array_2d):
    # Flatten the 2D array and compute the softmax
    flattened = np.exp(array_2d - np.max(array_2d))
    flattened /= np.sum(flattened)
    # Reshape the result back into the original shape
    return flattened.reshape(array_2d.shape)


file_name = "C:\\Users\\jckk2\\Downloads\\Training_Data\\play.sgf"
board_sgfmill, last_moves, next_move = read_sgf_until_first_pass(file_name)

import subprocess
import threading
import time
import math
import matplotlib.pyplot as plt

print("About to load KataGo", flush = True)
katago_command = ".//katago-v1.12.4-eigen-windows-x64//katago.exe gtp -model .//katago-v1.12.4-eigen-windows-x64//kata1-b6c96-best.txt.gz"
# katago_command = "katago.exe gtp -model b18c384nbt-uec.bin.gz"

process = subprocess.Popen(katago_command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

threading.Thread(args=(process, process.stdout, "Output"), daemon=True).start()
threading.Thread(args=(process, process.stderr, "Info"), daemon=True).start()
print("starting while", flush = True)
i = 0
while i < 1e5:
    i += 1
    line = process.stderr.readline().strip()
    # print(f"Info: {line}")
    if "GTP ready, beginning main protocol loop" in line:
        print(line, flush = True)
        break
print("done while", flush = True)

def get_kata_output(board, last_moves, global_data):

    arrays = []
    letters = "ABCDEFGHJKLMNOPQRST"
    commands = ["boardsize 19", "clear_board", "komi 7.5"]
    for command in commands:
        process.stdin.write(command + "\n")
        process.stdin.flush()

    move_num = 0
    for move in last_moves:
        if move_num % 2 == 0:
            player = "play B "
        else:
            player = "play W "

        go_coord = str(letters[move[0]]) + str(move[1] + 1)
        process.stdin.write(player + go_coord + "\n")
        process.stdin.flush()
        move_num += 1

    process.stdin.flush()
    process.stdin.write("kata-raw-nn 0\n")
    process.stdin.flush()

    count = 0
    policy_count = 0
    winrate_output = 0
    sb_output = 0

    while True:
        line = process.stdout.readline().strip()
        if "whiteWin" in line and policy_count == 0:
            parts = line.split()
            winrate_output = float(parts[1])

        if "whiteLoss" in line and policy_count == 0:
            parts = line.split()
            winrate_output -= float(parts[1])

        if "whiteLead" in line and policy_count == 0:
            parts = line.split()
            sb_output -= float(parts[1])
            sb_output = math.tanh(float(sb_output)/10.0)

        if "policy" in line and policy_count == 0:
            policy_count = 20
        elif policy_count > 1:
            array = np.fromstring(line.strip(), sep=' ')
            for i in range(len(array)):
                if not (array[i] <= 1 and array[i] >= 0):
                    array[i] = 0
            if len(array) == 19:
                arrays.append(array)
                policy_count -= 1
            if policy_count == 1:
                # print(arrays)
                combined_array = np.stack(arrays)
                flipped_array = np.transpose(combined_array)
                policy_output = np.fliplr(flipped_array)
                arrays.clear()

        if "whiteOwnership" in line:
            count = 19
        elif count > 0:
            array = np.fromstring(line.strip(), sep=' ')
            arrays.append(array)
            count -= 1
            if count == 0:
                combined_array = np.stack(arrays)
                flipped_array = np.transpose(combined_array)
                output = np.fliplr(flipped_array)
                arrays.clear()
                break


    # black_positions = np.argwhere(board == 1)
    # white_positions = np.argwhere(board == 2)
    # def plot_heatmap(data, ax, title):
    #     cax = ax.matshow(data, cmap='viridis')
    #     ax.scatter(black_positions[:, 1], black_positions[:, 0], c='black', marker='o', s=100, label="Black Stones")
    #     ax.scatter(white_positions[:, 1], white_positions[:, 0], c='white', marker='o', s=100, label="White Stones", edgecolor='black')
    #     ax.set_title(title)
    #     plt.colorbar(cax, ax=ax)
    #
    # fig, axs = plt.subplots(1, 1, figsize=(15,5))
    #
    # plot_heatmap(policy_output, axs, "Policy Array")
    # # plot_heatmap(value_array * correct_winrate_for_white_to_play, axs[1], "Value Array")
    # # plot_heatmap(puct_array, axs[2], "PUCT Array")
    #
    # # plt.tight_layout()
    # plt.show()

    winrate_output = np.array(-winrate_output)
    sb_output = np.array(sb_output/2 + 0.5)

    return policy_output.reshape(1, 19, 19), winrate_output.reshape(1, 1), sb_output.reshape(1, 1), output.reshape(1, 19, 19)

katago_command = ".//katago-v1.12.4-eigen-windows-x64//katago.exe gtp -model .//katago-v1.12.4-eigen-windows-x64//b18c384nbt-uec.bin.gz"
process2 = subprocess.Popen(katago_command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

threading.Thread(args=(process2, process.stdout, "Output"), daemon=True).start()
threading.Thread(args=(process2, process.stderr, "Info"), daemon=True).start()

print("Done loading KataGo", flush = True)

i = 0
while i < 1e4:
    i += 1
    line = process2.stderr.readline().strip()
    # print(f"Info: {line}")
    if "GTP ready, beginning main protocol loop" in line:
        break

def get_kata_output2(board, last_moves, global_data):
    board = board.reshape(19,19)
    komi = str(-global_data[4])

    # katago_command = "katago.exe gtp -model kata1-b6c96-best.txt.gz"

    arrays = []
    letters = "ABCDEFGHJKLMNOPQRST"
    commands = ["boardsize 19", "clear_board", "komi " + komi]

    for command in commands:
        process2.stdin.write(command + "\n")
        process2.stdin.flush()


    move_num = 0
    for move in last_moves:
        if move_num % 2 == len(last_moves) % 2:
            player = "play B "
        else:
            player = "play W "

        go_coord = str(letters[move[0]]) + str(move[1] + 1)
        process2.stdin.write(player + go_coord + "\n")
        process2.stdin.flush()
        move_num += 1

    process2.stdin.flush()
    process2.stdin.write("kata-raw-nn 0\n")
    process2.stdin.flush()

    count = 0
    policy_count = 0
    winrate_output = 0
    sb_output = 0

    while True:
        line = process2.stdout.readline().strip()
        if "whiteWin" in line and policy_count == 0:
            parts = line.split()
            winrate_output = float(parts[1])

        if "whiteLoss" in line and policy_count == 0:
            parts = line.split()
            winrate_output -= float(parts[1])

        if "whiteLead" in line and policy_count == 0:
            parts = line.split()
            sb_output -= float(parts[1])
            # sb_output = math.tanh(float(sb_output)/10.0)

        if "policy" in line and policy_count == 0:
            policy_count = 20
        elif policy_count > 1:
            array = np.fromstring(line.strip(), sep=' ')
            for i in range(len(array)):
                if not (array[i] <= 1 and array[i] >= 0):
                    array[i] = 0
            if len(array) == 19:
                arrays.append(array)
                policy_count -= 1
            if policy_count == 1:
                # print(arrays)
                combined_array = np.stack(arrays)
                flipped_array = np.transpose(combined_array)
                policy_output = np.fliplr(flipped_array)
                arrays.clear()

        if "whiteOwnership" in line:
            count = 19
        elif count > 0:
            array = np.fromstring(line.strip(), sep=' ')
            arrays.append(array)
            count -= 1
            if count == 0:
                combined_array = np.stack(arrays)
                flipped_array = np.transpose(combined_array)
                output = np.fliplr(flipped_array)
                arrays.clear()
                break


    # black_positions = np.argwhere(board == 1)
    # white_positions = np.argwhere(board == 2)
    # def plot_heatmap(data, ax, title):
    #     cax = ax.matshow(data, cmap='viridis')
    #     ax.scatter(black_positions[:, 1], black_positions[:, 0], c='black', marker='o', s=100, label="Black Stones")
    #     ax.scatter(white_positions[:, 1], white_positions[:, 0], c='white', marker='o', s=100, label="White Stones", edgecolor='black')
    #     ax.set_title(title)
    #     plt.colorbar(cax, ax=ax)
    #
    # fig, axs = plt.subplots(1, 1, figsize=(15,5))
    #
    # plot_heatmap(policy_output, axs, "Policy Array")
    # # plot_heatmap(value_array * correct_winrate_for_white_to_play, axs[1], "Value Array")
    # # plot_heatmap(puct_array, axs[2], "PUCT Array")
    #
    # # plt.tight_layout()
    # plt.show()

    # winrate_output = np.array(winrate_output/2 + 0.5) # From white's point of view always, from the interval 0-1
    winrate_output =  np.array(-winrate_output)
    sb_output = np.array(sb_output)

    return policy_output.reshape(1, 19, 19), winrate_output.reshape(1, 1), sb_output.reshape(1, 1), output.reshape(1, 19, 19)


print("Done loading KataGo", flush = True)
model_name = 'kata'
model_name2 = 'kata_black_winrate_fix'

board = sgfmill_board_to_numpy(board_sgfmill)
color = board[last_moves[4][0], last_moves[4][1]] % 2 + 1
capture_last_move = 0



model = GroupModel(config)
model(tf.zeros([1, 19, 19, 23]), tf.zeros([1, 5]), tf.zeros([1, 19, 19]))
model_name = 'model_new_cross_ent_step=268200'
model.load_weights(model_name+".h5")

# model2 = GroupModel(config)
# model2(tf.zeros([1, 19, 19, 23]), tf.zeros([1, 5]), tf.zeros([1, 19, 19]))
# model_name2 = 'model_new_cross_ent_step=268200'
# # model_name2 = 'model_ultim_3_step=113600'
# model2.load_weights(model_name2+".h5")


# model.load_weights('model_square_loss_step=138000.h5')
# model.load_weights('model_50k_games_step=222000.h5')

global move_number
move_number = 0

num_playouts = 512

# model2.load_weights('model_lr_16e-4_step=290000.h5') # player 2

def generate_move2(board, last_moves, color, capture_last_move, move_limit):
    # These are arrays to store inputs to the model.
    board_states = []
    move_history = []
    side_to_move = []
    previous_move_was_capture = []

    # These are arrays to store the output of the model.
    policy_outputs = []
    black_winrates = []
    liberty_matrix = []
    group_labels = []

    # These arrays don't have anything directly to do with the model. They contain important info about the PUCT algorithm.
    total_visits = []
    move_visits = []
    children_index = []

    # This array stores the parent of each position and the move that lead to the child.
    # For the previous nine arrays, they are appended to right after the model runs and we know all the necessary values to append.
    # However, this array is appended to right at the end of the loop. This is because this stores the parent of each position and the move that lead to the child.
    # However, the move is not known until after the PUCT algorithm runs - so it must be appended to at the end of the loop
    parents = [[-1, (-1, -1)]]

    position_number = 0
    depths = np.zeros(100)
    average_depth = 0

    while(True):
        ############## Part 1: Run the model ##############
        # The start of the loop: At the end of the last loop, we have just placed a stone on the board. We added it to last_moves.
        # This is Part 1. We are just concerned about getting the model to make a prediction about this position. That involves setting up things correctly and then running the model
        # All the PUCT algorithm upkeep is handled in the 2nd to 5th parts of the code.

        if color == 2:
            komi = 7.5
            board_temp = (3 - board) % 3
        else:
            komi = -7.5
            board_temp = board
        global_data = np.array([0, 0, 0, 0, komi, 0, 1, capture_last_move])

        # policy_output, value_output, sb_output, ownership_output = get_kata_output(board_temp, last_moves, global_data)

        input_data, legal_moves, global_input, group_label = create_input_data_individual3(board_temp, last_moves, global_data)
        liberty_matrix.append(input_data[:, :, 5])
        group_labels.append(group_label)

        # input_data[:, :, 11] = 0
        # input_data[:, :, 12] = 0
        # input_data[:, :, 13] = 0
        # input_data[:, :, 14] = 0
        # input_data[:, :, 15] = 0

        # input_data[:, :, 13] = (input_data[:, :, 13] > 0).astype(np.float32)
        # input_data[:, :, 14] = (input_data[:, :, 14] > 0).astype(np.float32)
        # input_data[:, :, 21] = 0
        # input_data[:, :, 22] = 0


        # Run the model
        policy_output, value_output, ownership_output = model2(np.expand_dims(input_data, axis=0), np.expand_dims(global_input, axis=0), np.expand_dims(group_label, axis=0))
        # Filter illegal moves
        policy_output = policy_output.numpy().reshape(19, 19)
        policy_output = policy_output
        min = np.min(policy_output)
        policy_output[legal_moves == 0] = min - 10000
        policy_output = softmax_2d(policy_output)
        value_output = tf.tanh(value_output)
        value_output = value_output.numpy()

        ############## Part 2: Figure out the suggested moves and winrate of the model ##############
        # This involves two things. Figure out the suggested moves and then finding the black_winrate


        # Find the black_winrate
        if color == 1:
            black_winrate = value_output
        else:
            black_winrate = -value_output

        ############## Part 3: Record this data to the lists which keep track of the search tree ##############

        # For this, we record the suggested moves and black's winrate from Part 2
        # It is also important to keep track of the input data from part 1. If we come back to this position later, we need to know what were the position, last moves, side to move and if the last move was a capture

        board_states.append(board.copy())
        move_history.append(last_moves.copy())
        side_to_move.append(color)
        previous_move_was_capture.append(capture_last_move)

        total_visits.append(1)
        move_visits.append(np.zeros((19, 19)))

        children_index.append(np.zeros((19, 19)))

        policy_outputs.append(policy_output.copy())
        black_winrates.append(np.ones((19,19)) * black_winrate)




        ############## Part 5: Choose a move to play ##############
        # In the previous parts of the code, the move was played, its details recorded, and then details were recorded in every parent position
        # Now, we need to consider a new move to examine. We have to consider all positions in the game tree and all moves in each of these positions.
        # To do this, we use a PUCT score, developed by the AlphaGo team. This assigns a score to each move in every position. The move with the highest score is played
        # There are three factors to getting a high score: Being suggested as a good move by the policy network, being deemed as having good winrate by the value network, and having not been analysed extensively previously

        # This algorithm will go through every position in the search tree looking for the move with the highest PUCT score
        # best_position is the position the PUCT algorithm decides to explore next. The algorithm eventually stops when the best_position is a position that has not been explored yet
        # best_position is always initialized to the starting position. That is, best_position = 0

        best_position = 0
        j = 0

        while True:
            correct_winrate_for_white_to_play = 1
            if side_to_move[best_position] == 2:
                correct_winrate_for_white_to_play = -1

            c_puct = 1
            value_array = black_winrates[best_position] / (1 + move_visits[best_position]) * correct_winrate_for_white_to_play
            policy_array = c_puct * policy_outputs[best_position] / (1 + move_visits[best_position]) * (total_visits[best_position])**0.5
            puct_array = value_array + policy_array

            max_puct_of_position = puct_array.max()
            best_coordinates = np.unravel_index(puct_array.argmax(), (19, 19))


            if children_index[best_position][best_coordinates] == 0: # Move is unexplored, end loop and look at it
                children_index[best_position][best_coordinates] = position_number + 1
                break

            # Else find the child of the position as indicated by the move best_coordinates
            best_position = int(children_index[best_position][best_coordinates])

            j += 1

        average_depth += j
        depths[j] = depths[j] + 1

        ############## Part 6: Create the board, color, last_moves and capture_last_move variables for the start of the loop ##############

        # Now that the move is played, we must do some tidy up. We should
        #       - play the move on the board
        #       - remove any captured stones that that move created
        #       - record whether there was a capture on the last turn - important for deciding on ko
        #       - record the move history for this move
        #       - record that this best_position is the parent of the next position to be analysed

        # This code will also set the values for board, color, last_moves and capture_last_move, which are used at the start of the loop.

        # We now know the move with the best PUCT score.
        # We should now play it on the board
        board = board_states[best_position].copy()
        color = side_to_move[best_position]

        if board[best_coordinates] != 0:
            print(board)
            print(best_coordinates)
            print(color + board[best_coordinates])
            print(len(board_states))
            print(position_number)
            a = 1/0
        board[best_coordinates] = color + board[best_coordinates]


        # Record the move history for this move. This means getting the move history for the parent, deleting the 5th last move, and then adding best_coordinates to the last moves array
        last_moves = move_history[best_position].copy()
        last_moves.pop(0)
        last_moves.append(best_coordinates)

        # Capture dead stones
        liberty_matrix = liberty_matrix[best_position].reshape(19, 19)
        group_label = group_labels[best_position].reshape(19, 19)
        capture_last_move = 0

        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        adjacent_point = [-1, -1]
        for d in range(4):
            adjacent_point[0] = best_coordinates[0] + directions[d][0]
            adjacent_point[1] = best_coordinates[1] + directions[d][1]

            if adjacent_point[0] < 0 or adjacent_point[0] > 18 or adjacent_point[1] < 0 or adjacent_point[1] > 18:
                continue

            if liberty_matrix[adjacent_point[0], adjacent_point[1]] != 1 or board[adjacent_point[0], adjacent_point[1]] != color % 2 + 1:
                continue

            group_to_remove = group_label[adjacent_point[0], adjacent_point[1]]
            board = np.where(group_label == group_to_remove, 0, board)
            board[adjacent_point[0], adjacent_point[1]] = 0
            capture_last_move = 1

        # Record the fact that this position is the parent of the next position we will look at and the move played to reach it
        parents.append([best_position, (best_coordinates)])

        position_number = position_number + 1

        ############## Part 4: Record the playout in every parent ##############

        # In order for the PUCT algorithm to work, we need to know the parents of every position in the search tree.
        # This is so that when a payout if added to a position down the search tree, it can be recorded that a playout was added to each of the parents
        # Only the immediate parent is added: We can work out the other parents later by going from parent to parent until we reach the root position


        # We now need to go through the loop, recording that we have added a playout to each of the parents of this position.
        # The parent of this position can be found in the 'parent' array.
        # Only the immediate parent is recorded: We can work out the other parents later by going from parent to parent until we reach the root position
        # Eventually, all positions must reach the original root position. This position has no parent, it was starting position for the analysis.
        # Its parent's position number is recorded as -1. The loop stops when that is reached

        parent_number = parents[position_number][0]
        move_played = parents[position_number][1]


        while(parent_number != -1):
            # We add a visit to the parent position and also add a visit to that move in that parent position
            total_visits[parent_number] += 1
            move_visits[parent_number][move_played] += 1

            # Each position also has an evalution assigned to it. This evaluation is the sum of all evaluations of the parent and its children, from the point of view of black.
            # We update the parents evaluation here.
            black_winrates[parent_number][move_played] += black_winrate

            # We then move up the line to the next parent
            move_played = parents[parent_number][1]           # Get the move that was played in the parent position to reach the child position
            parent_number = parents[parent_number][0]         # Get the parent of this position

        # The loop is now finished. It ends with a new move played on a board to be analysed.
        if np.max(policy_output) == min or move_limit <= position_number: # No more legal moves or we have reached the move limit - therefore stop the loop
            break




    # print("average_depth = ", average_depth / move_limit)
    # print(total_visits)
    # print(depths)

    return policy_outputs[0], black_winrates[0]/(1 + move_visits[0])

def generate_move_kata(board, last_moves, color, capture_last_move, move_limit):
    # These are arrays to store inputs to the model.
    board_states = []
    move_history = []
    side_to_move = []
    previous_move_was_capture = []

    # These are arrays to store the output of the model.
    policy_outputs = []
    black_winrates = []

    # These arrays don't have anything directly to do with the model. They contain important info about the PUCT algorithm.
    total_visits = []
    move_visits = []
    children_index = []

    # This array stores the parent of each position and the move that lead to the child.
    # For the previous nine arrays, they are appended to right after the model runs and we know all the necessary values to append.
    # However, this array is appended to right at the end of the loop. This is because this stores the parent of each position and the move that lead to the child.
    # However, the move is not known until after the PUCT algorithm runs - so it must be appended to at the end of the loop
    parents = [[-1, (-1, -1)]]

    position_number = 0
    ownership = np.zeros((19, 19), dtype=int)
    depths = np.zeros(100)
    average_depth = 0

    while(True):
        ############## Part 1: Run the model ##############
        # The start of the loop: At the end of the last loop, we have just placed a stone on the board. We added it to last_moves.
        # This is Part 1. We are just concerned about getting the model to make a prediction about this position. That involves setting up things correctly and then running the model
        # All the PUCT algorithm upkeep is handled in the 2nd to 5th parts of the code.

        if color == 2:
            komi = 7.5
            board_temp = (3 - board) % 3
        else:
            komi = -7.5
            board_temp = board
        global_data = np.array([0, 0, 0, 0, komi, 0, 1, capture_last_move])

        # policy_output, value_output, sb_output, ownership_output = get_kata_output(board_temp, last_moves, global_data)

        input_data, legal_moves, global_input, group_label = create_input_data_individual3(board_temp, last_moves, global_data)
        policy_output, value_output, ownership_output = model2(np.expand_dims(input_data, axis=0), np.expand_dims(global_input, axis=0), np.expand_dims(group_label, axis=0))

        #Filter illegal moves
        policy_output = policy_output.numpy().reshape(19, 19)
        policy_output = policy_output.reshape(19, 19)
        min = np.min(policy_output)
        policy_output[legal_moves == 0] = min - 100
        policy_output = softmax_2d(policy_output)
        value_output = tf.tanh(value_output)
        value_output = value_output.numpy()

        if position_number == 0:
            policy_save = policy_output

        ############## Part 2: Figure out the suggested moves and winrate of the model ##############
        # This involves two things. Figure out the suggested moves and then finding the black_winrate


        # Find the black_winrate
        if color == 1:
            black_winrate = value_output
        else:
            black_winrate = -value_output

        ############## Part 3: Record this data to the lists which keep track of the search tree ##############

        # For this, we record the suggested moves and black's winrate from Part 2
        # It is also important to keep track of the input data from part 1. If we come back to this position later, we need to know what were the position, last moves, side to move and if the last move was a capture

        board_states.append(board.copy())
        move_history.append(last_moves.copy())
        side_to_move.append(color)
        previous_move_was_capture.append(capture_last_move)

        total_visits.append(1)
        move_visits.append(np.zeros((19, 19)))

        children_index.append(np.zeros((19, 19)))

        policy_outputs.append(policy_output.copy())
        black_winrates.append(np.ones((19,19)) * black_winrate)




        ############## Part 5: Choose a move to play ##############
        # In the previous parts of the code, the move was played, its details recorded, and then details were recorded in every parent position
        # Now, we need to consider a new move to examine. We have to consider all positions in the game tree and all moves in each of these positions.
        # To do this, we use a PUCT score, developed by the AlphaGo team. This assigns a score to each move in every position. The move with the highest score is played
        # There are three factors to getting a high score: Being suggested as a good move by the policy network, being deemed as having good winrate by the value network, and having not been analysed extensively previously

        # This algorithm will go through every position in the search tree looking for the move with the highest PUCT score
        # best_position is the position the PUCT algorithm decides to explore next. The algorithm eventually stops when the best_position is a position that has not been explored yet
        # best_position is always initialized to the starting position. That is, best_position = 0

        best_position = 0
        j = 0

        while True:
            correct_winrate_for_white_to_play = 1
            if side_to_move[best_position] == 2:
                correct_winrate_for_white_to_play = -1

            c_puct = 0.4
            value_array = black_winrates[best_position] / (1 + move_visits[best_position]) * correct_winrate_for_white_to_play
            policy_array = c_puct * policy_outputs[best_position] / (1 + move_visits[best_position]) * (total_visits[best_position])**0.5
            puct_array = value_array + policy_array

            max_puct_of_position = puct_array.max()
            best_coordinates = np.unravel_index(puct_array.argmax(), (19, 19))


            if children_index[best_position][best_coordinates] == 0: # Move is unexplored, end loop and look at it
                children_index[best_position][best_coordinates] = position_number + 1
                break

            # Else find the child of the position as indicated by the move best_coordinates
            best_position = int(children_index[best_position][best_coordinates])

            j += 1

        average_depth += j
        depths[j] = depths[j] + 1

        ############## Part 6: Create the board, color, last_moves and capture_last_move variables for the start of the loop ##############

        # Now that the move is played, we must do some tidy up. We should
        #       - play the move on the board
        #       - remove any captured stones that that move created
        #       - record whether there was a capture on the last turn - important for deciding on ko
        #       - record the move history for this move
        #       - record that this best_position is the parent of the next position to be analysed

        # This code will also set the values for board, color, last_moves and capture_last_move, which are used at the start of the loop.

        # We now know the move with the best PUCT score.
        # We should now play it on the board
        board = board_states[best_position].copy()
        color = side_to_move[best_position]

        if board[best_coordinates] != 0:
            print(board)
            print(best_coordinates)
            print(color + board[best_coordinates])
            print(len(board_states))
            print(position_number)
            a = 1/0
        board[best_coordinates] = color + board[best_coordinates]


        # Record the move history for this move. This means getting the move history for the parent, deleting the 5th last move, and then adding best_coordinates to the last moves array
        last_moves = move_history[best_position].copy()
        last_moves.pop(0)
        last_moves.append(best_coordinates)

        # Capture dead stones
        input_data, legal_moves, winrates, global_input, ownership = create_input_data_individual(board, last_moves, global_data, ownership)
        board = np.where(input_data[:, :, 4] == 0, 0, board)
        board[best_coordinates] = color
        color = color % 2 + 1

        # Record the possible capture
        if np.sum(input_data[:, :, 4] == 0) > 0:
            capture_last_move = 1

        else:
            capture_last_move = 0

        # Record the fact that this position is the parent of the next position we will look at and the move played to reach it
        parents.append([best_position, (best_coordinates)])

        position_number = position_number + 1

        ############## Part 4: Record the playout in every parent ##############

        # In order for the PUCT algorithm to work, we need to know the parents of every position in the search tree.
        # This is so that when a payout if added to a position down the search tree, it can be recorded that a playout was added to each of the parents
        # Only the immediate parent is added: We can work out the other parents later by going from parent to parent until we reach the root position


        # We now need to go through the loop, recording that we have added a playout to each of the parents of this position.
        # The parent of this position can be found in the 'parent' array.
        # Only the immediate parent is recorded: We can work out the other parents later by going from parent to parent until we reach the root position
        # Eventually, all positions must reach the original root position. This position has no parent, it was starting position for the analysis.
        # Its parent's position number is recorded as -1. The loop stops when that is reached

        parent_number = parents[position_number][0]
        move_played = parents[position_number][1]


        while(parent_number != -1):
            # We add a visit to the parent position and also add a visit to that move in that parent position
            total_visits[parent_number] += 1
            move_visits[parent_number][move_played] += 1

            # Each position also has an evalution assigned to it. This evaluation is the sum of all evaluations of the parent and its children, from the point of view of black.
            # We update the parents evaluation here.
            black_winrates[parent_number][move_played] += black_winrate

            # We then move up the line to the next parent
            move_played = parents[parent_number][1]           # Get the move that was played in the parent position to reach the child position
            parent_number = parents[parent_number][0]         # Get the parent of this position

        # The loop is now finished. It ends with a new move played on a board to be analysed.
        if np.max(policy_output) == min or move_limit <= position_number: # No more legal moves or we have reached the move limit - therefore stop the loop
            break




    # print("average_depth = ", average_depth / move_limit)
    # print(total_visits)
    # print(depths)

    return policy_save, black_winrates[0]/(1 + move_visits[0]), move_visits[0]

def generate_move_kata_actual(board, last_moves, color, capture_last_move, move_limit):
    # These are arrays to store inputs to the model.
    board_states = []
    move_history = []
    side_to_move = []
    previous_move_was_capture = []

    # These are arrays to store the output of the model.
    policy_outputs = []
    black_winrates = []

    # These arrays don't have anything directly to do with the model. They contain important info about the PUCT algorithm.
    total_visits = []
    move_visits = []
    children_index = []

    # This array stores the parent of each position and the move that lead to the child.
    # For the previous nine arrays, they are appended to right after the model runs and we know all the necessary values to append.
    # However, this array is appended to right at the end of the loop. This is because this stores the parent of each position and the move that lead to the child.
    # However, the move is not known until after the PUCT algorithm runs - so it must be appended to at the end of the loop
    parents = [[-1, (-1, -1)]]

    position_number = 0
    ownership = np.zeros((19, 19), dtype=int)
    depths = np.zeros(100)
    average_depth = 0

    while(True):
        ############## Part 1: Run the model ##############
        # The start of the loop: At the end of the last loop, we have just placed a stone on the board. We added it to last_moves.
        # This is Part 1. We are just concerned about getting the model to make a prediction about this position. That involves setting up things correctly and then running the model
        # All the PUCT algorithm upkeep is handled in the 2nd to 5th parts of the code.

        if color == 2:
            komi = 7.5
            board_temp = (3 - board) % 3
        else:
            komi = -7.5
            board_temp = board
        global_data = np.array([0, 0, 0, 0, komi, 0, 1, capture_last_move])

        # policy_output, value_output, sb_output, ownership_output = get_kata_output(board_temp, last_moves, global_data)

        # input_data, legal_moves, winrates, global_input, ownership = create_input_data_individual(board_temp, last_moves, global_data, ownership)
        policy_output, value_output, _, _ = get_kata_output_place_stone(board_temp, last_moves, global_data)
        input_data, legal_moves, global_input, group_label = create_input_data_individual3(board_temp, last_moves, global_data)


        # Run the model
        # policy_output, value_output, ownership_output = model(np.expand_dims(input_data, axis=0), np.expand_dims(global_input, axis=0))
        # Filter illegal moves
        policy_output = policy_output.reshape(19, 19)
        min = np.min(policy_output)
        policy_output[legal_moves == 0] = min - 10000
        # policy_output = softmax_2d(policy_output)
        # value_output = tf.tanh(value_output)
        # value_output = value_output.numpy()

        ############## Part 2: Figure out the suggested moves and winrate of the model ##############
        # This involves two things. Figure out the suggested moves and then finding the black_winrate


        # Find the black_winrate
        if color == 1:
            black_winrate = value_output
        else:
            black_winrate = -value_output

        ############## Part 3: Record this data to the lists which keep track of the search tree ##############

        # For this, we record the suggested moves and black's winrate from Part 2
        # It is also important to keep track of the input data from part 1. If we come back to this position later, we need to know what were the position, last moves, side to move and if the last move was a capture

        board_states.append(board.copy())
        move_history.append(last_moves.copy())
        side_to_move.append(color)
        previous_move_was_capture.append(capture_last_move)

        total_visits.append(1)
        move_visits.append(np.zeros((19, 19)))

        children_index.append(np.zeros((19, 19)))

        policy_outputs.append(policy_output.copy())
        black_winrates.append(np.ones((19,19)) * black_winrate)




        ############## Part 5: Choose a move to play ##############
        # In the previous parts of the code, the move was played, its details recorded, and then details were recorded in every parent position
        # Now, we need to consider a new move to examine. We have to consider all positions in the game tree and all moves in each of these positions.
        # To do this, we use a PUCT score, developed by the AlphaGo team. This assigns a score to each move in every position. The move with the highest score is played
        # There are three factors to getting a high score: Being suggested as a good move by the policy network, being deemed as having good winrate by the value network, and having not been analysed extensively previously

        # This algorithm will go through every position in the search tree looking for the move with the highest PUCT score
        # best_position is the position the PUCT algorithm decides to explore next. The algorithm eventually stops when the best_position is a position that has not been explored yet
        # best_position is always initialized to the starting position. That is, best_position = 0

        best_position = 0
        j = 0

        while True:
            correct_winrate_for_white_to_play = 1
            if side_to_move[best_position] == 2:
                correct_winrate_for_white_to_play = -1

            c_puct = 1
            value_array = black_winrates[best_position] / (1 + move_visits[best_position]) * correct_winrate_for_white_to_play
            policy_array = c_puct * policy_outputs[best_position] / (1 + move_visits[best_position]) * (total_visits[best_position])**0.5
            puct_array = value_array + policy_array

            max_puct_of_position = puct_array.max()
            best_coordinates = np.unravel_index(puct_array.argmax(), (19, 19))


            if children_index[best_position][best_coordinates] == 0: # Move is unexplored, end loop and look at it
                children_index[best_position][best_coordinates] = position_number + 1
                break

            # Else find the child of the position as indicated by the move best_coordinates
            best_position = int(children_index[best_position][best_coordinates])

            j += 1

        average_depth += j
        depths[j] = depths[j] + 1

        ############## Part 6: Create the board, color, last_moves and capture_last_move variables for the start of the loop ##############

        # Now that the move is played, we must do some tidy up. We should
        #       - play the move on the board
        #       - remove any captured stones that that move created
        #       - record whether there was a capture on the last turn - important for deciding on ko
        #       - record the move history for this move
        #       - record that this best_position is the parent of the next position to be analysed

        # This code will also set the values for board, color, last_moves and capture_last_move, which are used at the start of the loop.

        # We now know the move with the best PUCT score.
        # We should now play it on the board
        board = board_states[best_position].copy()
        color = side_to_move[best_position]

        if board[best_coordinates] != 0:
            print(board)
            print(best_coordinates)
            print(color + board[best_coordinates])
            print(len(board_states))
            print(position_number)
            a = 1/0
        board[best_coordinates] = color + board[best_coordinates]


        # Record the move history for this move. This means getting the move history for the parent, deleting the 5th last move, and then adding best_coordinates to the last moves array
        last_moves = move_history[best_position].copy()
        last_moves.pop(0)
        last_moves.append(best_coordinates)

        # Capture dead stones
        input_data, legal_moves, global_input, group_label = create_input_data_individual3(board, last_moves, global_data)
        board = np.where(input_data[:, :, 4] == 0, 0, board)
        board[best_coordinates] = color
        color = color % 2 + 1

        # Record the possible capture
        if np.sum(input_data[:, :, 4] == 0) > 0:
            capture_last_move = 1

        else:
            capture_last_move = 0

        # Record the fact that this position is the parent of the next position we will look at and the move played to reach it
        parents.append([best_position, (best_coordinates)])

        position_number = position_number + 1

        ############## Part 4: Record the playout in every parent ##############

        # In order for the PUCT algorithm to work, we need to know the parents of every position in the search tree.
        # This is so that when a payout if added to a position down the search tree, it can be recorded that a playout was added to each of the parents
        # Only the immediate parent is added: We can work out the other parents later by going from parent to parent until we reach the root position


        # We now need to go through the loop, recording that we have added a playout to each of the parents of this position.
        # The parent of this position can be found in the 'parent' array.
        # Only the immediate parent is recorded: We can work out the other parents later by going from parent to parent until we reach the root position
        # Eventually, all positions must reach the original root position. This position has no parent, it was starting position for the analysis.
        # Its parent's position number is recorded as -1. The loop stops when that is reached

        parent_number = parents[position_number][0]
        move_played = parents[position_number][1]


        while(parent_number != -1):
            # We add a visit to the parent position and also add a visit to that move in that parent position
            total_visits[parent_number] += 1
            move_visits[parent_number][move_played] += 1

            # Each position also has an evalution assigned to it. This evaluation is the sum of all evaluations of the parent and its children, from the point of view of black.
            # We update the parents evaluation here.
            black_winrates[parent_number][move_played] += black_winrate

            # We then move up the line to the next parent
            move_played = parents[parent_number][1]           # Get the move that was played in the parent position to reach the child position
            parent_number = parents[parent_number][0]         # Get the parent of this position

        # The loop is now finished. It ends with a new move played on a board to be analysed.
        if np.max(policy_output) == min or move_limit <= position_number: # No more legal moves or we have reached the move limit - therefore stop the loop
            break




    # print("average_depth = ", average_depth / move_limit)
    # print(total_visits)
    # print(depths)

    return move_visits[0], black_winrates[0]/(1 + move_visits[0]), policy_outputs[0]

class generate_move():
    def __init__(self, board, last_moves, color, capture_last_move, num_search_threads):
        # Store the init arguments
        self.board = [board] * num_search_threads
        self.last_moves = [last_moves] * num_search_threads
        self.color = [color] * num_search_threads
        self.capture_last_move = [capture_last_move] * num_search_threads
        self.group_labels = []
        self.black_winrate_batch = np.zeros((num_search_threads))
        self.num_search_threads = num_search_threads

        # These are arrays to store inputs to the model.
        self.board_states = []
        self.move_history = []
        self.side_to_move = []
        self.previous_move_was_capture = []

        # These are arrays to store the output of the model.
        self.policy_outputs = []
        self.black_winrates = []
        self.liberty_matrix = []


        # These arrays don't have anything directly to do with the model. They contain important info about the PUCT algorithm.
        self.total_visits = []
        self.move_visits = []
        self.children_index = []

        # This array stores the parent of each position and the move that lead to the child.
        # For the previous nine arrays, they are appended to right after the model runs and we know all the necessary values to append.
        # However, this array is appended to right at the end of the loop. This is because this stores the parent of each position and the move that lead to the child.
        # However, the move is not known until after the PUCT algorithm runs - so it must be appended to at the end of the loop
        self.parents = [[-1, (-1, -1)]]

        self.position_number = 0
        self.depths = np.zeros(100)
        self.average_depth = 0

    def run(self, move_limit):

        batch_size = 1
        input_data = np.zeros((self.num_search_threads, 19, 19, 23), dtype=float)
        legal_moves = np.zeros((self.num_search_threads, 19, 19), dtype=int)
        global_input = np.zeros((self.num_search_threads, 5), dtype=float)
        group_labels = np.zeros((self.num_search_threads, 19, 19), dtype=int)

        while(True):
            ############## Part 1: Run the model ##############
            # The start of the loop: At the end of the last loop, we have just placed a stone on the board. We added it to last_moves.
            # This is Part 1. We are just concerned about getting the model to make a prediction about this position. That involves setting up things correctly and then running the model
            # All the PUCT algorithm upkeep is handled in the 2nd to 5th parts of the code.

            for pos in range(batch_size):
                if self.color[pos] == 2:
                    komi = 7.5
                else:
                    komi = -7.5

                global_data = np.array([0, 0, 0, 0, komi, 0, self.color[pos], self.capture_last_move[pos]])
                input_data[pos], legal_moves[pos], global_input[pos], group_labels[pos] = create_input_data_individual3(self.board[pos], self.last_moves[pos], global_data)

                self.group_labels.append(group_labels[pos].copy())
                self.liberty_matrix.append(input_data[pos, :, :, 5].copy())

            # Run the model
            policy_output, value_output, _ = model(input_data, global_input, group_labels)

            policy_output = policy_output.numpy().reshape(self.num_search_threads, 19, 19)
            policy_output[legal_moves == 0] -= 1000
            for pos in range(batch_size):
                policy_output[pos] = softmax_2d(policy_output[pos])
            policy_output[legal_moves == 0] = -1e-10
            value_output = tf.tanh(value_output)
            value_output = value_output.numpy().reshape(self.num_search_threads)

            ############## Part 2: Figure out the suggested moves and winrate of the model ##############
            # This involves two things. Figure out the suggested moves and then finding the black_winrate

            # Find the black_winrate
            for pos in range(batch_size):
                if self.color[pos] == 1:
                    self.black_winrate_batch[pos] = value_output[pos]
                else:
                    self.black_winrate_batch[pos] = -value_output[pos]

            ############## Part 3: Record this data to the lists which keep track of the search tree ##############

            # For this, we record the suggested moves and black's winrate from Part 2
            # It is also important to keep track of the input data from part 1. If we come back to this position later, we need to know what were the position, last moves, side to move and if the last move was a capture

            for pos in range(batch_size):
                last_move = self.last_moves[pos][-1]
                stone = self.board[pos][last_move[0], last_move[1]]
                if stone < 1 or stone > 2:
                    print("Last move not alligning")
                    print("last_move = ", last_move)
                    print("stone = ", stone)
                    self.board[pos][last_move[0], last_move[1]] += 3
                    print(self.board[pos])
                    x = 1/0
                self.board_states.append(self.board[pos].copy())
                self.move_history.append(self.last_moves[pos].copy())
                self.side_to_move.append(self.color[pos])
                self.previous_move_was_capture.append(self.capture_last_move[pos])

                self.total_visits.append(1)
                self.move_visits.append(np.zeros((19, 19)))

                self.children_index.append(np.zeros((19, 19)))

                self.policy_outputs.append(policy_output[pos].copy())
                self.black_winrates.append(np.ones((19,19)) * self.black_winrate_batch[pos])

                ############## Part 4: Record the playout in every parent ##############

                # In order for the PUCT algorithm to work, we need to know the parents of every position in the search tree.
                # This is so that when a payout if added to a position down the search tree, it can be recorded that a playout was added to each of the parents
                # Only the immediate parent is added: We can work out the other parents later by going from parent to parent until we reach the root position


                # We now need to go through the loop, recording that we have added a playout to each of the parents of this position.
                # The parent of this position can be found in the 'parent' array.
                # Only the immediate parent is recorded: We can work out the other parents later by going from parent to parent until we reach the root position
                # Eventually, all positions must reach the original root position. This position has no parent, it was starting position for the analysis.
                # Its parent's position number is recorded as -1. The loop stops when that is reached

                parent_number = self.parents[self.position_number + 1 - batch_size + pos][0]
                move_played = self.parents[self.position_number + 1 - batch_size + pos][1]


                while(parent_number != -1):
                    # We add a visit to the parent position and also add a visit to that move in that parent position
                    self.total_visits[parent_number] += 1
                    self.move_visits[parent_number][move_played] += 1

                    # Each position also has an evalution assigned to it. This evaluation is the sum of all evaluations of the parent and its children, from the point of view of black.
                    # We update the parents evaluation here.
                    self.black_winrates[parent_number][move_played] += self.black_winrate_batch[pos]

                    # We then move up the line to the next parent
                    move_played = self.parents[parent_number][1]           # Get the move that was played in the parent position to reach the child position
                    parent_number = self.parents[parent_number][0]         # Get the parent of this position

            # The loop is now finished. It ends with a new move played on a board to be analysed.
            if np.max(policy_output) == min or move_limit <= self.position_number: # No more legal moves or we have reached the move limit - therefore stop the loop
                break

            ############## Part 5: Choose a move to play ##############
            # In the previous parts of the code, the move was played, its details recorded, and then details were recorded in every parent position
            # Now, we need to consider a new move to examine. We have to consider all positions in the game tree and all moves in each of these positions.
            # To do this, we use a PUCT score, developed by the AlphaGo team. This assigns a score to each move in every position. The move with the highest score is played
            # There are three factors to getting a high score: Being suggested as a good move by the policy network, being deemed as having good winrate by the value network, and having not been analysed extensively previously

            # This algorithm will go through every position in the search tree looking for the move with the highest PUCT score
            # best_position is the position the PUCT algorithm decides to explore next. The algorithm eventually stops when the best_position is a position that has not been explored yet
            # best_position is always initialized to the starting position. That is, best_position = 0
            batch_size = self.num_search_threads
            end_position = self.position_number + 1
            for pos in range(batch_size):
                best_position = 0
                j = 0
                temp_move_visits = np.zeros((self.position_number + batch_size, 19, 19))
                adjust = 0
                loop_counter =  0

                # temp_total_visits = np.zeros((self.position_number))
                #
                # print("These numbers should be the same: ", end_position, " = ", len(self.move_visits), flush = True)

                while True:
                    if best_position >= end_position: # We input multiple positions to the model in each batch. This means that we are attempting to enter the same position twice!
                        # The solution is to start the PUCT algorithm again, but this time with adjust = 1.
                        # This will discourage it from looking at the same line again by boosting the visits for already explored lines
                        best_position = 0
                        j = 0
                        adjust = 1
                        loop_counter += 1
                        continue

                    correct_winrate_for_white_to_play = 1
                    if self.side_to_move[best_position] == 2:
                        correct_winrate_for_white_to_play = -1

                    c_puct = 0.5
                    value_array = self.black_winrates[best_position] / (1 + self.move_visits[best_position]) * correct_winrate_for_white_to_play
                    policy_array = c_puct * self.policy_outputs[best_position] / (1 + self.move_visits[best_position] + adjust * temp_move_visits[best_position]) * (self.total_visits[best_position])**0.5
                    puct_array = value_array + policy_array * 1.1 ** loop_counter

                    puct_array_mask_illegals = np.where(self.policy_outputs[best_position] < 0, -1000, puct_array)

                    max_puct_of_position = puct_array_mask_illegals.max()
                    best_coordinates = np.unravel_index(puct_array_mask_illegals.argmax(), (19, 19))
                    temp_move_visits[best_position][best_coordinates] += 1.1 ** loop_counter


                    if np.sum(temp_move_visits[best_position]) > 1e15:
                        print(temp_move_visits[best_position])

                        black_positions = np.argwhere(self.board[pos] == 1)
                        white_positions = np.argwhere(self.board[pos] == 2)
                        def plot_heatmap(data, ax, title):
                            cax = ax.matshow(data, cmap='viridis')
                            ax.scatter(black_positions[:, 1], black_positions[:, 0], c='black', marker='o', s=100, label="Black Stones")
                            ax.scatter(white_positions[:, 1], white_positions[:, 0], c='white', marker='o', s=100, label="White Stones", edgecolor='black')
                            ax.set_title(title)
                            plt.colorbar(cax, ax=ax)

                        fig, axs = plt.subplots(2, 2, figsize=(8,8))

                        plot_heatmap(self.policy_outputs[0] , axs[0, 0], "Original policy")
                        plot_heatmap(policy_array, axs[0, 1], "Mod Policy")
                        plot_heatmap(self.move_visits[best_position] + adjust * temp_move_visits[best_position], axs[1, 0], "Mod Value")
                        plot_heatmap(puct_array, axs[1, 1], "Puct Array")
                        plt.show()

                        a = 1/0



                    if self.children_index[best_position][best_coordinates] == 0: # Move is unexplored, end loop and look at it
                        temp_move_visits = np.zeros((self.position_number + batch_size, 19, 19))
                        loop_counter = 0
                        self.children_index[best_position][best_coordinates] = self.position_number + 1
                        break



                    # Else find the child of the position as indicated by the move best_coordinates
                    best_position = int(self.children_index[best_position][best_coordinates])

                    j += 1

                self.average_depth += j
                self.depths[j] = self.depths[j] + 1

                ############## Part 6: Create the board, color, last_moves and capture_last_move variables for the start of the loop ##############

                # Now that the move is played, we must do some tidy up. We should
                #       - play the move on the board
                #       - remove any captured stones that that move created
                #       - record whether there was a capture on the last turn - important for deciding on ko
                #       - record the move history for this move
                #       - record that this best_position is the parent of the next position to be analysed

                # This code will also set the values for board, color, last_moves and capture_last_move, which are used at the start of the loop.

                # We now know the move with the best PUCT score.
                # We should now play it on the board
                debug = False
                if best_coordinates[0] == 17 and best_coordinates[1] == 11:
                    debug = True

                self.board[pos] = self.board_states[best_position].copy()
                self.color[pos] = self.side_to_move[best_position]

                if self.board[pos][best_coordinates] != 0:
                    print(self.board[pos])
                    print(best_coordinates)
                    print(self.color[pos] + self.board[pos][best_coordinates])
                    print(len(self.board_states))
                    print(self.position_number)
                    black_positions = np.argwhere(self.board[pos] == 1)
                    white_positions = np.argwhere(self.board[pos] == 2)
                    def plot_heatmap(data, ax, title):
                        cax = ax.matshow(data, cmap='viridis')
                        ax.scatter(black_positions[:, 1], black_positions[:, 0], c='black', marker='o', s=100, label="Black Stones")
                        ax.scatter(white_positions[:, 1], white_positions[:, 0], c='white', marker='o', s=100, label="White Stones", edgecolor='black')
                        ax.set_title(title)
                        plt.colorbar(cax, ax=ax)

                    fig, axs = plt.subplots(2, 2, figsize=(8,8))

                    plot_heatmap(self.policy_outputs[0] , axs[0, 0], "Original policy")
                    plot_heatmap(policy_array, axs[0, 1], "Mod Policy")
                    plot_heatmap(self.move_visits[best_position] + adjust * temp_move_visits[best_position], axs[1, 0], "Mod Value")
                    plot_heatmap(puct_array, axs[1, 1], "Puct Array")
                    plt.tight_layout()
                    plt.show()
                    b = 3/0

                self.board[pos][best_coordinates] = self.color[pos] + self.board[pos][best_coordinates]

                # black_positions = np.argwhere(self.board[pos] == 1)
                # white_positions = np.argwhere(self.board[pos] == 2)
                # def plot_heatmap(data, ax, title):
                #     cax = ax.matshow(data, cmap='viridis')
                #     ax.scatter(black_positions[:, 1], black_positions[:, 0], c='black', marker='o', s=100, label="Black Stones")
                #     ax.scatter(white_positions[:, 1], white_positions[:, 0], c='white', marker='o', s=100, label="White Stones", edgecolor='black')
                #     ax.set_title(title)
                #     plt.colorbar(cax, ax=ax)
                #
                # fig, axs = plt.subplots(1, 2, figsize=(10,5))
                #
                # plot_heatmap(puct_array, axs[0], "Puct Array")
                # plot_heatmap(self.policy_outputs[best_position], axs[1], "Policy Array = " + str(np.sum(self.policy_outputs[best_position])))
                # plt.show()


                # Record the move history for this move. This means getting the move history for the parent, deleting the 5th last move, and then adding best_coordinates to the last moves array
                self.last_moves[pos] = self.move_history[best_position].copy()
                self.last_moves[pos].pop(0)
                self.last_moves[pos].append(best_coordinates)

                # Capture dead stones
                liberty_matrix = self.liberty_matrix[best_position].copy().reshape(19, 19)
                group_label = self.group_labels[best_position].copy().reshape(19, 19)
                self.capture_last_move[pos] = 0

                directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                adjacent_point = [-1, -1]
                for d in range(4):
                    adjacent_point[0] = best_coordinates[0] + directions[d][0]
                    adjacent_point[1] = best_coordinates[1] + directions[d][1]

                    if adjacent_point[0] < 0 or adjacent_point[0] > 18 or adjacent_point[1] < 0 or adjacent_point[1] > 18:
                        continue

                    if liberty_matrix[adjacent_point[0], adjacent_point[1]] != 1 or self.board[pos][adjacent_point[0], adjacent_point[1]] != self.color[pos] % 2 + 1:
                        continue

                    group_to_remove = group_label[adjacent_point[0], adjacent_point[1]]
                    if group_to_remove != 0:
                        self.board[pos] = np.where(group_label == group_to_remove, 0, self.board[pos])
                    else:
                        self.board[pos][adjacent_point[0], adjacent_point[1]] = 0

                    self.capture_last_move[pos] = 1

                self.color[pos] = self.color[pos] % 2 + 1

                # Record the fact that this position is the parent of the next position we will look at and the move played to reach it
                self.parents.append([best_position, (best_coordinates)])
                self.position_number = self.position_number + 1


        # for i in range(100):
        #     if True or self.total_visits[i] < 1000 or np.sum(self.board_states[0]) > 150:
        #         continue
        #
        #     black_positions = np.argwhere(self.board_states[i] == 1)
        #     white_positions = np.argwhere(self.board_states[i] == 2)
        #
        #     def plot_heatmap(data, ax, title):
        #         cax = ax.matshow(data, cmap='viridis')
        #         ax.scatter(black_positions[:, 1], black_positions[:, 0], c='black', marker='o', s=100, label="Black Stones")
        #         ax.scatter(white_positions[:, 1], white_positions[:, 0], c='white', marker='o', s=100, label="White Stones", edgecolor='black')
        #         ax.set_title(title)
        #         plt.colorbar(cax, ax=ax)
        #
        #     fig, axs = plt.subplots(1, 3, figsize=(14,6))
        #
        #     plot_heatmap(self.policy_outputs[i] , axs[0], "Original policy")
        #     plot_heatmap(self.black_winrates[i]/(1 + self.move_visits[i]), axs[1], "Value")
        #     plot_heatmap(self.move_visits[i], axs[2], "Puct")
        #     # plot_heatmap(puct_array, axs[1, 1], "Drive")
        #     plt.tight_layout()
        #     plt.show()


        return self.move_visits[0], self.black_winrates[0]/(1 + self.move_visits[0]), self.policy_outputs[0]

class generate_move_with_kata():
    def __init__(self, board, last_moves, color, capture_last_move, num_search_threads):
        # Store the init arguments
        self.board = [board] * num_search_threads
        self.last_moves = [last_moves] * num_search_threads
        self.color = [color] * num_search_threads
        self.capture_last_move = [capture_last_move] * num_search_threads
        self.group_labels = []
        self.black_winrate_batch = np.zeros((num_search_threads))
        self.num_search_threads = num_search_threads

        # These are arrays to store inputs to the model.
        self.board_states = []
        self.move_history = []
        self.side_to_move = []
        self.previous_move_was_capture = []

        # These are arrays to store the output of the model.
        self.policy_outputs = []
        self.black_winrates = []
        self.liberty_matrix = []


        # These arrays don't have anything directly to do with the model. They contain important info about the PUCT algorithm.
        self.total_visits = []
        self.move_visits = []
        self.children_index = []

        # This array stores the parent of each position and the move that lead to the child.
        # For the previous nine arrays, they are appended to right after the model runs and we know all the necessary values to append.
        # However, this array is appended to right at the end of the loop. This is because this stores the parent of each position and the move that lead to the child.
        # However, the move is not known until after the PUCT algorithm runs - so it must be appended to at the end of the loop
        self.parents = [[-1, (-1, -1)]]

        self.position_number = 0
        self.depths = np.zeros(100)
        self.average_depth = 0

    def run(self, move_limit):

        batch_size = 1
        input_data = np.zeros((self.num_search_threads, 19, 19, 23), dtype=float)
        legal_moves = np.zeros((self.num_search_threads, 19, 19), dtype=int)
        global_input = np.zeros((self.num_search_threads, 5), dtype=float)
        group_labels = np.zeros((self.num_search_threads, 19, 19), dtype=int)
        policy_output = np.zeros((self.num_search_threads, 19, 19), dtype=float)
        value_output = np.zeros((self.num_search_threads), dtype=float)

        while(True):
            ############## Part 1: Run the model ##############
            # The start of the loop: At the end of the last loop, we have just placed a stone on the board. We added it to last_moves.
            # This is Part 1. We are just concerned about getting the model to make a prediction about this position. That involves setting up things correctly and then running the model
            # All the PUCT algorithm upkeep is handled in the 2nd to 5th parts of the code.
            for pos in range(batch_size):

                if self.color[pos] == 2:
                    komi = 7.5
                    board_temp = (3 - self.board[pos]) % 3
                else:
                    komi = -7.5
                    board_temp = self.board[pos]

                global_data = np.array([0, 0, 0, 0, komi, 0, self.color[pos], self.capture_last_move[pos]])
                if len(self.last_moves[pos]) < 5:
                    use_instead = self.last_moves[pos].copy()
                    use_instead.insert(0, (-1, -1))
                    input_data[pos], legal_moves[pos], global_input[pos], group_labels[pos] = create_input_data_individual3(self.board[pos], use_instead, global_data)
                else:
                    input_data[pos], legal_moves[pos], global_input[pos], group_labels[pos] = create_input_data_individual3(self.board[pos], self.last_moves[pos][-5:], global_data)

                self.group_labels.append(group_labels[pos].copy())
                self.liberty_matrix.append(input_data[pos, :, :, 5].copy())



                policy_output[pos], value_output[pos], _, _ = get_kata_output(-1, self.last_moves[pos], -1)
                policy_output[pos][legal_moves[pos] == 0] -= 10

            # Run the model
            # policy_output, value_output, _ = model(input_data, global_input, group_labels)

            policy_output = policy_output.reshape(self.num_search_threads, 19, 19)

            # for pos in range(batch_size):
            #     policy_output[pos] = softmax_2d(policy_output[pos])
            # policy_output[legal_moves == 0] = -1e-10
            # value_output = tf.tanh(value_output)
            value_output = value_output.reshape(self.num_search_threads)

            ############## Part 2: Figure out the suggested moves and winrate of the model ##############
            # This involves two things. Figure out the suggested moves and then finding the black_winrate

            # Find the black_winrate

            for pos in range(batch_size):
                self.black_winrate_batch[pos] = value_output[pos]
                # if self.color[pos] == 1:
                #     self.black_winrate_batch[pos] = value_output[pos]
                # else:
                #     self.black_winrate_batch[pos] = -value_output[pos]



            ############## Part 3: Record this data to the lists which keep track of the search tree ##############

            # For this, we record the suggested moves and black's winrate from Part 2
            # It is also important to keep track of the input data from part 1. If we come back to this position later, we need to know what were the position, last moves, side to move and if the last move was a capture

            for pos in range(batch_size):
                last_move = self.last_moves[pos][-1]
                stone = self.board[pos][last_move[0], last_move[1]]
                if stone < 1 or stone > 2:
                    print("Last move not alligning")
                    print("last_move = ", last_move)
                    print("stone = ", stone)
                    self.board[pos][last_move[0], last_move[1]] += 3
                    print(self.board[pos])
                    x = 1/0
                self.board_states.append(self.board[pos].copy())
                self.move_history.append(self.last_moves[pos].copy())
                self.side_to_move.append(self.color[pos])
                self.previous_move_was_capture.append(self.capture_last_move[pos])

                self.total_visits.append(1)
                self.move_visits.append(np.zeros((19, 19)))

                self.children_index.append(np.zeros((19, 19)))

                self.policy_outputs.append(policy_output[pos].copy())
                self.black_winrates.append(np.ones((19,19)) * self.black_winrate_batch[pos])

                ############## Part 4: Record the playout in every parent ##############

                # In order for the PUCT algorithm to work, we need to know the parents of every position in the search tree.
                # This is so that when a payout if added to a position down the search tree, it can be recorded that a playout was added to each of the parents
                # Only the immediate parent is added: We can work out the other parents later by going from parent to parent until we reach the root position


                # We now need to go through the loop, recording that we have added a playout to each of the parents of this position.
                # The parent of this position can be found in the 'parent' array.
                # Only the immediate parent is recorded: We can work out the other parents later by going from parent to parent until we reach the root position
                # Eventually, all positions must reach the original root position. This position has no parent, it was starting position for the analysis.
                # Its parent's position number is recorded as -1. The loop stops when that is reached

                parent_number = self.parents[self.position_number + 1 - batch_size + pos][0]
                move_played = self.parents[self.position_number + 1 - batch_size + pos][1]


                while(parent_number != -1):
                    # We add a visit to the parent position and also add a visit to that move in that parent position
                    self.total_visits[parent_number] += 1
                    self.move_visits[parent_number][move_played] += 1

                    # Each position also has an evalution assigned to it. This evaluation is the sum of all evaluations of the parent and its children, from the point of view of black.
                    # We update the parents evaluation here.
                    self.black_winrates[parent_number][move_played] += self.black_winrate_batch[pos]

                    # We then move up the line to the next parent
                    move_played = self.parents[parent_number][1]           # Get the move that was played in the parent position to reach the child position
                    parent_number = self.parents[parent_number][0]         # Get the parent of this position

            # The loop is now finished. It ends with a new move played on a board to be analysed.
            if np.max(policy_output) == min or move_limit <= self.position_number: # No more legal moves or we have reached the move limit - therefore stop the loop
                break

            ############## Part 5: Choose a move to play ##############
            # In the previous parts of the code, the move was played, its details recorded, and then details were recorded in every parent position
            # Now, we need to consider a new move to examine. We have to consider all positions in the game tree and all moves in each of these positions.
            # To do this, we use a PUCT score, developed by the AlphaGo team. This assigns a score to each move in every position. The move with the highest score is played
            # There are three factors to getting a high score: Being suggested as a good move by the policy network, being deemed as having good winrate by the value network, and having not been analysed extensively previously

            # This algorithm will go through every position in the search tree looking for the move with the highest PUCT score
            # best_position is the position the PUCT algorithm decides to explore next. The algorithm eventually stops when the best_position is a position that has not been explored yet
            # best_position is always initialized to the starting position. That is, best_position = 0
            batch_size = self.num_search_threads
            end_position = self.position_number + 1
            for pos in range(batch_size):
                best_position = 0
                j = 0
                temp_move_visits = np.zeros((self.position_number + batch_size, 19, 19))
                adjust = 0
                loop_counter =  0

                # temp_total_visits = np.zeros((self.position_number))
                #
                # print("These numbers should be the same: ", end_position, " = ", len(self.move_visits), flush = True)

                while True:
                    if best_position >= end_position: # We input multiple positions to the model in each batch. This means that we are attempting to enter the same position twice!
                        # The solution is to start the PUCT algorithm again, but this time with adjust = 1.
                        # This will discourage it from looking at the same line again by boosting the visits for already explored lines
                        best_position = 0
                        j = 0
                        adjust = 1
                        loop_counter += 1
                        continue

                    correct_winrate_for_white_to_play = 1
                    if self.side_to_move[best_position] == 2:
                        correct_winrate_for_white_to_play = -1

                    c_puct = 1.2
                    value_array = self.black_winrates[best_position] / (1 + self.move_visits[best_position]) * correct_winrate_for_white_to_play
                    policy_array = c_puct * self.policy_outputs[best_position] / (1 + self.move_visits[best_position] + adjust * temp_move_visits[best_position]) * (self.total_visits[best_position])**0.5
                    puct_array = value_array + policy_array * 1.1 ** loop_counter

                    puct_array_mask_illegals = np.where(self.policy_outputs[best_position] < 0, -1000, puct_array)

                    max_puct_of_position = puct_array_mask_illegals.max()
                    best_coordinates = np.unravel_index(puct_array_mask_illegals.argmax(), (19, 19))
                    temp_move_visits[best_position][best_coordinates] += 1.1 ** loop_counter


                    if np.sum(temp_move_visits[best_position]) > 1e15:
                        print(temp_move_visits[best_position])

                        black_positions = np.argwhere(self.board[pos] == 1)
                        white_positions = np.argwhere(self.board[pos] == 2)
                        def plot_heatmap(data, ax, title):
                            cax = ax.matshow(data, cmap='viridis')
                            ax.scatter(black_positions[:, 1], black_positions[:, 0], c='black', marker='o', s=100, label="Black Stones")
                            ax.scatter(white_positions[:, 1], white_positions[:, 0], c='white', marker='o', s=100, label="White Stones", edgecolor='black')
                            ax.set_title(title)
                            plt.colorbar(cax, ax=ax)

                        fig, axs = plt.subplots(2, 2, figsize=(8,8))

                        plot_heatmap(self.policy_outputs[best_position] , axs[0, 0], "Original policy")
                        plot_heatmap(policy_array, axs[0, 1], "Mod Policy")
                        plot_heatmap(self.move_visits[best_position] + adjust * temp_move_visits[best_position], axs[1, 0], "Mod Value")
                        plot_heatmap(puct_array, axs[1, 1], "Puct Array")
                        plt.show()

                        a = 1/0



                    if self.children_index[best_position][best_coordinates] == 0: # Move is unexplored, end loop and look at it
                        temp_move_visits = np.zeros((self.position_number + batch_size, 19, 19))
                        loop_counter = 0
                        self.children_index[best_position][best_coordinates] = self.position_number + 1
                        break



                    # Else find the child of the position as indicated by the move best_coordinates
                    best_position = int(self.children_index[best_position][best_coordinates])

                    j += 1

                self.average_depth += j
                self.depths[j] = self.depths[j] + 1

                ############## Part 6: Create the board, color, last_moves and capture_last_move variables for the start of the loop ##############

                # Now that the move is played, we must do some tidy up. We should
                #       - play the move on the board
                #       - remove any captured stones that that move created
                #       - record whether there was a capture on the last turn - important for deciding on ko
                #       - record the move history for this move
                #       - record that this best_position is the parent of the next position to be analysed

                # This code will also set the values for board, color, last_moves and capture_last_move, which are used at the start of the loop.

                # We now know the move with the best PUCT score.
                # We should now play it on the board
                debug = False
                if best_coordinates[0] == 17 and best_coordinates[1] == 11:
                    debug = True

                self.board[pos] = self.board_states[best_position].copy()
                self.color[pos] = self.side_to_move[best_position]

                if self.board[pos][best_coordinates] != 0:
                    print(self.board[pos])
                    print(best_coordinates)
                    print(self.color[pos] + self.board[pos][best_coordinates])
                    print(len(self.board_states))
                    print(self.position_number)
                    black_positions = np.argwhere(self.board[pos] == 1)
                    white_positions = np.argwhere(self.board[pos] == 2)
                    def plot_heatmap(data, ax, title):
                        cax = ax.matshow(data, cmap='viridis')
                        ax.scatter(black_positions[:, 1], black_positions[:, 0], c='black', marker='o', s=100, label="Black Stones")
                        ax.scatter(white_positions[:, 1], white_positions[:, 0], c='white', marker='o', s=100, label="White Stones", edgecolor='black')
                        ax.set_title(title)
                        plt.colorbar(cax, ax=ax)

                    fig, axs = plt.subplots(2, 2, figsize=(8,8))

                    plot_heatmap(self.policy_outputs[0] , axs[0, 0], "Original policy")
                    plot_heatmap(policy_array, axs[0, 1], "Mod Policy")
                    plot_heatmap(self.move_visits[best_position] + adjust * temp_move_visits[best_position], axs[1, 0], "Mod Value")
                    plot_heatmap(puct_array, axs[1, 1], "Puct Array")
                    plt.tight_layout()
                    plt.show()
                    b = 3/0

                self.board[pos][best_coordinates] = self.color[pos] + self.board[pos][best_coordinates]

                # black_positions = np.argwhere(self.board[pos] == 1)
                # white_positions = np.argwhere(self.board[pos] == 2)
                # def plot_heatmap(data, ax, title):
                #     cax = ax.matshow(data, cmap='viridis')
                #     ax.scatter(black_positions[:, 1], black_positions[:, 0], c='black', marker='o', s=100, label="Black Stones")
                #     ax.scatter(white_positions[:, 1], white_positions[:, 0], c='white', marker='o', s=100, label="White Stones", edgecolor='black')
                #     ax.set_title(title)
                #     plt.colorbar(cax, ax=ax)
                #
                # fig, axs = plt.subplots(1, 2, figsize=(10,5))
                #
                # plot_heatmap(puct_array, axs[0], "Puct Array")
                # plot_heatmap(self.policy_outputs[best_position], axs[1], "Policy Array = " + str(np.sum(self.policy_outputs[best_position])))
                # plt.show()


                # Record the move history for this move. This means getting the move history for the parent, deleting the 5th last move, and then adding best_coordinates to the last moves array
                self.last_moves[pos] = self.move_history[best_position].copy()
                # self.last_moves[pos].pop(0)
                self.last_moves[pos].append(best_coordinates)

                # Capture dead stones
                liberty_matrix = self.liberty_matrix[best_position].copy().reshape(19, 19)
                group_label = self.group_labels[best_position].copy().reshape(19, 19)
                self.capture_last_move[pos] = 0

                directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                adjacent_point = [-1, -1]
                for d in range(4):
                    adjacent_point[0] = best_coordinates[0] + directions[d][0]
                    adjacent_point[1] = best_coordinates[1] + directions[d][1]

                    if adjacent_point[0] < 0 or adjacent_point[0] > 18 or adjacent_point[1] < 0 or adjacent_point[1] > 18:
                        continue

                    if liberty_matrix[adjacent_point[0], adjacent_point[1]] != 1 or self.board[pos][adjacent_point[0], adjacent_point[1]] != self.color[pos] % 2 + 1:
                        continue

                    group_to_remove = group_label[adjacent_point[0], adjacent_point[1]]
                    if group_to_remove != 0:
                        self.board[pos] = np.where(group_label == group_to_remove, 0, self.board[pos])
                    else:
                        self.board[pos][adjacent_point[0], adjacent_point[1]] = 0

                    self.capture_last_move[pos] = 1

                self.color[pos] = self.color[pos] % 2 + 1

                # Record the fact that this position is the parent of the next position we will look at and the move played to reach it
                self.parents.append([best_position, (best_coordinates)])
                self.position_number = self.position_number + 1



        # for i in range(100):
        #     print("using if self.total_visits[i] < 5:", flush = True)
        #     if self.total_visits[i] < 5:
        #         continue
        #
        #     black_positions = np.argwhere(self.board_states[i] == 1)
        #     white_positions = np.argwhere(self.board_states[i] == 2)
        #
        #     def plot_heatmap(data, ax, title):
        #         cax = ax.matshow(data, cmap='viridis')
        #         ax.scatter(black_positions[:, 1], black_positions[:, 0], c='black', marker='o', s=100, label="Black Stones")
        #         ax.scatter(white_positions[:, 1], white_positions[:, 0], c='white', marker='o', s=100, label="White Stones", edgecolor='black')
        #         ax.set_title(title)
        #         plt.colorbar(cax, ax=ax)
        #
        #     fig, axs = plt.subplots(1, 3, figsize=(14,6))
        #
        #     plot_heatmap(self.policy_outputs[i] , axs[0], "Original policy")
        #     plot_heatmap(self.black_winrates[i]/(1 + self.move_visits[i]), axs[1], "Value")
        #     plot_heatmap(self.move_visits[i], axs[2], "Puct")
        #     # plot_heatmap(puct_array, axs[1, 1], "Drive")
        #     plt.tight_layout()
        #     plt.show()


        return self.move_visits[0], self.black_winrates[0]/(1 + self.move_visits[0]), self.policy_outputs[0]




def get_captures(board, row, col):
    opp_color = board[row][col] % 2 + 1
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    capture_last_move = 0

    for dr, dc in directions:
        nr, nc = row + dr, col + dc

        if nr < 0 or nr >= 19 or nc < 0 or nc >= 19:
            continue

        if board[nr][nc] == opp_color:
            libs = get_liberties(board, nr, nc)

            if libs == 0:
                capture_last_move = 1

    return capture_last_move

def get_liberties(board, row, col):
    board_size = len(board)
    color = board[row][col]
    visited = set()
    stack = [(row, col)]
    liberties = 0

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while stack:
        current_row, current_col = stack.pop()

        if (current_row, current_col) in visited:
            continue

        visited.add((current_row, current_col))

        for dr, dc in directions:
            nr, nc = current_row + dr, current_col + dc

            if nr < 0 or nr >= board_size or nc < 0 or nc >= board_size:
                continue

            if board[nr][nc] == 0 and (nr, nc) not in visited:
                liberties += 1
                visited.add((nr, nc))
            elif board[nr][nc] == color and (nr, nc) not in visited:
                stack.append((nr, nc))

    if liberties == 0:
        for point in visited:
            board[point] = 0

    return liberties



start_string = "(;GM[1]FF[4]RU[Chinese]SZ[19]KM[7.50]PW[W]PB[B]\n("
score = 0

corner_moves = [(2, 3), (3, 2), (3, 3), (2, 2)]
previous_boards = []
num_positions = 0
player_1_wins = 0
player_2_wins = 0
points_score_1 = 0
points_score_2 = 0
points_score_3 = 0
points_score_4 = 0
scores = []
same_move_choice = 0
different_move_choice = 0
same_move_choice_p1 = 0
different_move_choice_p1 = 0

for j in range(256):
    move_list = []
    move_list.append(corner_moves[j % 4])
    move_list.append((18 - corner_moves[int(j / 4) % 4][0], 18 - corner_moves[int(j / 4) % 4][1]))
    move_list.append((18 - corner_moves[int(j / 16) % 4][0], corner_moves[int(j / 16) % 4][1]))
    move_list.append((corner_moves[int(j / 64) % 4][0], 18 - corner_moves[int(j / 64) % 4][1]))

    board = np.zeros((19, 19))
    board[move_list[0][0], move_list[0][1]] = 1
    board[move_list[1][0], move_list[1][1]] = 2
    board[move_list[2][0], move_list[2][1]] = 1
    board[move_list[3][0], move_list[3][1]] = 2

    board_vertical_flip = np.flip(board, 0)  # I assumed axis=0 for a vertical flip. Change if needed.

    skip = False
    for b in previous_boards:
        if np.array_equal(b, board_vertical_flip):
            skip = True
            break

    if skip:
        continue

    # Add the current board to previous_boards
    previous_boards.append(board.copy())  # Use copy() to ensure you're saving a unique instance of the board.

    last_moves = [(-1, -1), move_list[0], move_list[1], move_list[2], move_list[3]]
    color = 1
    player_1 = 1
    player_1_color = 1

    capture_last_move = 0
    ownership = np.zeros((19,19))


    for each_color in range(2):
        # if j*2 + each_color <= 0:
        #     continue
        evals = [1, 2, 3, 4]
        for move_idx in range(280):
            # print("move_idx = ", move_idx, flush = True)
            if player_1_color == color:
                if color == 2:
                    komi = 7.5
                    board_temp = (3 - board) % 3
                else:
                    komi = -7.5
                    board_temp = board
                current_player = 1
                global_data = np.array([0, 0, 0, 0, komi, 0, 1, capture_last_move])

                # num_visits = 1
                # threads = 1
                # if move_idx > 200:
                #     threads = 1
                #     num_visits = 1
                gen = generate_move(board, last_moves, color, capture_last_move, 1)
                move_output, winrate_output, policy_output = gen.run(1)
                del gen
                # move_output, winrate_output, sb_output, ownership = get_kata_output2(board_temp, move_list, global_data)
                # move_output, winrate_output, policy_output = generate_move_kata_actual(board, last_moves, color, capture_last_move, 100)

                max_move_value = move_output.max()
                puct_choice = np.unravel_index(move_output.argmax(), (19, 19))

                max_move_value = policy_output.max()
                policy_choice = np.unravel_index(policy_output.argmax(), (19, 19))

                # print("for player 1")
                # print("puct_choice vs policy_choice", puct_choice, " v ", policy_choice)
                if policy_choice[0] == puct_choice[0] and policy_choice[1] == puct_choice[1]:
                    # print("They are the same")
                    same_move_choice_p1 += 1
                    evals.append(50*winrate_output[0, 0] + 50)
                else:
                    different_move_choice_p1 += 1
                    evals.append(50*winrate_output[0, 0] + 1050)




            else: # color == 1
                if color == 2:
                    komi = 7.5
                    board_temp = (3 - board) % 3
                else:
                    komi = -7.5
                    board_temp = board
                current_player = 2
                global_data = np.array([0, 0, 0, 0, komi, 0, 1, capture_last_move])
                # move_output, winrate_output, policy_output = generate_move_kata(board, last_moves, color, capture_last_move, 100)

                gen = generate_move_with_kata(board, move_list, color, capture_last_move, 1)
                move_output, winrate_output, policy_output = gen.run(1)
                del gen
                max_move_value = move_output.max()
                puct_choice = np.unravel_index(move_output.argmax(), (19, 19))

                max_move_value = policy_output.max()
                policy_choice = np.unravel_index(policy_output.argmax(), (19, 19))

                # print("puct_choice vs policy_choice", puct_choice, " v ", policy_choice)
                if policy_choice[0] == puct_choice[0] and policy_choice[1] == puct_choice[1]:
                    # print("They are the same")
                    same_move_choice += 1
                    evals.append(50*winrate_output[0, 0] + 50)
                else:
                    different_move_choice += 1
                    evals.append(50*winrate_output[0, 0] + 1050)
                # policy_output = move_output

                # move_output, winrate_output, sb_output, ownership = get_kata_output(board_temp, move_list, global_data)
                # if color == 2:
                #     winrate_output = -winrate_output
                evals.append(50*winrate_output[0, 0] + 50)

            max_move_value = move_output.max()
            chosen_move = np.unravel_index(move_output.argmax(), (19, 19))

            print("Move num:", move_idx + 4, "  -  ", move_output)

            if color == 2:
                komi = 7.5
                board_temp = (3 - board) % 3
                global_data = np.array([0, 0, 0, 0, komi, 0, 1, capture_last_move])
                move_output_18b, winrate_output_kata, sb_output, ownership = get_kata_output2(board_temp, move_list, global_data)
                winrate_output_kata = -winrate_output_kata/2 + 0.5
                sb_output = -sb_output

            else:
                komi = -7.5
                board_temp = board
                global_data = np.array([0, 0, 0, 0, komi, 0, 1, capture_last_move])
                move_output_18b, winrate_output_kata, sb_output, ownership = get_kata_output2(board_temp, move_list, global_data)
                winrate_output_kata = winrate_output_kata/2 + 0.5
                sb_output = sb_output


            kata_move = move_output_18b[0, chosen_move[0], chosen_move[1]]
            max_move_value = policy_output.max()
            played_move = max_move_value

            scores.append([current_player, winrate_output_kata[0, 0], sb_output[0, 0], kata_move, played_move, 50*winrate_output[0, 0] + 50])


            move_list.append(chosen_move)

            if color == 2:
                komi = 7.5
            else:
                komi = -7.5

            board[chosen_move] = color
            capture_last_move = get_captures(board, chosen_move[0], chosen_move[1])

            color = color % 2 + 1
            last_moves.pop(0)
            last_moves.append(chosen_move)


        # Function to convert a coordinate to the corresponding letter
        def coord_to_letter(coord):
            return chr(ord('a') + coord)

        player = 'B'
        move_strings = []

        for move, eval in zip(move_list, evals):
            move_str = ";{}[{}{}]C[{}]".format(player, coord_to_letter(move[0]), coord_to_letter(move[1]), eval)
            move_strings.append(move_str)
            # Alternate between Black and White
            player = 'W' if player == 'B' else 'B'

        resulting_moves = "\n".join(move_strings)
        # os.makedirs(r'D:\matches', exist_ok=True)
        folder_name = 'D:/matches/' + model_name + model_name2 + '_playouts=' + str(num_playouts)
        os.makedirs(folder_name, exist_ok=True)

        # Write to the file
        with open(folder_name + '/game' + str(j*2 + each_color) + '.sgf', 'w') as file:
            file.write(start_string)
            file.write(resulting_moves)
            file.write("))")

        if player_1_color == color:
            player_1_wins += 1 - np.round(winrate_output_kata)
            player_2_wins += np.round(winrate_output_kata)

            if 1 - np.round(winrate_output_kata) == 1:
                points_score_1 += sb_output

            else:
                points_score_2 += sb_output
        else:
            player_1_wins += np.round(winrate_output_kata)
            player_2_wins += 1 - np.round(winrate_output_kata)

            if np.round(winrate_output_kata) == 1:
                points_score_3 += sb_output

            else:
                points_score_4 += sb_output

        print(np.round(winrate_output_kata))
        print("same_move_choice_p1 = ", same_move_choice_p1)
        print("different_move_choice_p1 = ", different_move_choice_p1)
        print("same_move_choice = ", same_move_choice)
        print("different_move_choice = ", different_move_choice)
        print("player_1 = ", player_1)
        print("player_1_color = ", player_1_color)
        print("color = ", color)
        print("winrate_output = ", winrate_output_kata)
        print("game number = ", num_positions)
        print("sb_output = ", sb_output)
        print("player_1_wins = ", player_1_wins)
        print("player_2_wins = ", player_2_wins)

        print("points_score_1 = ", points_score_1)
        print("points_score_2 = ", points_score_2)
        print("points_score_3 = ", points_score_3)
        print("points_score_4 = ", points_score_4, flush = True)
        # a = 1/0



        with open(folder_name + '.txt', 'a') as f:
            print("start", file=f)
            for line in scores:
                print(line, file=f)
        scores = []

        player_1_color = player_1_color % 2 + 1
        board = previous_boards[-1].copy()  # Use copy() to ensure you're saving a unique instance of the board.
        move_list = move_list[0:4]
        num_positions += 1

        last_moves = [(-1, -1), move_list[0], move_list[1], move_list[2], move_list[3]]
        color = 1

#
