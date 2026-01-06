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

import tensorflow as tf
from tensorflow.keras.layers import Layer

import subprocess
import threading
import time
import math
import matplotlib.pyplot as plt
import matplotlib.colors

def flip_board(board, flip_type):
    if flip_type == 1:
        return board
    elif flip_type == 2:
        return np.flipud(board)
    elif flip_type == 3:
        return np.fliplr(board)
    elif flip_type == 4:
        return np.transpose(board)
    elif flip_type == 5:
        return np.transpose(np.flipud(board))
    elif flip_type == 6:
        return np.transpose(np.fliplr(board))
    elif flip_type == 7:
        return np.fliplr(np.flipud(board))
    else:  # flip == 8
        return np.transpose(np.fliplr(np.flipud(board)))

def flip_point(point, flip_type):
    max_index = 18  # As the board is 19x19
    x, y = point

    if flip_type == 1 or x < 0:  # No flip
        return point
    elif flip_type == 2:  # Flip up-down
        return max_index - x, y
    elif flip_type == 3:  # Flip left-right
        return x, max_index - y
    elif flip_type == 4:  # Transpose
        return y, x
    elif flip_type == 5:  # Flip up-down, then transpose
        return y, max_index - x
    elif flip_type == 6:  # Flip left-right, then transpose
        return max_index - y, x
    elif flip_type == 7:  # Flip up-down, then flip left-right
        return max_index - x, max_index - y
    else:  # Flip up-down, flip left-right, then transpose (flip_type == 8)
        return max_index - y, max_index - x

def create_input_data(board, last_moves, global_data, ownership, moves):
    batch = board.shape[0]

    input_data = np.zeros((batch, 19, 19, 23), dtype=float)
    legal_moves = np.zeros((batch, 19, 19), dtype=int)
    winrates = np.zeros((batch, 2), dtype=float)
    global_input = np.zeros((batch, 5), dtype=float)
    target_moves = np.zeros((batch, 19, 19), dtype=float)
    group_labels = np.zeros((batch, 19, 19), dtype=int)

    for i in range(0, batch):
        input_data[i], legal_moves[i], winrates[i], global_input[i], ownership[i], target_moves[i], group_labels[i] = create_input_data_individual(board[i], last_moves[i], global_data[i], ownership[i], moves[i])

    return input_data, legal_moves, winrates, global_input, ownership, target_moves, group_labels

def get_dll_data(board, last_move, last_move_was_a_capture):
    if False and last_move[0] >= 0:
        if board[last_move[0], last_move[1]] > 2 or board[last_move[0], last_move[1]] < 1:
            print("This is now the dll part")
            print("error")
            print("Here are the last moves: ", last_moves)
            print("flip_type = ",  flip_type)
            board[last_move[0], last_move[1]] = board[last_move[0], last_move[1]] + 3
            print("Here is where that is on the board: ", board[last_move[0], last_move[1]])
            print(board)
            print("last_move_was_a_capture = ", last_move_was_a_capture)
            print("target move = ", move[last_move[0], last_move[1]], flush = True)
            board[last_move[0], last_move[1]] = board[last_move[0], last_move[1]] - 3
            # a = 1/0
    # load the DLL
    lib = ctypes.CDLL('.\\InputLabel.dll')

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

def create_input_data_individual(board, last_moves, global_data, ownership, move):
    side_to_move = global_data[6]
    last_move_was_a_capture = global_data[7]
    winrates = np.zeros((2), dtype=float)

    winrates[0] = (global_data[0] - global_data[1]) + 0 * global_data[5]
    winrates[1] = global_data[3]

    komi = global_data[4]

    input_data = np.zeros((19, 19, 23), dtype=float)
    move_history = np.zeros((19, 19, 5), dtype=int)

    if True:
        for i in range(0, 5):
            if i < len(last_moves):
                r, c = last_moves[i]
                if r >= 0:
                    move_history[r, c, i] = 1


    additional_inputs = get_dll_data(board.astype(int), last_moves[4], last_move_was_a_capture)
    legal_moves = additional_inputs[8]

    group_labels = additional_inputs[4].copy()
    group_labels[group_labels >= 1000] -= 1000
    additional_inputs[4] = (additional_inputs[4] - group_labels)/1000

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

    # group_labels = additional_inputs[12]

    return input_data, legal_moves, winrates, global_input, (ownership + 1) / 2, move, group_labels

def load_and_prepare_data(pickle_file):
    with open(pickle_file, 'rb') as f:
        all_positions, all_moves, all_global_data, all_last_moves, all_ownership = pickle.load(f)

    # Convert to numpy arrays
    all_positions = np.array(all_positions, dtype=np.int8)
    all_moves = np.array(all_moves, dtype=np.int16)
    all_global_data = np.array(all_global_data, dtype=np.float32)
    print("all_global_data.shape = ", all_global_data.shape)
    all_last_moves = np.array(all_last_moves, dtype=np.int8)
    all_ownership = np.array(all_ownership, dtype=np.int8)

    print(all_positions.shape)
    print(all_moves.shape)
    print(all_global_data.shape)
    print(all_last_moves.shape)
    print(all_ownership.shape)

    # Create a tf.data.Dataset object for efficient data loading
    dataset = tf.data.Dataset.from_tensor_slices((all_positions, all_moves, all_global_data, all_last_moves, all_ownership))

    return dataset

global unmodified_value_loss
global basic_unmodified_value_loss
global square_value_loss
global basic_square_value_loss
unmodified_value_loss = 0
basic_unmodified_value_loss = 0
square_value_loss = 0
basic_square_value_loss = 0
global store_losses
global store_targets
global store_pols
store_losses = []
store_evals = []
basic_store_losses = []
store_targets = []
store_pols = []

global cross_ent
cross_ent = 0
global basic_cross_ent
basic_cross_ent = 0
global moves_4_loss
global moves_1_loss
global kld_square_loss
global max_values
max_values = 0
moves_4_loss = 0
moves_1_loss = 0
kld_square_loss = 0


global own_1
global own_2
global own_3
global own_4
global ownership_terr
global ownership_loss_terr
global ownership_loss_life

ownership_terr = 0
ownership_loss_terr = 0
ownership_loss_life = 0
own_1 = 0
own_2 = 0
own_3 = 0
own_4 = 0

global kld_1_important
global kld_2_important
global model_fails
global kata_fails
kld_1_important = 0
kld_2_important = 0
model_fails = 0
kata_fails = 0

global correct_side_winning
correct_side_winning = 0

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

class GoModel(tf.keras.Model):
    def __init__(self, config):
        super(GoModel, self).__init__()
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
                save_res = res_block_output[:, :, :, 0:32]
                sum_slice = self.group_link_sum(res_block_output[:, :, :, 0:16], masks) + res_block_output[:, :, :, 0:16] * mean_mask/10
                mean_slice = self.group_link_mean(res_block_output[:, :, :, 16:32], masks) + res_block_output[:, :, :, 16:32] * mean_mask
                res_block_output = tf.concat([sum_slice, mean_slice, res_block_output[:, :, :, 32:]], axis=-1)




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

    def calculate_L2_loss(self):
        # For Dense and Conv2D layers
        def l2_loss_for_layer(layer):
            return tf.reduce_sum(layer.kernel ** 2)

        # List of layers
        layers = [
            self.input_conv,
            self.global_input_dense, self.conv, self.g1_conv, self.g1_fc,
            self.policy_conv, self.ownership_conv, self.value_head_conv,
            self.value_head_dense1, self.value_head_dense3
        ]

        # Add all the res_conv1, res_bn1, res_conv2, res_bn2, global_conv, and global_fc layers
        layers.extend(self.res_conv1)
        layers.extend(self.res_bn1)
        layers.extend(self.res_conv2)
        layers.extend(self.res_bn2)
        layers.extend(self.global_conv)
        layers.extend(self.global_fc)

        # Calculate L2 loss for each layer and sum them up
        l2_loss = sum([l2_loss_for_layer(layer) for layer in layers if hasattr(layer, 'kernel')])

        return l2_loss

    def compute_l2_loss(self):
        l2_loss = 0.0
        for layer in self.layers:
            if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
                weights = layer.get_weights()
                if weights:  # Check if layer has weights
                    kernel_weights = weights[0]  # Assuming the first element is the kernel weights
                    l2_loss += tf.reduce_sum(tf.square(kernel_weights))

                    if len(weights) > 1:
                        bias_weights = weights[1]  # Assuming the second element is the bias
                        l2_loss += tf.reduce_sum(tf.square(bias_weights)) / 20.0  # Scale down bias L2 loss
        return l2_loss

    def loss_function(self, output_policy, output_winrate, output_ownership, target_policy, target_winrate, target_ownership, input_legal_moves, input_data_batch, input_global_batch, last_moves, global_data, group_labels):
        BATCH_SIZE = output_policy.shape[0]
        update_interval = 100
        global unmodified_value_loss
        global basic_unmodified_value_loss
        global square_value_loss
        global basic_square_value_loss
        global cross_ent
        global basic_cross_ent
        global store_pols
        global own_1
        global own_2
        global own_3
        global own_4
        global moves_4_loss
        global moves_1_loss
        global kld_square_loss
        global max_values
        global kld_1_important
        global kld_2_important
        global model_fails
        global kata_fails
        global ownership_terr
        global ownership_loss_terr
        global ownership_loss_life
        global correct_side_winning

        ############## Value Loss ##############

        # Reshape
        input_legal_moves_flat = tf.reshape(input_legal_moves, [BATCH_SIZE, -1])
        output_policy_flat = tf.reshape(output_policy, [BATCH_SIZE, -1])

        # Mask illegal moves
        output_policy_flat = output_policy_flat + 10000.0 * input_legal_moves_flat - 10000.0
        softmaxed_policy_output = tf.nn.softmax(output_policy_flat, axis=1)
        output_winrate = tf.tanh(output_winrate) / 2 + 0.5
        output_ownership = tf.tanh(output_ownership) / 2 + 0.5

        # softmaxed_policy_output =  output_policy_flat
        # output_winrate = output_winrate / 2 + 0.5
        # output_ownership = -output_ownership/2 + 0.5

        num_of_stones = input_global_batch[:, 1]
        importance = tf.ones_like(num_of_stones)

        mask30 = num_of_stones < 30 /360.
        mask180 = num_of_stones > 180/360.

        # Apply the first condition
        importance = tf.where(mask30, 0.5 * tf.ones_like(importance), importance)
        importance = tf.where(mask180, 1 + 1.8 * (1 - 2 * num_of_stones), importance)

        importance = tf.maximum(importance, 0)

        # crazy_mask = tf.cast(tf.greater(tf.reduce_sum(input_legal_moves_flat, axis = -1), 3 * tf.reduce_sum(good_moves_mask, axis = -1)), tf.float32)
        # importance = importance * crazy_mask


        winrate_size = 0

        x = target_winrate[:, 0] / 2

        a = 10000
        winrate_tensor = 0.5 + tf.sign(x) * (a ** tf.abs(x) - 1) * (0.5 / (a ** 0.5 - 1))

        unmodified_value_loss_list = tf.abs(output_winrate - winrate_tensor)
        basic_unmodified_value_loss_list = tf.abs(0.5 - winrate_tensor)
        square_value_loss_list = tf.square(output_winrate - winrate_tensor)
        basic_square_value_loss_list = tf.square(0.5 - winrate_tensor)

        store_losses.extend(unmodified_value_loss_list.numpy().tolist())
        store_evals.extend(output_winrate.numpy().tolist())
        store_targets.extend(winrate_tensor.numpy().tolist())
        basic_store_losses.extend(basic_unmodified_value_loss_list.numpy().tolist())

        unmodified_value_loss += tf.reduce_mean(unmodified_value_loss_list)
        basic_unmodified_value_loss += tf.reduce_mean(basic_unmodified_value_loss_list)
        square_value_loss += tf.reduce_mean(square_value_loss_list)
        basic_square_value_loss += tf.reduce_mean(basic_square_value_loss_list)

        transform_constant = 0.998
        output_winrate = transform_constant * output_winrate + (1 - transform_constant) / 2
        winrate_tensor = transform_constant * winrate_tensor + (1 - transform_constant) / 2

        # Computing complementary values
        complement_winrate_tensor = 1.0 - winrate_tensor
        complement_output_winrate = 1.0 - output_winrate

        # Stacking the original and complementary values together
        stacked_winrate = tf.stack([winrate_tensor, complement_winrate_tensor], axis=-1)
        stacked_output = tf.stack([output_winrate, complement_output_winrate], axis=-1)

        # Computing KLD
        value_loss = 5 * tf.reduce_mean(tf.reduce_sum(stacked_winrate * tf.math.log(1e-10 + stacked_winrate / (stacked_output + 1e-10)), axis=-1) * importance)
        basic_value_loss = 5 * tf.reduce_mean(tf.reduce_sum(stacked_winrate * tf.math.log(1e-10 + stacked_winrate / (stacked_output * 0 + 0.5)), axis=-1)* importance)

        correct_side_winning += tf.reduce_sum((winrate_tensor - 0.5) * (output_winrate - 0.5))
        offset_error = 0

        ############## Policy Loss ##############

        # Normalize the target_policy so that the sum of each batch equals 1
        target_policy = tf.reshape(target_policy, [-1, 361])
        sums = tf.reduce_sum(target_policy, axis=1, keepdims=True)
        normalized_target_policy = target_policy / (sums + 1e-10)
        normalized_target_policy_save = normalized_target_policy * 0.999999

        three_moves_mask = tf.cast(tf.greater(normalized_target_policy, 0.002), tf.float32)
        normalized_target_policy = normalized_target_policy - 0.0015
        normalized_target_policy = tf.maximum(normalized_target_policy, 0)

        normalized_target_policy = tf.tanh(140 * normalized_target_policy) + 2 * normalized_target_policy

        # normalized_target_policy = normalized_target_policy + 1e-5 * input_legal_moves_flat
        sums = tf.reduce_sum(normalized_target_policy, axis=1, keepdims=True)
        normalized_target_policy = normalized_target_policy / (sums + 1e-10)

        # KL Divergence loss
        softmaxed_policy_output = tf.where(input_legal_moves_flat == 0, 0.0, softmaxed_policy_output)
        max_values += tf.reduce_mean(tf.reduce_max(softmaxed_policy_output))

        cross_ent_2 = tf.reduce_sum(normalized_target_policy * tf.math.log( 1/(1e-10 + softmaxed_policy_output)), axis=-1)
        false_pos_loss_2 = tf.reduce_sum(softmaxed_policy_output / (normalized_target_policy + 1e-10), axis=-1)
        cross_ent = 1 * tf.reduce_mean(cross_ent_2)
        false_pos_loss = 10e-5 * tf.reduce_mean(false_pos_loss_2)

        good_moves_mask = tf.cast(tf.greater(target_policy, 0), tf.float32)
        good_move_probs = good_moves_mask * softmaxed_policy_output
        sum_of_good_moves = tf.reduce_sum(good_move_probs, axis = -1)
        false_pos_loss = -tf.math.log(sum_of_good_moves + 1e-10)
        false_pos_loss = 10 * tf.reduce_mean(false_pos_loss)

        policy_loss = cross_ent + false_pos_loss

        top_values, top_indices = tf.math.top_k(softmaxed_policy_output, k=1)
        mask_1 = tf.reduce_any(tf.equal(tf.expand_dims(softmaxed_policy_output, 2), tf.expand_dims(top_values, 1)), axis=2)
        mask_1 = tf.cast(mask_1,  tf.float32)

        if tf.reduce_sum(mask_1) == BATCH_SIZE:
            top_1 = tf.cast(mask_1,  tf.float32) * tf.cast(tf.equal(target_policy, 0),  tf.float32)
            moves_1_loss_temp = tf.reduce_sum(top_1, axis = -1) * importance
            moves_1_loss_temp = tf.reduce_sum(moves_1_loss_temp)

            top_4 = tf.cast(mask_1,  tf.float32) * (1 - three_moves_mask)
            moves_4_loss_temp = tf.reduce_sum(top_4, axis = -1) * importance
            moves_4_loss_temp = tf.reduce_sum(moves_4_loss_temp)

        else:
            moves_1_loss_temp = 0.06 * BATCH_SIZE
            moves_4_loss_temp = 0

        moves_4_loss += moves_4_loss_temp
        moves_1_loss += moves_1_loss_temp

        ############## Ownership Loss ##############
        target_ownership  = tf.cast(target_ownership, tf.float32)
        output_ownership = tf.reshape(output_ownership, (BATCH_SIZE, 19, 19))


        stones = input_data_batch[:, :, :, 0] + input_data_batch[:, :, :, 1]
        stones  = tf.cast(stones, tf.float32)

        my  = tf.cast(input_data_batch[:, :, :, 0], tf.float32)
        opp = tf.cast(input_data_batch[:, :, :, 1], tf.float32)

        target_life = target_ownership * stones
        output_life = output_ownership * stones

        target_my = target_ownership * my
        target_opp = target_ownership * opp
        output_my = output_ownership * my
        output_opp = output_ownership * opp

        #Reshape
        target_ownership_flat = tf.reshape(target_ownership, [BATCH_SIZE, -1])
        output_ownership_flat = tf.reshape(output_ownership, [BATCH_SIZE, -1])

        target_life_flat = tf.reshape(target_life, [BATCH_SIZE, -1])
        output_life_flat = tf.reshape(output_life, [BATCH_SIZE, -1])

        target_my_flat = tf.reshape(target_my, [BATCH_SIZE, -1])
        target_opp_flat = tf.reshape(target_opp, [BATCH_SIZE, -1])
        output_my_flat = tf.reshape(output_my, [BATCH_SIZE, -1])
        output_opp_flat = tf.reshape(output_opp, [BATCH_SIZE, -1])

        # Calculate loss
        # ownership_loss = 10 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=target_ownership_flat, logits=output_ownership_flat))


        ownership_loss_terr_temp = 3 * (-tf.reduce_mean(target_ownership * tf.math.log(output_ownership + 1e-10) +
                        (1. - target_ownership) * tf.math.log(1. - output_ownership + 1e-10), axis = (1, 2)))

        ownership_loss_life_temp = 30 * (-tf.reduce_mean(target_life * tf.math.log(output_life + 1e-10) +
                        (1. - target_life) * tf.math.log(1. - output_life + 1e-10), axis = (1, 2)))

        ownership_loss_terr += tf.reduce_mean(ownership_loss_terr_temp)
        ownership_loss_life += 1.11 * tf.reduce_mean(ownership_loss_life_temp * tf.math.square(importance))
        ownership_loss = ownership_loss_terr + ownership_loss_life

        loss_1 = 120 * -tf.reduce_mean((1. - target_my_flat) * tf.math.log(1. - output_my_flat + 1e-10)) * importance ** 2
        loss_2 = 120 * (-tf.reduce_mean(target_my_flat * tf.math.log(output_my_flat + 1e-10))) * importance ** 2

        loss_3 = 120 * -tf.reduce_mean((1. -target_opp_flat) * tf.math.log(1. - output_opp_flat + 1e-10)) * importance ** 2
        loss_4 = 120 * (-tf.reduce_mean(target_opp_flat * tf.math.log(output_opp_flat + 1e-10))) * importance ** 2

        own_1 += loss_1.numpy()
        own_2 += loss_2.numpy()
        own_3 += loss_3.numpy()
        own_4 += loss_4.numpy()

        ############## L2 Regularization Loss ##############
        l2_coeff = 0.0001
        total_l2_loss = l2_coeff * self.calculate_L2_loss()

        self.value_loss.assign_add(value_loss)
        self.basic_value_loss.assign_add(basic_value_loss)
        self.policy_loss.assign_add(policy_loss)
        self.cross_ent_loss.assign_add(cross_ent)
        self.false_pos_loss.assign_add(false_pos_loss)
        # kld_1_important += kld_1_important_loss
        # kld_2_important += kld_2_important_loss
        # kld_square_loss += kld_square
        self.L2_loss.assign_add(total_l2_loss)
        self.winrate_size_loss.assign_add(winrate_size)
        self.offset_error.assign_add(offset_error)
        self.global_counter.assign_add(1)



        if self.global_counter.numpy() % update_interval == 0:
            # This will fetch the current learning rate from the optimizer
            # current_learning_rate = self.optimizer.learning_rate(self.global_counter).numpy()

            print("\nStep                 = ", self.global_counter.numpy())
            print("model_fails = ", model_fails)
            print("kata_fails = ", kata_fails)
            print("kld_1                  = ", self.cross_ent_loss.numpy()/update_interval)
            print("kld_2                  = ", self.false_pos_loss.numpy()/update_interval)
            print("moves_4                = ", 100*moves_4_loss.numpy()/(BATCH_SIZE*update_interval))
            print("moves_1                = ", 100*moves_1_loss.numpy()/(BATCH_SIZE*update_interval))
            print("l2_loss                = ", self.L2_loss.numpy()/update_interval)
            print("own_loss               = ", ownership_loss.numpy()/update_interval)
            print("own_loss_life          = ", ownership_loss_life.numpy()/update_interval)
            print("own_loss_terr          = ", ownership_loss_terr.numpy()/update_interval)
            print("max_values             = ", max_values.numpy()/update_interval)
            # print("cross_entropy          = ", cross_ent/update_interval, "\t", cross_ent/basic_cross_ent)
            print("winrate_size_loss      = ", self.winrate_size_loss.numpy()/update_interval)
            print("\noffset_error           = ",self.offset_error.numpy()/update_interval)
            print("unmodified_value_loss  = ", unmodified_value_loss.numpy()/update_interval, "\t", unmodified_value_loss.numpy()/basic_unmodified_value_loss)
            print("square_value_loss      = ", square_value_loss/update_interval, "\t", square_value_loss/basic_square_value_loss)
            print("value_loss             = ", self.value_loss.numpy()/update_interval, "\t", self.value_loss.numpy()/self.basic_value_loss.numpy())
            print("policy_loss            = ", self.policy_loss.numpy()/update_interval)
            print("Average loss           = ", (self.L2_loss.numpy() + self.sb_loss.numpy() + self.ownership_loss.numpy() + self.value_loss.numpy() + self.policy_loss.numpy())/update_interval, flush=True)
            print("own1                   = ", own_1/update_interval)
            print("own2                   = ", own_2/update_interval)
            print("own3                   = ", own_3/update_interval)
            print("own4                   = ", own_4/update_interval)
            print("correct_side_winning   = ", correct_side_winning)

            bins = np.arange(0, 1.05, 0.05)  # This will give [0, 0.05, 0.1, ..., 1.0]

            fig, axs = plt.subplots(1, 1, figsize=(7.5, 7.5), tight_layout=True)  # 1 row, 3 columns
            # Plot store_losses
            axs.hist(store_losses, bins=bins, edgecolor='black', alpha=0.7)
            # axs.set_title('Error between Target and Output Winrate for the Model')
            axs.set_title('Error between Training and Output Winrate for the Model')
            axs.set_xlabel('Loss')
            axs.set_ylabel('Frequency')
            axs.set_xticks(bins)
            axs.grid(axis='y')
            # Set the y-axis max value
            axs.set_ylim([0, BATCH_SIZE * update_interval * 0.6])
            plt.show()

            # Plot store_evals
            fig, axs = plt.subplots(1, 1, figsize=(7.5, 7.5), tight_layout=True)  # 1 row, 3 columns
            n_evals, bins, patches = axs.hist(store_evals, bins=bins, edgecolor='black', alpha=0.7)
            # axs.set_title('Output Winrates for the Model')
            axs.set_title('Output Winrates for the Model')
            axs.set_xlabel('Loss')
            axs.set_ylabel('Frequency')
            axs.set_xticks(bins)
            axs.grid(axis='y')
            # Set the y-axis max value
            axs.set_ylim([0, BATCH_SIZE * update_interval * 0.4])
            plt.show()

            n_targets, bins, patches = axs.hist(store_targets, bins=bins, edgecolor='black', alpha=0.7)


            # Print the values of each bin
            for i in range(len(n_evals)):
                print(f"Bin {bins[i]:.2f} to {bins[i+1]:.2f}: {n_evals[i]}")

            print("I will now print evals")
            print(n_evals)

            print("Just printed targets")
            print(n_targets)


            # Plot store_targets
            fig, axs = plt.subplots(1, 1, figsize=(7.5, 7.5), tight_layout=True)  # 1 row, 3 columns
            n_evals, bins, patches = axs.hist(store_targets, bins=bins, edgecolor='black', alpha=0.7)
            axs.set_title('Target Winrates in Training Data')
            # axs.set_title('Winrates given by KataGo b18c384nbt')
            axs.set_xlabel('Loss')
            axs.set_ylabel('Frequency')
            axs.set_xticks(bins)
            axs.grid(axis='y')
            # Set the y-axis max value
            axs.set_ylim([0, BATCH_SIZE * update_interval * 0.4])
            plt.show()

            print("Std of store_targets = ", np.std(store_targets))
            print("std of store_evals", np.std(store_evals))



            # Create a figure with two subplots
            # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            #
            # # Histogram with 1000 bins from 0 to 1
            # # print(store_pols, flush = True)
            # axs[0].hist(store_pols, bins=100, range=(0, 1))
            # axs[0].set_title("model_new_cross_ent_step=158100")
            # axs[0].set_xlabel("Value")
            # axs[0].set_ylabel("Frequency")
            #
            # # Histogram with 50 bins from 0 to 0.05
            # axs[1].hist(store_pols, bins=50, range=(0, 0.05))
            # axs[1].set_title("Histogram with 50 Bins (0 to 0.05)")
            # axs[1].set_xlabel("Value")
            # axs[1].set_ylabel("Frequency")
            #
            # # Show the plots
            # plt.tight_layout()
            # plt.show()

            with open('model_evals', 'w') as file:
                file.write(str(store_evals))
                file.write(str(store_targets))
                file.write(str(store_losses))

            a = 1/0

        total_loss = ownership_loss + policy_loss + value_loss + total_l2_loss + winrate_size
        return total_loss

    def train(self, batch_size):
        self.global_counter.assign(1)

        print(f"Now loading small_training_data_file.pkl", flush = True)
        dataset = load_and_prepare_data(f'small_training_data_file.pkl')
        print("Now shuffling", flush = True)
        data = dataset.shuffle(100000).batch(batch_size)
        print("Done shuffling", flush = True)

        for j in range(0, 1000):
            print(f"small_training_data_file.pkl", flush = True)
            # Explicitly delete the old dataset before loading a new one

            # data = load_and_prepare_data(f'combined_{(j%3)}.pkl')
            # data = load_and_prepare_data('small_complete_0.pkl')
            # data = data.batch(batch_size)

            for i, (positions, moves, global_data, last_moves, ownership) in enumerate(data):
                if i % 10 == 0:
                    print(i, flush = True)
                input_data_batch, input_legal_moves_batch, target_winrate_batch, input_global_batch, target_ownership_batch, target_policy_batch, group_labels_batch = create_input_data(positions.numpy(), last_moves.numpy(), global_data.numpy(), ownership.numpy(), moves.numpy())

                input_data_batch = tf.convert_to_tensor(input_data_batch, tf.float32)
                input_legal_moves_batch = tf.convert_to_tensor(input_legal_moves_batch, tf.float32)
                input_global_batch = tf.convert_to_tensor(input_global_batch, tf.float32)
                target_winrate_batch = tf.convert_to_tensor(target_winrate_batch, tf.float32)
                target_ownership_batch = tf.convert_to_tensor(target_ownership_batch, tf.int8)
                target_policy_batch = tf.convert_to_tensor(target_policy_batch, tf.float32)
                group_labels_batch = tf.convert_to_tensor(group_labels_batch, tf.float32)

                with tf.GradientTape() as tape:
                    output_policy, output_winrate, output_ownership = self.call(input_data_batch, input_global_batch, group_labels_batch, training=False)

                    loss = self.loss_function(
                        output_policy, output_winrate, output_ownership,
                        target_policy_batch, target_winrate_batch, target_ownership_batch, input_legal_moves_batch, input_data_batch, input_global_batch, last_moves.numpy(), global_data.numpy(), group_labels_batch.numpy()
                    )
                #
                # # Backward pass
                # gradients = tape.gradient(loss, self.trainable_variables)
                # self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


                if self.global_counter.numpy() % 1000 == 0:
                    filename = f'model_group_prop_step={int(self.global_counter.numpy())}.h5'
                    print("filename = ", filename, flush = True)
                    try:
                        model.save_weights(filename)
                    except ValueError as e:
                        print(f"Error occurred while saving: {e}")
                        # traceback.print_exc()

                    print("I have just saved the model", flush = True)


model = GoModel(config)
model(tf.zeros([1, 19, 19, 23]), tf.zeros([1, 5]), tf.zeros([1, 19, 19]))

print(model.summary())
model.load_weights('model_new_cross_ent_step=268200.h5')


model.train(batch_size=25)
