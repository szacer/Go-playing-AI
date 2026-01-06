import numpy as np
import matplotlib.pyplot as plt

# game number =  271
# sb_output =  [[-2.868]]
# player_1_wins =  [[119.]]
# player_2_wins =  [[153.]]
# points_score_1 =  [[-2574.028]]
# points_score_2 =  [[2376.83]]
# points_score_3 =  [[2177.295]]
# points_score_4 =  [[-2371.958]]


def read_data(delta_win_type_1, delta_win_type_2, delta_score_type_1, delta_score_type_2, policy_type_1, policy_type_2, belief_type_1, belief_type_2, stop_number):
    previous_values = {}
    move_number = 0
    data_type = 3
    with open('D:\matches\model_new_cross_ent_step=268200model_1_visit_both_player_for_real_playouts=0.5.txt', 'r') as file:
        for line in file:
            line = line.strip()
            if line == 'start':
                if data_type == 1:
                    policy_type_1.pop()
                    belief_type_1.pop()

                if data_type == 2:
                    policy_type_2.pop()
                    belief_type_2.pop()


                previous_values = {}  # Reset for each new match
                move_number = 0
                continue

            if move_number > stop_number:
                move_number += 1
                continue

            # Parse the line
            print(line)
            data = eval(line)  # Convert string representation of list to actual list


            # Extract values
            data_type, win, score, policy, belief, player_winrate = data
            if data_type == 1:
                # Handle type 1
                if 2 in previous_values:  # Check if there's a previous type 2 value
                    if win + previous_values[2]['win'] < 0.00 or win + previous_values[2]['win'] > 2:
                        delta_win_type_2.pop()
                        delta_score_type_2.pop()
                        policy_type_2.pop()
                        belief_type_2.pop()

                    delta_win_type_2.append((win - previous_values[2]['win']) * 2 * (move_number % 2 - 0.5))
                    delta_score_type_2.append((score - previous_values[2]['score']) * 2 * (move_number % 2 - 0.5))
                    policy_type_1.append(policy)
                    belief_type_1.append(belief)

                if move_number == 0:
                    policy_type_1.append(policy)
                    belief_type_1.append(belief)

            elif data_type == 2:
                # Handle type 2
                if 1 in previous_values:  # Check if there's a previous type 1 value
                    if win + previous_values[1]['win'] < 0.00 or win + previous_values[1]['win'] > 2:
                        delta_win_type_1.pop()
                        delta_score_type_1.pop()
                        policy_type_1.pop()
                        belief_type_1.pop()

                    delta_win_type_1.append((win - previous_values[1]['win']) * 2 * (move_number % 2 - 0.5))
                    delta_score_type_1.append((score - previous_values[1]['score']) * 2 * (move_number % 2 - 0.5))
                    policy_type_2.append(policy)
                    belief_type_2.append(belief)

                if move_number == 0:
                    policy_type_2.append(policy)
                    belief_type_2.append(belief)


            # Update previous values
            previous_values[data_type] = {'win': win, 'score': score}
            move_number += 1
    if data_type == 1:
        policy_type_1.pop()
        belief_type_1.pop()

    if data_type == 2:
        policy_type_2.pop()
        belief_type_2.pop()

    print(len(delta_win_type_1))
    print(len(delta_win_type_2))
    print(len(delta_score_type_1))
    print(len(delta_score_type_2))
    print(len(policy_type_1))
    print(len(policy_type_2))
    print(len(belief_type_1))
    print(len(belief_type_2))



actual_loss = []
possible_loss = []

for i in range(28, 29):
    # Initialize arrays
    delta_win_type_1 = []
    delta_win_type_2 = []
    delta_score_type_1 = []
    delta_score_type_2 = []
    policy_type_1 = []
    policy_type_2 = []
    belief_type_1 = []
    belief_type_2 = []

    read_data(delta_win_type_1, delta_win_type_2, delta_score_type_1, delta_score_type_2, policy_type_1, policy_type_2, belief_type_1, belief_type_2, i * 10)

    # Convert lists to numpy arrays
    delta_win_type_1 = np.array(delta_win_type_1)
    delta_win_type_2 = np.array(delta_win_type_2)
    delta_score_type_1 = np.array(delta_score_type_1)
    delta_score_type_2 = np.array(delta_score_type_2)
    policy_type_1 = np.array(policy_type_1)
    policy_type_2 = np.array(policy_type_2)
    belief_type_1 = np.array(belief_type_1)
    belief_type_2 = np.array(belief_type_2)

    number_points = 109
    limit = 1
    x = np.linspace(0, 100 * limit, number_points)
    counts_1 = np.zeros(number_points)
    sums_1 = np.zeros(number_points)

    for i in range(len(delta_win_type_1)):
        if policy_type_1[i] < limit:
            index = int(1000 * policy_type_1[i])
            if index >= 10:
                index = int(index / 10) + 9
            counts_1[index] += 1
            sums_1[index] += delta_win_type_1[i]

    y1 = sums_1 / (counts_1 + 1)

    counts_2 = np.zeros(number_points)
    sums_2 = np.zeros(number_points)

    for i in range(len(delta_win_type_2)):
        if policy_type_2[i] < limit:
            index = int(1000 * policy_type_2[i])
            if index >= 10:
                index = int(index / 10) + 9
            counts_2[index] += 1
            sums_2[index] += delta_win_type_2[i]

    y2 = sums_2 / (counts_2 + 1)

    print("Actual Loss = ", np.sum(sums_1) - np.sum(sums_2))
    print("Possible Loss = ", np.sum(counts_1 * y2) - np.sum(counts_2 * y2), flush = True)

    actual_loss.append(np.sum(sums_1) - np.sum(sums_2))
    possible_loss.append(np.sum(counts_1 * y2) - np.sum(counts_2 * y2))

# x = np.linspace(10, 280, 28)
# plt.plot(x, actual_loss, label='Actual Loss')
# plt.plot(x, possible_loss, label='Possible Loss')
# # plt.plot(counts_1/np.sum(counts_1))
# plt.title('Loss vs Move Number')
# plt.xlabel('Move Number')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
#
# a = 1/0

number_points = 100
limit = 1
x = np.linspace(0.5, 99.5, number_points)
counts_1 = np.zeros(number_points)
sums_1 = np.zeros(number_points)

for i in range(len(delta_win_type_1)):
    if policy_type_1[i] < limit:
        index = int(100 * policy_type_1[i])
        # if index >= 10:
        #     index = int(index / 10) + 9
        counts_1[index] += 1
        sums_1[index] += delta_win_type_1[i]

y1 = sums_1 / (counts_1 + 1)

counts_2 = np.zeros(number_points)
sums_2 = np.zeros(number_points)

for i in range(len(delta_win_type_2)):
    if policy_type_2[i] < limit:
        index = int(100 * policy_type_2[i])
        # if index >= 10:
        #     index = int(index / 10) + 9
        counts_2[index] += 1
        sums_2[index] += delta_win_type_2[i]

y2 = sums_2 / (counts_2 + 1)

for i in range(sums_1.shape[0]):
    print(sums_1[i] - sums_2[i])

print("Sums_1 = ", np.sum(sums_1))
print("Sums_2 = ", np.sum(sums_2))

print("Precise_1 = ", np.sum(delta_win_type_1))
print("Precise_2 = ", np.sum(delta_win_type_2))

print("Actual Loss = ", np.sum(sums_1) - np.sum(sums_2))
print("Possible Loss = ", np.sum(counts_1 * y2) - np.sum(counts_2 * y2), flush = True)

# Create subplots: 3 rows, 1 column
# fig, axs = plt.subplots(1, 1, figsize=(15, 8))

print(100 * counts_1 / np.sum(counts_1))
print(100 * counts_2 / np.sum(counts_2))
# Second subplot: counts_1 and counts_2
plt.plot(100 * counts_1 / np.sum(counts_1), label='Model Allignment', color = 'blue')
plt.plot(100 * counts_2 / np.sum(counts_2), label='6 block KataGo Allignment', linestyle='dashed', color = 'red')
plt.title('Allignment of Moves with 40 block KataGo')
plt.legend()
plt.xlabel('40 block policy value')
plt.ylabel('Move count (%)')
plt.tight_layout()
plt.show()

plt.plot(100 * (sums_1 - sums_2) / 1332)
# axs.plot(counts_1 * y2 - counts_2 * y2, label='possible')
plt.title('Gain by Model over KataGo for each Policy Value')
plt.xlabel('40 block policy value')
plt.ylabel('Gain (%)')
plt.tight_layout()
plt.show()


#
# print("size of delta win = ", delta_win_type_1.shape, flush = True)
#
#
# number_points = 20
# x = np.linspace(1/(2*number_points), 100 - 1/(2*number_points), number_points)
# counts_1 = np.zeros(number_points)
# sums_1 = np.zeros(number_points)
#
# for i in range(len(delta_win_type_1)):
#     index = int(number_points * belief_type_1[i])
#     counts_1[index] += 1
#     sums_1[index] += delta_win_type_1[i]
#
# y1 = sums_1 / (counts_1 + 1)
#
# counts_2 = np.zeros(number_points)
# sums_2 = np.zeros(number_points)
#
# for i in range(len(delta_win_type_2)):
#     index = int(number_points * belief_type_2[i])
#     counts_2[index] += 1
#     sums_2[index] += delta_win_type_2[i]
#
# y2 = sums_2 / (counts_2 + 1)
#
#
# plt.plot(x, y1, label='Loss Type 1')
# plt.plot(x, y2, label='Loss Type 2')
# # plt.plot(counts_1/np.sum(counts_1))
# plt.title('Loss vs Belief')
# plt.xlabel('Belief Value')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
# #
# bins = np.linspace(0, 1, num=100)
#
# plt.figure(figsize=(15, 5))
# plt.subplot(1, 2, 1)
# plt.hist(belief_type_1, bins=bins, alpha=0.7, label='Policy Type 1')
# plt.title('Histogram of Belief Type 1')
# plt.xlabel('Policy Value')
# plt.ylabel('Frequency')
#
# # Plot histogram for policy_type_2
# plt.subplot(1, 2, 2)
# plt.hist(belief_type_2, bins=bins, alpha=0.7, label='Policy Type 2', color='orange')
# plt.title('Histogram of Belief Type 2')
# plt.xlabel('Policy Value')
# plt.ylabel('Frequency')
# plt.show()
#
# number_points = 109
# limit = 1
# x = np.linspace(0, 100 * limit, number_points)
# counts_1 = np.zeros(number_points)
# sums_1 = np.zeros(number_points)
#
# for i in range(len(delta_win_type_1)):
#     if policy_type_1[i] < limit:
#         index = int(1000 * policy_type_1[i])
#         if index >= 10:
#             index = int(index / 10) + 9
#         counts_1[index] += 1
#         sums_1[index] += delta_win_type_1[i]
#
# y1 = sums_1 / (counts_1 + 1)
#
# counts_2 = np.zeros(number_points)
# sums_2 = np.zeros(number_points)
#
# for i in range(len(delta_win_type_2)):
#     if policy_type_2[i] < limit:
#         index = int(1000 * policy_type_2[i])
#         if index >= 10:
#             index = int(index / 10) + 9
#         counts_2[index] += 1
#         sums_2[index] += delta_win_type_2[i]
#
# y2 = sums_2 / (counts_2 + 1)
#
# for i in range(sums_1.shape[0]):
#     print(sums_1[i] - sums_2[i])
#
# print("Sums_1 = ", np.sum(sums_1))
# print("Sums_2 = ", np.sum(sums_2))
#
# print("Precise_1 = ", np.sum(delta_win_type_1))
# print("Precise_2 = ", np.sum(delta_win_type_2))
#
# print("Actual Loss = ", np.sum(sums_1) - np.sum(sums_2))
# print("Possible Loss = ", np.sum(counts_1 * y2) - np.sum(counts_2 * y2), flush = True)
#
# # Create subplots: 3 rows, 1 column
# fig, axs = plt.subplots(2, 2, figsize=(15, 8))
#
# # First subplot: y1 and y2
# axs[0, 0].plot(y1, label='y1')
# axs[0, 0].plot(y2, label='y2')
# axs[0, 0].set_title('y1 and y2')
# axs[0, 0].legend()
#
# # Second subplot: counts_1 and counts_2
# axs[0, 1].plot(counts_1, label='counts_1')
# axs[0, 1].plot(counts_2, label='counts_2')
# axs[0, 1].set_title('counts_1 and counts_2')
# axs[0, 1].legend()
#
# # Third subplot: y1 * counts_1 and y2 * counts_2
# axs[1, 0].plot(sums_1, label='sums_1')
# axs[1, 0].plot(sums_2, label='sums_2')
# axs[1, 0].set_title('y1 * counts_1 and y2 * counts_2')
# axs[1, 0].legend()
#
# axs[1, 1].plot(sums_1 - sums_2, label='actual')
# axs[1, 1].plot(counts_1 * y2 - counts_2 * y2, label='possible')
# axs[1, 1].set_title('diff')
# axs[1, 1].legend()
#
# # Adjust layout
# plt.tight_layout()
#
# # Show plot
# plt.show()
#
# number_points = 10
# limit = 0.001
# x = np.linspace(0, 100 * limit, number_points)
# counts_1 = np.zeros(number_points)
# sums_1 = np.zeros(number_points)
#
# for i in range(len(delta_win_type_1)):
#     if policy_type_1[i] < limit:
#         index = int(10000 * policy_type_1[i])
#         counts_1[index] += 1
#         sums_1[index] += delta_win_type_1[i]
#
# y1 = sums_1 / (counts_1 + 1)
#
# counts_2 = np.zeros(number_points)
# sums_2 = np.zeros(number_points)
#
# for i in range(len(delta_win_type_2)):
#     if policy_type_2[i] < limit:
#         index = int(10000 * policy_type_2[i])
#         counts_2[index] += 1
#         sums_2[index] += delta_win_type_2[i]
#
# y2 = sums_2 / (counts_2 + 1)
#
# # Create subplots: 3 rows, 1 column
# fig, axs = plt.subplots(2, 2, figsize=(15, 8))
#
# # First subplot: y1 and y2
# axs[0, 0].plot(y1, label='y1')
# axs[0, 0].plot(y2, label='y2')
# axs[0, 0].set_title('y1 and y2')
# axs[0, 0].legend()
#
# # Second subplot: counts_1 and counts_2
# axs[0, 1].plot(counts_1, label='counts_1')
# axs[0, 1].plot(counts_2, label='counts_2')
# axs[0, 1].set_title('counts_1 and counts_2')
# axs[0, 1].legend()
#
# # Third subplot: y1 * counts_1 and y2 * counts_2
# axs[1, 0].plot(sums_1, label='sums_1')
# axs[1, 0].plot(sums_2, label='sums_2')
# axs[1, 0].set_title('y1 * counts_1 and y2 * counts_2')
# axs[1, 0].legend()
#
# axs[1, 1].plot(sums_1 - sums_2)
# axs[1, 1].set_title('diff')
# axs[1, 1].legend()
#
# # Adjust layout
# plt.tight_layout()
#
# # Show plot
# plt.show()
#
#
# #
# print("mean of delta_win_type_1 = ", np.mean(delta_win_type_1))
# print("mean of delta_win_type_2 = ", np.mean(delta_win_type_2))
#
# # print("mean of delta_win_type_1 = ", np.mean(np.abs(delta_win_type_1)))
# # print("mean of delta_win_type_2 = ", np.mean(np.abs(delta_win_type_2)))
#
# print("mean of policy_type_1 = ", np.mean(policy_type_1))
# print("mean of policy_type_2 = ", np.mean(policy_type_2))
#
# print("mean of belief_type_1 = ", np.mean(belief_type_1))
# print("mean of belief_type_2 = ", np.mean(belief_type_2))
#
# # print("delta_win_type_1 = ", np.sum(delta_win_type_1 < -0.1))
# # print("delta_win_type_2 = ", np.sum(delta_win_type_2 < -0.1))
# #
# # print("delta_win_type_1 = ", np.sum(delta_win_type_1 > 0.1))
# # print("delta_win_type_2 = ", np.sum(delta_win_type_2 > 0.1))
# #
# #
# #
# # Plot histogram for policy_type_1
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 3, 1)  # 1 row, 3 columns, first plot
# plt.hist(policy_type_1, bins=200, alpha=0.7, label='Policy Type 1')
# plt.title('Histogram of Policy Type 1')
# plt.xlabel('Policy Value')
# plt.ylabel('Frequency')
#
# # Plot histogram for policy_type_2
# plt.subplot(1, 3, 2)  # 1 row, 3 columns, second plot
# plt.hist(policy_type_2, bins=200, alpha=0.7, label='Policy Type 2', color='orange')
# plt.title('Histogram of Policy Type 2')
# plt.xlabel('Policy Value')
# plt.ylabel('Frequency')
#
# # Plot combined histogram
# plt.subplot(1, 3, 3)  # 1 row, 3 columns, third plot
# plt.hist(policy_type_1, bins=200, alpha=0.7, label='Policy Type 1')
# plt.hist(policy_type_2, bins=200, alpha=0.7, label='Policy Type 2', color='orange')
# plt.title('Combined Histogram of Policy Types')
# plt.xlabel('Policy Value')
# plt.ylabel('Frequency')
# plt.legend()
#
# plt.tight_layout()  # Adjust the layout
# plt.show()
#
# # Custom bin edges
# # positive_bins = [0, 0.01, 0.02, 0.03, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30, 0.40, 0.45, 0.50, 1]
# # negative_bins = [-x for x in positive_bins[::-1]]  # Mirroring for negative values
# # bins = negative_bins + positive_bins[1:]  # Combine and avoid duplicate zero
#
# bins = np.linspace(-1, 1, num=200)
#
# plt.figure(figsize=(15, 5))
# plt.subplot(1, 3, 1)
# plt.hist(delta_win_type_1, bins=bins, alpha=0.7, label='Policy Type 1')
# plt.title('Histogram of Delta Win Type 1')
# plt.xlabel('Policy Value')
# plt.ylabel('Frequency')
#
# # Plot histogram for policy_type_2
# plt.subplot(1, 3, 2)
# plt.hist(delta_win_type_2, bins=bins, alpha=0.7, label='Policy Type 2', color='orange')
# plt.title('Histogram of Delta Win Type 2')
# plt.xlabel('Policy Value')
# plt.ylabel('Frequency')
#
# # Plot combined histogram
# plt.subplot(1, 3, 3)
# plt.hist(delta_win_type_1, bins=bins, alpha=0.7, label='Policy Type 1')
# plt.hist(delta_win_type_2, bins=bins, alpha=0.7, label='Policy Type 2', color='orange')
# plt.title('Combined Histogram of Delta Win Types')
# plt.xlabel('Policy Value')
# plt.ylabel('Frequency')
# plt.legend()
#
# plt.tight_layout()
# plt.show()
#
# # Compute histograms for delta_win_type_1 and delta_win_type_2
# counts_1, _ = np.histogram(delta_win_type_1, bins=bins)
# counts_2, _ = np.histogram(delta_win_type_2, bins=bins)
#
# bin_av = (bins[0:199] + bins[1:200])/2
# # Calculate the difference
# delta_counts = (counts_1 - counts_2) * bin_av
#
# # Plotting the difference histogram
# plt.figure(figsize=(7, 5))
# plt.bar(bins[:-1], delta_counts, width=bins[1]-bins[0], alpha=0.7, label='Delta Win Type 1 - Delta Win Type 2')
# plt.title('Histogram of Delta Win Type Difference')
# plt.xlabel('Delta Win Value')
# plt.ylabel('Frequency Difference')
# plt.legend()
# plt.show()
