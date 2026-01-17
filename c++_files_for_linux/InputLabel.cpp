//#include "pch.h"
#include <iostream>
#include <vector>
#include <utility>
#include <stack>
#include <cstdlib>
#include <ctime>
#include <map>
#include <set>
#include <cmath>
#include <random>
#include <tuple>
#include <algorithm>
#include "InputLabel.h"


const int EMPTY = 0;

int visited[19][19] = { 0 };
int visit_number = 0;

const std::pair<int, int> DIRECTIONS[4] = {
    {-1, 0},  // Up
    {1, 0},   // Down
    {0, -1},  // Left
    {0, 1}    // Right
};

bool isInBoard(int x, int y, int size) {
    return x >= 0 && x < size&& y >= 0 && y < size;
}

void getGroup(int i, int j, int board[19][19], int* liberty_matrix[19][19], std::set<std::pair<int, int>>* group_matrix[19][19]) {
    std::stack<std::pair<int, int>> stack;

    visit_number++;
    stack.push({ i, j });
    std::set<std::pair<int, int>>* group = new std::set<std::pair<int, int>>();
    int* liberty = new int(0);

    while (!stack.empty()) {
        std::pair<int, int> pos = stack.top();
        stack.pop();

        if (board[pos.first][pos.second] == 0) {
            ++(*liberty);
            continue;
        }

        if (board[pos.first][pos.second] == board[i][j]) {
            group_matrix[pos.first][pos.second] = group;
            liberty_matrix[pos.first][pos.second] = liberty;

            group->insert({ pos.first, pos.second });

            if (pos.first < 18) {
                if (visited[pos.first + 1][pos.second] != visit_number) {
                    if (board[pos.first + 1][pos.second] == board[i][j]) { stack.push({ pos.first + 1, pos.second }); }
                    else if (board[pos.first + 1][pos.second] == 0) {
                        visited[pos.first + 1][pos.second] = visit_number;
                        ++(*liberty);
                    }

                }
            }
            if (pos.first > 0) {
                if (visited[pos.first - 1][pos.second] != visit_number) {
                    if (board[pos.first - 1][pos.second] == board[i][j]) { stack.push({ pos.first - 1, pos.second }); }
                    else if (board[pos.first - 1][pos.second] == 0) {
                        visited[pos.first - 1][pos.second] = visit_number;
                        ++(*liberty);
                    }
                }
            }
            if (pos.second < 18) {
                if (visited[pos.first][pos.second + 1] != visit_number) {
                    if (board[pos.first][pos.second + 1] == board[i][j]) { stack.push({ pos.first, pos.second + 1 }); }
                    else if (board[pos.first][pos.second + 1] == 0) {
                        visited[pos.first][pos.second + 1] = visit_number;
                        ++(*liberty);
                    }
                }
            }
            if (pos.second > 0) {
                if (visited[pos.first][pos.second - 1] != visit_number) {
                    if (board[pos.first][pos.second - 1] == board[i][j]) { stack.push({ pos.first, pos.second - 1 }); }
                    else if (board[pos.first][pos.second - 1] == 0) {
                        visited[pos.first][pos.second - 1] = visit_number;
                        ++(*liberty);
                    }
                }
            }
        }

        visited[pos.first][pos.second] = visit_number;
    }
}

void create_group_matrix(int board[19][19], std::set<std::pair<int, int>>* (&group_matrix)[19][19], int* (&liberty_matrix)[19][19]) {
    int i, j;
    std::stack<std::pair<int, int>> stack;

    for (i = 0; i < 19; ++i) {
        for (j = 0; j < 19; ++j) {
            group_matrix[i][j] = nullptr;
            liberty_matrix[i][j] = nullptr;
        }
    }

    for (i = 0; i < 19; ++i) {
        for (j = 0; j < 19; ++j) {
            if (board[i][j] == 0 || liberty_matrix[i][j] != nullptr) { continue; }
            visit_number++;
            stack.push({ i, j });
            std::set<std::pair<int, int>>* group = new std::set<std::pair<int, int>>();
            int* liberty = new int(0);

            while (!stack.empty()) {
                std::pair<int, int> pos = stack.top();
                stack.pop();

                if (board[pos.first][pos.second] == 0) {
                    ++(*liberty);
                    continue;
                }

                if (board[pos.first][pos.second] == board[i][j]) {
                    group_matrix[pos.first][pos.second] = group;
                    liberty_matrix[pos.first][pos.second] = liberty;

                    group->insert({ pos.first, pos.second });

                    if (pos.first < 18) {
                        if (visited[pos.first + 1][pos.second] != visit_number) {
                            if (board[pos.first + 1][pos.second] == board[i][j]) { stack.push({ pos.first + 1, pos.second }); }
                            else if (board[pos.first + 1][pos.second] == 0) {
                                visited[pos.first + 1][pos.second] = visit_number;
                                ++(*liberty);
                            }

                        }
                    }
                    if (pos.first > 0) {
                        if (visited[pos.first - 1][pos.second] != visit_number) {
                            if (board[pos.first - 1][pos.second] == board[i][j]) { stack.push({ pos.first - 1, pos.second }); }
                            else if (board[pos.first - 1][pos.second] == 0) {
                                visited[pos.first - 1][pos.second] = visit_number;
                                ++(*liberty);
                            }
                        }
                    }
                    if (pos.second < 18) {
                        if (visited[pos.first][pos.second + 1] != visit_number) {
                            if (board[pos.first][pos.second + 1] == board[i][j]) { stack.push({ pos.first, pos.second + 1 }); }
                            else if (board[pos.first][pos.second + 1] == 0) {
                                visited[pos.first][pos.second + 1] = visit_number;
                                ++(*liberty);
                            }
                        }
                    }
                    if (pos.second > 0) {
                        if (visited[pos.first][pos.second - 1] != visit_number) {
                            if (board[pos.first][pos.second - 1] == board[i][j]) { stack.push({ pos.first, pos.second - 1 }); }
                            else if (board[pos.first][pos.second - 1] == 0) {
                                visited[pos.first][pos.second - 1] = visit_number;
                                ++(*liberty);
                            }
                        }
                    }
                }

                visited[pos.first][pos.second] = visit_number;
            }
        }
    }
}

int get_liberties(int board[19][19], const std::set<std::pair<int, int>>& group) {
    int liberties = 0;
    int x, y, nx, ny;
    visit_number++;

    for (const auto& stone : group) {
        x = stone.first;
        y = stone.second;

        for (const auto& dir : DIRECTIONS) {
            nx = x + dir.first;
            ny = y + dir.second;



            if (isInBoard(nx, ny, 19) && visited[nx][ny] != visit_number) {
                if (board[nx][ny] == EMPTY) {
                    liberties++;
                }
                visited[nx][ny] = visit_number;
            }
        }
    }

    return liberties;
}

std::set<std::pair<int, int>> place_stone(int board[19][19], int* liberty_matrix[19][19], std::set<std::pair<int, int>>* group_matrix[19][19], std::pair<int, int> played_point, int color, bool ko, bool& encountered_ko, bool& move_legal, bool& multiple_merges) {
    int row = played_point.first;
    int col = played_point.second;

    // Check if the move is legal
    if (board[row][col] != 0) {
        move_legal = false;
        return {};
    }

    board[row][col] = color;
    int opponent_color = 3 - color;
    int num_merges = 0;
    int captures = 0;
    int nr, nc, r, c;
    std::set<std::pair<int, int>> captured_groups;
    std::set<std::set<std::pair<int, int>>*> group_pointers_to_delete;
    std::set<int*> liberty_pointers_to_delete;

    bool legal_move = false;
    bool surrounded_by_opponent_stones = true;
    int num_stones_captured = 0;

    for (const auto& dir : DIRECTIONS) {
        nr = row + dir.first;
        nc = col + dir.second;
        if (nr < 0 || nr >= 19 || nc < 0 || nc >= 19) {
            continue;
        }
        if (board[nr][nc] == opponent_color && (*liberty_matrix[nr][nc]) == 1) {
            captured_groups.insert(group_matrix[nr][nc]->begin(), group_matrix[nr][nc]->end());
            group_pointers_to_delete.insert(group_matrix[nr][nc]);
            liberty_pointers_to_delete.insert(liberty_matrix[nr][nc]);
            num_stones_captured += group_matrix[nr][nc]->size();
        }
        else if (board[nr][nc] == color) {
            surrounded_by_opponent_stones = false;
            if ((*liberty_matrix[nr][nc]) >= 2) {
                legal_move = true;
            }
        }
        else if (board[nr][nc] == 0) {
            surrounded_by_opponent_stones = false;
            legal_move = true;
        }
    }

    if (!legal_move) {
        if (num_stones_captured == 1 && surrounded_by_opponent_stones) {
            encountered_ko = true;
        }
        if (num_stones_captured == 0 || (num_stones_captured == 1 && ko && surrounded_by_opponent_stones)) {
            board[row][col] = 0;
            move_legal = false;
            return {};
        }
    }

    liberty_matrix[row][col] = new int(0);
    group_matrix[row][col] = new std::set<std::pair<int, int>>{ {row, col} };

    // If our played stone is placed beside another of our groups, merge the groups
    for (const auto& dir : DIRECTIONS) {
        nr = row + dir.first;
        nc = col + dir.second;
        if (nr < 0 || nr >= 19 || nc < 0 || nc >= 19) {
            continue;
        }
        if (board[nr][nc] == color && group_matrix[nr][nc] != group_matrix[row][col]) {
            group_matrix[nr][nc]->insert(group_matrix[row][col]->begin(), group_matrix[row][col]->end());    ////CHECK THIS LINE
            group_pointers_to_delete.insert(group_matrix[row][col]);
            liberty_pointers_to_delete.insert(liberty_matrix[row][col]);


            num_merges++;
            (*liberty_matrix[nr][nc]) += (*liberty_matrix[row][col]) - 1;

            for (const auto& p : *group_matrix[row][col]) {
                group_matrix[p.first][p.second] = group_matrix[nr][nc];
                liberty_matrix[p.first][p.second] = liberty_matrix[nr][nc];
            }
        }
    }

    // Get liberties of new group
    if (num_merges < 2) {
        for (const auto& dir : DIRECTIONS) {
            nr = row + dir.first;
            nc = col + dir.second;
            if (nr < 0 || nr >= 19 || nc < 0 || nc >= 19) {
                continue;
            }
            if (board[nr][nc] == 0) {
                std::vector<std::pair<int, int>> possible_self_group = { {nr - 1, nc}, {nr + 1, nc}, {nr, nc - 1}, {nr, nc + 1} };
                (*liberty_matrix[row][col])++;

                for (const auto& p : possible_self_group) {
                    r = p.first;
                    c = p.second;
                    if (r < 0 || r >= 19 || c < 0 || c >= 19) {
                        continue;
                    }
                    if (liberty_matrix[r][c] == liberty_matrix[row][col]) {
                        if (r != row || c != col) {
                            (*liberty_matrix[row][col])--;
                            break;
                        }
                    }
                }
            }
        }
    }
    else {
        *liberty_matrix[row][col] = get_liberties(board, *group_matrix[row][col]);
    }

    move_legal = true;
    multiple_merges = (num_merges > 1);

    // Remove the captured stones from the board
    if (!captured_groups.empty()) {
        for (const auto& p : captured_groups) {
            r = p.first;
            c = p.second;
            board[r][c] = 0;

            // Update the liberties of neighboring friendly groups
            std::set<std::pair<int, int>> groups_updated;
            for (const auto& dir : DIRECTIONS) {
                nr = r + dir.first;
                nc = c + dir.second;
                if (nr < 0 || nr >= 19 || nc < 0 || nc >= 19) {
                    continue;
                }
                if (board[nr][nc] == color && !groups_updated.count({ nr, nc })) {
                    (*liberty_matrix[nr][nc])++;
                    groups_updated.insert(group_matrix[nr][nc]->begin(), group_matrix[nr][nc]->end());
                }
            }

            group_matrix[r][c] = nullptr;
            liberty_matrix[r][c] = nullptr;
        }
    }

    // Update the liberties of neighboring opponent groups
    std::set<std::pair<int, int>> opponent_groups_updated;
    for (const auto& dir : DIRECTIONS) {
        nr = row + dir.first;
        nc = col + dir.second;
        if (nr < 0 || nr >= 19 || nc < 0 || nc >= 19) {
            continue;
        }
        if (board[nr][nc] == opponent_color) {
            if (!opponent_groups_updated.count({ nr, nc })) {
                (*liberty_matrix[nr][nc])--;
                opponent_groups_updated.insert(group_matrix[nr][nc]->begin(), group_matrix[nr][nc]->end());
            }
        }
    }

    for (auto it = group_pointers_to_delete.begin(); it != group_pointers_to_delete.end(); ) {
        delete* it;  // Delete the memory the pointer points to.
        it = group_pointers_to_delete.erase(it);  // Erase the pointer from the set and get the next iterator.
    }
    for (auto it = liberty_pointers_to_delete.begin(); it != liberty_pointers_to_delete.end(); ) {
        delete* it;  // Delete the memory the pointer points to.
        it = liberty_pointers_to_delete.erase(it);  // Erase the pointer from the set and get the next iterator.
    }

    move_legal = true;
    multiple_merges = num_merges >= 2;

    return captured_groups;
}

void undo_move(int board[19][19], std::set<std::pair<int, int>>& captures, int* liberty_matrix[19][19], std::set<std::pair<int, int>>* group_matrix[19][19], bool double_merge, std::pair<int, int> played_point) {
    int row = played_point.first;
    int col = played_point.second;
    int color = board[row][col];
    int opponent_color = 3 - color;
    board[row][col] = 0;
    int nr, nc, r, c;

    std::set<int*> visited_liberties;
    std::set<std::pair<int, int>> groups;
    for (auto& dir : DIRECTIONS) {
        nr = row + dir.first;
        nc = col + dir.second;
        if (nr >= 0 && nr < 19 && nc >= 0 && nc < 19) {
            if (board[nr][nc] != opponent_color) { continue; }
            if (visited_liberties.find(liberty_matrix[nr][nc]) != visited_liberties.end()) { continue; }
            visited_liberties.insert(liberty_matrix[nr][nc]);
            groups.insert({ nr, nc });
        }
    }

    for (auto& g : groups) {
        nr = g.first;
        nc = g.second;
        (*liberty_matrix[nr][nc])++;
    }

    for (auto stone : captures) {
        board[stone.first][stone.second] = opponent_color;
    }

    int liberty_for_capture[1][1] = { {1} };


    for (auto stone : captures) {
        groups.clear();
        std::set<std::pair<int, int>> groups_updated;

        for (auto& dir : DIRECTIONS) {
            nr = stone.first + dir.first;
            nc = stone.second + dir.second;
            if (nr >= 0 && nr < 19 && nc >= 0 && nc < 19) {
                if (board[nr][nc] == color && !groups_updated.count({ nr, nc })) {
                    groups_updated.insert(group_matrix[nr][nc]->begin(), group_matrix[nr][nc]->end());
                    groups.insert({ nr, nc });
                }
            }
        }

        for (auto stone : groups) {
            (*liberty_matrix[stone.first][stone.second])--;
        }
    }

    for (auto stone : captures) {
        if (liberty_matrix[stone.first][stone.second] == nullptr || *liberty_matrix[stone.first][stone.second] == 0) {
            getGroup(stone.first, stone.second, board, liberty_matrix, group_matrix);
        }
    }

    if (double_merge) {
        if (liberty_matrix[row][col] != nullptr) {
            delete liberty_matrix[row][col];
            liberty_matrix[row][col] = nullptr;
        }
        auto old_group = group_matrix[row][col];
        group_matrix[row][col] = nullptr;

        for (auto& dir : DIRECTIONS) {
            nr = row + dir.first;
            nc = col + dir.second;
            if (nr >= 0 && nr < 19 && nc >= 0 && nc < 19) {
                if (board[nr][nc] == color && group_matrix[nr][nc] == old_group) {
                    getGroup(nr, nc, board, liberty_matrix, group_matrix);
                }
            }
        }
    }
    else if ((*group_matrix[row][col]).size() == 1) {
        delete liberty_matrix[row][col];
        liberty_matrix[row][col] = nullptr;
        delete group_matrix[row][col];
        group_matrix[row][col] = nullptr;
    }
    else {
        (*liberty_matrix[row][col])++;
        for (auto& dir : DIRECTIONS) {
            nr = row + dir.first;
            nc = col + dir.second;
            if (nr >= 0 && nr < 19 && nc >= 0 && nc < 19) {
                if (board[nr][nc] == 0 || captures.count(std::make_pair(nr, nc))) {
                    for (auto& adj_dir : DIRECTIONS) {
                        r = nr + adj_dir.first;
                        c = nc + adj_dir.second;
                        if (r < 0 || r >= 19 || c < 0 || c >= 19) {
                            continue;
                        }
                        if (board[r][c] == color && group_matrix[r][c] == group_matrix[row][col]) {
                            (*liberty_matrix[row][col])++;
                            break;
                        }
                    }
                    (*liberty_matrix[row][col])--;
                }
            }
        }

        liberty_matrix[row][col] = nullptr;

        // Remove the point row, col from the set *group_matrix[row][col]
        group_matrix[row][col]->erase(std::make_pair(row, col));
    }
}

void get_captures(int board[19][19], int* liberty_matrix[19][19], std::set<std::pair<int, int>>* group_matrix[19][19], std::pair<int, int> played_point, std::set<std::pair<int, int>>& capture_locations) {
    int row = played_point.first;
    int col = played_point.second;
    int nr, nc, r, c;
    if (board[row][col] == 0 || *liberty_matrix[row][col] > 1) {
        return;
    }

    // Find the point that captures the group
    std::set<std::pair<int, int>> group_stones = *group_matrix[row][col];
    for (auto stone : group_stones) {
        r = stone.first;
        c = stone.second;
        for (auto& dir : DIRECTIONS) {
            nr = r + dir.first;
            nc = c + dir.second;
            if (nr >= 0 && nr < 19 && nc >= 0 && nc < 19 && board[nr][nc] == 0) {
                capture_locations.insert(std::make_pair(nr, nc));
                return;
            }
        }
    }
}

void get_captures_of_neighbors(int board[19][19], int* liberty_matrix[19][19], std::set<std::pair<int, int>>* group_matrix[19][19], std::pair<int, int> group_point, std::set<std::pair<int, int>>& capture_locations) {
    int row = group_point.first;
    int col = group_point.second;
    int color = board[row][col];
    int opponent_color = 3 - color;
    int r, c, nr, nc, adj_r, adj_c;

    std::set<std::set<std::pair<int, int>>*> visited_groups;
    std::set<std::pair<int, int>> group_coords;

    std::set<std::pair<int, int>> group = *group_matrix[row][col];

    for (auto stone : group) {
        r = stone.first;
        c = stone.second;
        for (auto& dir : DIRECTIONS) {
            nr = r + dir.first;
            nc = c + dir.second;
            if (nr >= 0 && nr < 19 && nc >= 0 && nc < 19) {
                if (board[nr][nc] == opponent_color && *liberty_matrix[nr][nc] == 1 && visited_groups.count(group_matrix[nr][nc]) == 0) {
                    group_coords.insert(std::make_pair(nr, nc));
                    visited_groups.insert(group_matrix[nr][nc]);
                }
            }
        }
    }

    for (auto coord : group_coords) {
        r = coord.first;
        c = coord.second;
        for (auto stone : *group_matrix[r][c]) {
            nr = stone.first;
            nc = stone.second;
            for (auto& dir : DIRECTIONS) {
                adj_r = nr + dir.first;
                adj_c = nc + dir.second;
                if (adj_r >= 0 && adj_r < 19 && adj_c >= 0 && adj_c < 19) {
                    if (board[adj_r][adj_c] == 0) {
                        capture_locations.insert(std::make_pair(adj_r, adj_c));
                        continue;
                    }
                }
            }
        }
    }
}

int quick_check(int board[19][19], std::set<std::pair<int, int>>& liberty_locations, std::pair<int, int>& point1, std::pair<int, int>& point2) {
    auto it = liberty_locations.begin();
    point1 = *it++;
    point2 = *it;

    int r1 = point1.first;
    int c1 = point1.second;
    int r2 = point2.first;
    int c2 = point2.second;

    // Check if points are adjacent
    bool are_adjacent = std::abs(r1 - r2) + std::abs(c1 - c2) == 1;

    int empty_neighbors1 = 0;
    int empty_neighbors2 = 0;

    for (auto& dir : DIRECTIONS) {
        int nr1 = r1 + dir.first;
        int nc1 = c1 + dir.second;
        int nr2 = r2 + dir.first;
        int nc2 = c2 + dir.second;

        // Count the empty neighbors for both points
        if (nr1 >= 0 && nr1 < 19 && nc1 >= 0 && nc1 < 19 && board[nr1][nc1] == 0) ++empty_neighbors1;
        if (nr2 >= 0 && nr2 < 19 && nc2 >= 0 && nc2 < 19 && board[nr2][nc2] == 0) ++empty_neighbors2;
    }

    // Determine which points to return
    if (!are_adjacent) {
        if (empty_neighbors1 >= 3 && empty_neighbors2 >= 3) {
            return 0;
        }
        else if (empty_neighbors1 >= 3) {
            return 1;
        }
        else if (empty_neighbors2 >= 3) {
            std::swap(point1, point2);
            return 1;
        }
        else {
            return 2;
        }
    }
    else {
        return 2;
    }
}

bool play_ladder(int board[19][19], int* liberty_matrix[19][19], std::set<std::pair<int, int>>* group_matrix[19][19], std::pair<int, int> ladder_point, std::vector<std::pair<int, int>> initial_liberty_locations, bool& encoutered_ko, int ko_for_defender, int& max_depth, int& counter) {
    int row = ladder_point.first;
    int col = ladder_point.second;
    int defender_color = board[row][col];
    int attacker_color = 3 - board[row][col];
    int side_to_move = attacker_color;
    int arrSize = 450;
    bool double_merge_flag = false;
    bool move_was_legal, defender_to_move, ko;

    int depth = 0;
    bool new_position = false;
    int r, c, nr, nc, r1, c1, r2, c2, num_good_moves;
    std::pair<int, int> opponent_last_move, rc, move1, move2;
    std::set<std::pair<int, int>> joined_groups, liberties, captures, captured_stones;

    std::vector<int> move_id(450, 0); // An array of 450 ints initialized to 0
    std::vector<std::vector<std::pair<int, int>>> move_stack(450); // An array of 450 empty vectors of pairs of ints
    std::vector<std::vector<std::pair<int, int>>> liberty_locations(450); // An array of 450 empty vectors of pairs of ints
    std::vector<std::set<std::pair<int, int>>> capture_locations(450); // An array of 450 empty sets of pairs of ints
    std::vector<bool> merge(450, false); // An array of 450 bools initialized to false
    std::vector<std::vector<bool>> result(450); // An array of 450 empty vectors of bools

    move_stack[0] = initial_liberty_locations;

    int defender_plays_first;
    if (initial_liberty_locations.size() == 1) { defender_plays_first = 1; }
    if (initial_liberty_locations.size() == 2) { defender_plays_first = 0; }

    if (defender_plays_first == 1) { new_position = true; }

    while (true) {
        counter++;
        defender_to_move = depth % 2 != defender_plays_first;
        ko = defender_to_move == ko_for_defender;

        if (new_position) {
            result[depth + 1] = {};

            if (defender_to_move) {
                captures.clear();
                if (depth >= 2 && move_stack[depth - 2].size() == 1) {
                    opponent_last_move = move_stack[depth - 1][move_id[depth - 1] - 1];
                    get_captures(board, liberty_matrix, group_matrix, opponent_last_move, captures);

                    if (depth >= 2) {
                        get_captures_of_neighbors(board, liberty_matrix, group_matrix, ladder_point, captures);
                    }
                }
                else {
                    get_captures_of_neighbors(board, liberty_matrix, group_matrix, ladder_point, captures);
                }

                if (depth != 0) {
                    rc = move_stack[depth - 1][move_id[depth - 1] % 2];
                    captures.erase(rc);
                    move_stack[depth] = std::vector<std::pair<int, int>>(captures.begin(), captures.end());
                    move_stack[depth].push_back(move_stack[depth - 1][move_id[depth - 1] % 2]);
                }

                else {
                    move_stack[0] = std::vector<std::pair<int, int>>(captures.begin(), captures.end());
                    move_stack[0].insert(move_stack[depth].end(), initial_liberty_locations.begin(), initial_liberty_locations.end());
                }

            }
            else { // attacker to move
                liberties.clear();
                joined_groups.clear();
                if (move_stack[depth - 1].size() == move_id[depth - 1]) { // The defender played on their liberty point last move
                    rc = move_stack[depth - 1][move_id[depth - 1] - 1]; // The defender's last move
                    r = rc.first;
                    c = rc.second;

                    for (auto& direction : DIRECTIONS) {
                        nr = r + direction.first;
                        nc = c + direction.second;
                        if (nr < 0 || nr >= 19 || nc < 0 || nc >= 19) {
                            continue;
                        }
                        if (board[nr][nc] == 0) {
                            liberties.insert({ nr, nc });
                        }
                        else if (board[nr][nc] == defender_color) {
                            joined_groups.insert({ nr, nc });
                        }
                    }

                    if (liberties.size() < 2) {
                        for (auto& stone : *group_matrix[r][c]) {
                            r1 = stone.first;
                            c1 = stone.second;
                            for (auto& direction : DIRECTIONS) {
                                r2 = r1 + direction.first;
                                c2 = c1 + direction.second;
                                if (r2 < 0 || r2 >= 19 || c2 < 0 || c2 >= 19) {
                                    continue;
                                }
                                if (board[r2][c2] == 0) {
                                    liberties.insert({ r2, c2 });
                                }
                            }
                        }
                    }
                }
                else {
                    liberties.clear();
                    captured_stones = capture_locations[depth - 1];
                    for (auto& stone : captured_stones) {
                        r = stone.first;
                        c = stone.second;
                        for (auto& direction : DIRECTIONS) {
                            r2 = r + direction.first;
                            c2 = c + direction.second;
                            if (r2 < 0 || r2 >= 19 || c2 < 0 || c2 >= 19) {
                                continue;
                            }
                            if (group_matrix[r2][c2] == group_matrix[row][col]) {
                                liberties.insert({ r, c });
                                continue;
                            }
                        }
                    }
                    liberties.insert(move_stack[depth - 1].back());
                }

                move1, move2;

                num_good_moves = quick_check(board, liberties, move1, move2);

                if (num_good_moves == 1) {
                    move_id[depth] = 1;
                }
                else if (num_good_moves == 0) {
                    move_id[depth] = 2;
                }

                move_stack[depth] = { move2, move1 };
            }
            new_position = false;
        }
        else { // This means new_position is false, so an old position. Maybe the last move played was good and we can return
            if (!result[depth + 1].empty() && defender_to_move != result[depth + 1].back()) {
                move_id[depth] = 1000;
            }
        }

        // Play the next move
        if (move_id[depth] < move_stack[depth].size()) {

            std::pair<int, int> move_to_play = move_stack[depth][move_id[depth]];
            std::set<std::pair<int, int>> captured_stones;

            if (defender_to_move) {
                captured_stones = place_stone(board, liberty_matrix, group_matrix, move_to_play, defender_color, ko, encoutered_ko, move_was_legal, double_merge_flag);
            }
            else {
                captured_stones = place_stone(board, liberty_matrix, group_matrix, move_to_play, attacker_color, ko, encoutered_ko, move_was_legal, double_merge_flag);
            }

            move_id[depth] += 1;

            if (!move_was_legal) {
                continue;
            }

            capture_locations[depth] = captured_stones;
            merge[depth] = double_merge_flag;

            // If this player has run out of moves, they have failed
        }
        else {
            bool outcome;

            if (defender_to_move) {
                outcome = true;

                for (auto r : result[depth + 1]) {
                    if (!r) {
                        outcome = false;
                        break;
                    }
                }
            }
            else { // attacker_to_move
                outcome = false;

                for (auto r : result[depth + 1]) {
                    if (r) {
                        outcome = true;
                        break;
                    }
                }
            }

            if (depth == 0) {
                return outcome;
            }

            if (defender_to_move == outcome) {
                result[depth - 1].push_back(outcome);
                depth -= 1;
                undo_move(board, capture_locations[depth], liberty_matrix, group_matrix, merge[depth], move_stack[depth][move_id[depth] - 1]);
            }
            else {
                result[depth].push_back(outcome);
            }

            if (depth == 0) {
                return outcome;
            }

            depth -= 1;
            undo_move(board, capture_locations[depth], liberty_matrix, group_matrix, merge[depth], move_stack[depth][move_id[depth] - 1]);
            continue;
        }

        if (depth == 0 && defender_plays_first == 0 && *liberty_matrix[row][col] >= 2) {
            undo_move(board, capture_locations[depth], liberty_matrix, group_matrix, merge[depth],
                move_stack[depth][move_id[depth] - 1]);
            result[depth].push_back(false);
            continue;
        }

        if (depth == 0 && defender_plays_first == 1 && *liberty_matrix[row][col] <= 1) {
            undo_move(board, capture_locations[depth], liberty_matrix, group_matrix, merge[depth],
                move_stack[depth][move_id[depth] - 1]);
            result[depth].push_back(true);
            continue;
        }

        if (defender_to_move) {
            if (*liberty_matrix[row][col] == 1) {
                undo_move(board, capture_locations[depth], liberty_matrix, group_matrix, merge[depth], move_stack[depth][move_id[depth] - 1]);
                new_position = false;
                result[depth].push_back(true);
                depth -= 1;
                undo_move(board, capture_locations[depth], liberty_matrix, group_matrix, merge[depth], move_stack[depth][move_id[depth] - 1]);
                continue;
            }

            if (*liberty_matrix[row][col] >= 3) {
                result[depth + 1].push_back(false);
                undo_move(board, capture_locations[depth], liberty_matrix, group_matrix, merge[depth], move_stack[depth][move_id[depth] - 1]);
                continue;
            }
        }
        else {
            if (*liberty_matrix[row][col] >= 2) {
                undo_move(board, capture_locations[depth], liberty_matrix, group_matrix, merge[depth],
                    move_stack[depth][move_id[depth] - 1]);
                result[depth].push_back(false);
                depth -= 1;
                new_position = false;
                undo_move(board, capture_locations[depth], liberty_matrix, group_matrix, merge[depth],
                    move_stack[depth][move_id[depth] - 1]);
                continue;
            }
            else if (*liberty_matrix[row][col] == 0) {
                undo_move(board, capture_locations[depth], liberty_matrix, group_matrix, merge[depth],
                    move_stack[depth][move_id[depth] - 1]);
                result[depth + 1].push_back(true);
                continue;
            }
        }

        depth += 1;
        if (depth > max_depth) {
            max_depth = depth;
        }
        new_position = true;
        result[depth].clear();
        move_id[depth] = 0;
    }
}

double find_ladder(int board[19][19], int* liberty_matrix[19][19], std::set<std::pair<int, int>>* group_matrix[19][19], int row, int col, bool& ladder_success_1, bool& ladder_success_2) {
    int opponent_color = 3 - board[row][col];
    visit_number++;
    std::vector<std::pair<int, int>> liberty_locations;

    // Get the liberties and liberty_locations for the group at (row, col)

    // Find the liberty locations for play_ladder
    std::set<std::pair<int, int>> group_coords = *group_matrix[row][col];
    for (auto stone : group_coords) {
        std::vector<std::pair<int, int>> neighbors = { {stone.first - 1, stone.second}, {stone.first + 1, stone.second}, {stone.first, stone.second - 1}, {stone.first, stone.second + 1} };
        for (auto coord : neighbors) {
            if (coord.first < 0 || coord.first >= 19 || coord.second < 0 || coord.second >= 19) {
                continue;
            }
            if (board[coord.first][coord.second] == 0 && visited[coord.first][coord.second] != visit_number) {
                liberty_locations.push_back({ coord.first, coord.second });
                visited[coord.first][coord.second] = visit_number;
            }
        }
    }

    // Call play_ladder with the calculated parameters
    int max_depth_1 = 0, max_depth_2 = 0, counter_1 = 0, counter_2 = 0;
    bool encoutered_ko = false;
    ladder_success_1 = play_ladder(board, liberty_matrix, group_matrix, std::make_pair(row, col), liberty_locations, encoutered_ko, 1, max_depth_1, counter_1);
    ladder_success_2 = ladder_success_1;
    max_depth_2 = 0;
    if (encoutered_ko && ladder_success_1) {
        ladder_success_2 = play_ladder(board, liberty_matrix, group_matrix, std::make_pair(row, col), liberty_locations, encoutered_ko, 0, max_depth_2, counter_2);
    }
    return (std::max)(max_depth_1, max_depth_2);
}

void ladder_matrix(int board[19][19], int* liberty_matrix[19][19], std::set<std::pair<int, int>>* group_matrix[19][19],
    int ladder_matrix_attacker_advantage[19][19], int ladder_matrix_defender_advantage[19][19], int ladder_success_depth[19][19], int ladder_fail_depth[19][19]) {
    std::set<int*> visited_liberties;
    bool ladder_success_1 = false;
    bool ladder_success_2 = false;

    for (int x = 0; x < 19; ++x) {
        for (int y = 0; y < 19; ++y) {
            if (board[x][y] > 0 && (*liberty_matrix[x][y] == 2 || *liberty_matrix[x][y] == 1)) {
                if (visited_liberties.find(liberty_matrix[x][y]) != visited_liberties.end()) { continue; }
                visited_liberties.insert(liberty_matrix[x][y]);

                int depth = find_ladder(board, liberty_matrix, group_matrix, x, y, ladder_success_1, ladder_success_2);


                if (ladder_success_1) {
                    for (const auto& coord : *group_matrix[x][y]) {
                        ladder_matrix_attacker_advantage[coord.first][coord.second] = 1;
                    }
                }

                if (ladder_success_2) {
                    for (const auto& coord : *group_matrix[x][y]) {
                        ladder_matrix_defender_advantage[coord.first][coord.second] = 1;
                    }
                }

                if (ladder_success_1 || ladder_success_2) {
                    for (const auto& coord : *group_matrix[x][y]) {
                        ladder_success_depth[coord.first][coord.second] = depth;
                    }
                }
                else {
                    for (const auto& coord : *group_matrix[x][y]) {
                        ladder_fail_depth[coord.first][coord.second] = depth;
                    }
                }
            }
        }
    }
}

void find_kos(int board[19][19], int liberty_matrix[19][19], std::set<std::pair<int, int>>* group_matrix_pointer[19][19], std::pair<int, int> last_move, int last_move_was_a_capture, int ko_matrix[19][19], int potential_ko_matrix[19][19]) {
    int surround[19][19] = { 0 };
    int x, y, x2, y2, opponent_color, num_of_captures, not_opponent_stones;
    bool actual_ko;

    for (x = 0; x < 19; ++x) {
        for (y = 0; y < 19; ++y) {
            opponent_color = 0;
            num_of_captures = 0;
            not_opponent_stones = 0;
            actual_ko = false;

            if (board[x][y] != 0) { continue; }

            for (auto const& direction : DIRECTIONS) {
                x2 = x + direction.first;
                y2 = y + direction.second;

                if (x2 < 0 || x2 >= 19 || y2 < 0 || y2 >= 19) { continue; }
                if (board[x2][y2] == 0) {
                    not_opponent_stones++;
                    continue;
                }

                if (opponent_color == 0) {
                    opponent_color = board[x2][y2];
                }

                if (board[x2][y2] == 3 - opponent_color) {
                    not_opponent_stones = 10;
                    break; // Can't be ko if beside a friendly stone
                }

                if (liberty_matrix[x2][y2] == 1) {
                    num_of_captures += group_matrix_pointer[x2][y2]->size();

                    if (last_move.first == x2 && last_move.second == y2 and last_move_was_a_capture == 1) {
                        actual_ko = true;
                    }
                }
            }

            if (not_opponent_stones == 1 && num_of_captures == 0) {
                surround[x][y] = opponent_color;
            }

            else if (not_opponent_stones == 0 && num_of_captures == 1) {
                surround[x][y] = opponent_color;

                if (actual_ko) {
                    ko_matrix[x][y] = 1;
                }
            }
        }
    }

    for (x = 0; x < 19; ++x) {
        for (y = 0; y < 19; ++y) {
            for (auto const& direction : DIRECTIONS) {
                x2 = x + direction.first;
                y2 = y + direction.second;

                if (x2 < 0 || x2 >= 19 || y2 < 0 || y2 >= 19) { continue; }

                if (surround[x][y] * surround[x2][y2] == 2) {
                    potential_ko_matrix[x][y] = 1;
                }
            }
        }
    }
}

void fill_liberties(int board[19][19]) {
    for (int i = 0; i < 19; ++i) {
        for (int j = 0; j < 19; ++j) {
            if (board[i][j] == 1) { continue; }
            for (auto& d : DIRECTIONS) {
                if (i + d.first < 0 || i + d.first >= 19 || j + d.second < 0 || j + d.second >= 19) { continue; }
                if (board[i + d.first][j + d.second] == 1) {
                    board[i][j] = 2;
                }
            }
        }
    }
}

void remove_captured(int board[19][19], std::set<std::pair<int, int>>* group_matrix[19][19], int* liberty_matrix[19][19], int colour) {
    for (int i = 0; i < 19; ++i) {
        for (int j = 0; j < 19; ++j) {
            if (board[i][j] != colour) { continue; }
            if (*liberty_matrix[i][j] != 0) { continue; }

            std::set<int*> visited_liberties;


            for (auto& stone : *group_matrix[i][j]) {
                board[stone.first][stone.second] = 0;

                for (auto& d : DIRECTIONS) {
                    if (stone.first + d.first < 0 || stone.first + d.first >= 19 || stone.second + d.second < 0 || stone.second + d.second >= 19) { continue; }
                    if (visited_liberties.find(liberty_matrix[stone.first + d.first][stone.second + d.second]) != visited_liberties.end()) { continue; }
                    visited_liberties.insert(liberty_matrix[stone.first + d.first][stone.second + d.second]);
                    (*liberty_matrix[stone.first + d.first][stone.second + d.second])++;
                }
            }
        }
    }
}

void remove_immediate_captured(int board[19][19], std::set<std::pair<int, int>>* group_matrix[19][19], int* liberty_matrix[19][19], int colour) {
    for (int i = 0; i < 19; ++i) {
        for (int j = 0; j < 19; ++j) {
            if (board[i][j] != colour) { continue; }
            if (*liberty_matrix[i][j] != 0) { continue; }

            for (auto& stone : *group_matrix[i][j]) {
                board[stone.first][stone.second] = 0;
            }
        }
    }
}

void remove_atari(int board[19][19], std::set<std::pair<int, int>>* group_matrix[19][19], int* liberty_matrix[19][19]) {
    for (int i = 0; i < 19; ++i) {
        for (int j = 0; j < 19; ++j) {
            if (board[i][j] != 1) { continue; }
            if (*liberty_matrix[i][j] > 1) { continue; }

            visit_number++;

            for (auto& stone : *group_matrix[i][j]) {
                board[stone.first][stone.second] = 0;

                for (auto& d : DIRECTIONS) {
                    if (stone.first + d.first < 0 || stone.first + d.first >= 19 || stone.second + d.second < 0 || stone.second + d.second >= 19) { continue; }
                    if (board[stone.first + d.first][stone.second + d.second] != 0) { continue; }
                    if (visited[stone.first + d.first][stone.second + d.second] == visit_number) { continue; }
                    visited[stone.first + d.first][stone.second + d.second] = visit_number;


                    board[stone.first + d.first][stone.second + d.second] = 2;
                    for (auto& v : DIRECTIONS) {
                        if (stone.first + d.first + v.first < 0 || stone.first + d.first + v.first >= 19 || stone.second + d.second + v.second < 0 || stone.second + d.second + v.second >= 19) { continue; }
                        if (board[stone.first + d.first + v.first][stone.second + d.second + v.second] != 1) { continue; }

                        (*liberty_matrix[stone.first + d.first + v.first][stone.second + d.second + v.second])--;
                    }
                }
            }
        }
    }
}

void alive_groups(int board[19][19], std::set<std::pair<int, int>>* group_matrix[19][19], int* liberty_matrix[19][19], int two_eye_groups[19][19]) {
    int* new_liberty_matrix[19][19];
    std::set<std::pair<int, int>>* new_group_matrix[19][19];

    remove_immediate_captured(board, group_matrix, liberty_matrix, 1);
    fill_liberties(board);
    create_group_matrix(board, new_group_matrix, new_liberty_matrix);
    remove_captured(board, new_group_matrix, new_liberty_matrix, 2);
    remove_atari(board, new_group_matrix, new_liberty_matrix);
    remove_atari(board, new_group_matrix, new_liberty_matrix);

    std::set<int*> deleted;
    for (int i = 0; i < 19; ++i) {
        for (int j = 0; j < 19; ++j) {
            if (new_liberty_matrix[i][j] != nullptr && deleted.find(new_liberty_matrix[i][j]) == deleted.end()) {
                deleted.insert(new_liberty_matrix[i][j]);
                delete new_liberty_matrix[i][j];
                delete new_group_matrix[i][j];
            }
        }
    }

    fill_liberties(board);
    create_group_matrix(board, new_group_matrix, new_liberty_matrix);
    remove_captured(board, new_group_matrix, new_liberty_matrix, 2);


    for (int i = 0; i < 19; ++i) {
        for (int j = 0; j < 19; ++j) {
            if (board[i][j] == 1 and *new_liberty_matrix[i][j] >= 2) {
                two_eye_groups[i][j] = 1;
            }
        }
    }

    deleted.clear();
    for (int i = 0; i < 19; ++i) {
        for (int j = 0; j < 19; ++j) {
            if (new_liberty_matrix[i][j] != nullptr && deleted.find(new_liberty_matrix[i][j]) == deleted.end()) {
                deleted.insert(new_liberty_matrix[i][j]);
                delete new_liberty_matrix[i][j];
                delete new_group_matrix[i][j];
            }
        }
    }

    // Fill liberties
    // create_group_matrix
    // Remove opponent captured groups and increase adjacent liberties by one
    // Remove our stones in 'atari'
    // For each opponent stone we place during this, decrease any of our adjacent libs by one

    // Remove our stones in 'atari'
    // For each opponent stone we place during this, decrease any of our adjacent libs by one

    // Remaining stones are alive with more than 0 libs are alive
}

void relative_liberties(int board[19][19], std::set<std::pair<int, int>>* group_matrix[19][19], int* liberty_matrix[19][19], int data[10][19][19]) {
    //Initialize visted_liberties as a set
    int  i, j;
    for (i = 0; i < 19; ++i) {
        for (j = 0; j < 19; ++j) {
            if (liberty_matrix[i][j] == nullptr) { continue; }
            data[6][i][j] = *liberty_matrix[i][j];
            data[7][i][j] = group_matrix[i][j]->size();
        }
    }
}

void find_legal_moves(int board[19][19], int* liberty_matrix[19][19], int kos[19][19], int legal_moves[19][19], std::pair<int, int> last_move) {
    int row, col;
    int opponent_colour = board[last_move.first][last_move.second];

    for (int i = 0; i < 19; ++i) {
        for (int j = 0; j < 19; ++j) {
            if (board[i][j] != 0) { continue; }     // Can't play on non-empty point
            if (kos[i][j] == 1) { continue; }       // Ko rule prohibits this play

            for (auto& dir : DIRECTIONS) {
                row = i + dir.first;
                col = j + dir.second;
                if (row < 0 || row >= 19 || col < 0 || col >= 19) { continue; }

                // A move is legal if an adjacent point is empty, it captures stones, or if it connects to a group with 2 or more liberties

                if (board[row][col] == 0) {
                    legal_moves[i][j] = 1;
                    break;
                }

                else if (board[row][col] == opponent_colour && *liberty_matrix[row][col] == 1) {
                    legal_moves[i][j] = 1;
                    break;
                }

                else if (board[row][col] == 3 - opponent_colour && *liberty_matrix[row][col] >= 2) {
                    legal_moves[i][j] = 1;
                    break;
                }
            }

        }
    }
}

void connection_points(int board[19][19], std::set<std::pair<int, int>>* group_matrix[19][19], int* liberty_matrix[19][19], int black_connect[19][19], int white_connect[19][19]) {
    int row, col, group_size_sum, group_size_max;

    for (int i = 0; i < 19; ++i) {
        for (int j = 0; j < 19; ++j) {
            if (board[i][j] != 0) { continue; }

            group_size_max = 0;
            group_size_sum = 0;
            std::set<int*> visited_liberties;

            for (auto& dir : DIRECTIONS) {
                row = i + dir.first;
                col = j + dir.second;
                if (row < 0 || row >= 19 || col < 0 || col >= 19) { continue; }
                if (board[row][col] != 1) { continue; }
                if (visited_liberties.find(liberty_matrix[row][col]) != visited_liberties.end()) { continue; }
                visited_liberties.insert(liberty_matrix[row][col]);

                group_size_sum = group_size_sum + group_matrix[row][col]->size();

                if (group_size_max < group_matrix[row][col]->size()) { group_size_max = group_matrix[row][col]->size(); }
            }

            black_connect[i][j] = group_size_sum - group_size_max;
        }
    }

    for (int i = 0; i < 19; ++i) {
        for (int j = 0; j < 19; ++j) {
            if (board[i][j] != 0) { continue; }

            group_size_max = 0;
            group_size_sum = 0;
            std::set<int*> visited_liberties;

            for (auto& dir : DIRECTIONS) {
                row = i + dir.first;
                col = j + dir.second;
                if (row < 0 || row >= 19 || col < 0 || col >= 19) { continue; }
                if (board[row][col] != 2) { continue; }
                if (visited_liberties.find(liberty_matrix[row][col]) != visited_liberties.end()) { continue; }
                visited_liberties.insert(liberty_matrix[row][col]);

                group_size_sum = group_size_sum + group_matrix[row][col]->size();

                if (group_size_max < group_matrix[row][col]->size()) { group_size_max = group_matrix[row][col]->size(); }
            }

            white_connect[i][j] = group_size_sum - group_size_max;
        }
    }
}

void label_groups(int board[19][19], std::set<std::pair<int, int>>* group_matrix[19][19], int* liberty_matrix[19][19], int labels[19][19]) {
    int label_number = 0;
    for (int i = 0; i < 19; ++i) {
        for (int j = 0; j < 19; ++j) {
            if (board[i][j] == 0) { continue; }
            if (group_matrix[i][j]->size() <= 1) { continue; }
            if (labels[i][j] != 0) { continue; }
            
            label_number++;

            for (auto& stone : *group_matrix[i][j]) {
                labels[stone.first][stone.second] = label_number;
            }
            if (label_number >= 128){break;}
        }
        if (label_number >= 128){break;}
    }          
}

void create_inputs(int board[19][19], int data[13][19][19], int last_move_x, int last_move_y, int last_move_was_a_capture) {
    std::pair<int, int> last_move;
    last_move.first = last_move_x;
    last_move.second = last_move_y;
    int* liberty_matrix[19][19];
    std::set<std::pair<int, int>>* group_matrix[19][19];

    create_group_matrix(board, group_matrix, liberty_matrix);
    ladder_matrix(board, liberty_matrix, group_matrix, data[0], data[1], data[2], data[3]);
    relative_liberties(board, group_matrix, liberty_matrix, data);
    find_kos(board, data[6], group_matrix, last_move, last_move_was_a_capture, data[4], data[5]);
    find_legal_moves(board, liberty_matrix, data[4], data[8], last_move);
    connection_points(board, group_matrix, liberty_matrix, data[10], data[11]);
    label_groups(board, group_matrix, liberty_matrix, data[12]);
    

    // 0: ladder matrix where attacker has infinite ko threats
    // 1: ladder matrix where defender has infinite ko threats
    // 2: The max depth of working ladders 
    // 3: The max depth of failed ladders 
    // 4: Matrix showing the illegal move due to ko (if there is one)
    // 5: Matrix showing legal moves that would start a ko
    // 6: Number of liberties of each group
    // 7: Number of stones in each group
    // 8: Legal moves
    // 9: Groups that have two eyes
    // 10: Number of black stones that would connect if play here
    // 11: Number of white stones that would connect if play here
    // 12: Group Labels

    int black[19][19] = { 0 };
    int white[19][19] = { 0 };
    for (int i = 0; i < 19; ++i) {
        for (int j = 0; j < 19; ++j) {
            if (board[i][j] == 2) { white[i][j] = 1; }
            if (board[i][j] == 1) { black[i][j] = 1; }
        }
    }

    alive_groups(white, group_matrix, liberty_matrix, data[9]);
    alive_groups(black, group_matrix, liberty_matrix, data[9]);

    // Don't forget to free the memory
    std::set<int*> deleted;
    for (int i = 0; i < 19; ++i) {
        for (int j = 0; j < 19; ++j) {
            if (liberty_matrix[i][j] != nullptr && deleted.find(liberty_matrix[i][j]) == deleted.end()) {
                deleted.insert(liberty_matrix[i][j]);
                delete liberty_matrix[i][j];
                delete group_matrix[i][j];
            }
        }
    }
}