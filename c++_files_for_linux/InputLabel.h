#pragma once

// Define INPUTFEATURES_API as empty for Linux
#define INPUTLABEL_API

extern "C" {
    INPUTLABEL_API void create_inputs(int board[19][19], int data[13][19][19], int last_move_x, int last_move_y, int last_move_was_a_capture);
}

//#pragma once

//#ifdef INPUTFEATURES_EXPORTS
//#define INPUTFEATURES_API __declspec(dllexport)
//#else
//#define INPUTFEATURES_API __declspec(dllimport)
//#endif

//extern "C" INPUTFEATURES_API void create_inputs(int board[19][19], int data[10][19][19], std::pair<int, int> last_move);