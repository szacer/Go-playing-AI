Contents:

Python Code:
Train.py - The code used to train the network

Find_losses.py - The code used to find the policy, value and ownership losses

Play_model_vs_katago.py - Model plays against KataGo. This requires KataGo to run on the computer, which can be tricky. 
Currently, a windows version of KataGo is contained within the folder katago-v1.12.4-eigen-windows-x64. 
This should work for most windows laptops. 

Game_reviewer.py - This produces some graphs from the 1000 game match played against KataGo. This data is contained within the summary file “model_new_cross_ent_step=268200kata_black_winrate_fix_playouts=0.5.txt”

C++ code: 
Some C++ code is used to generate input features for the network. 
This code is contained within the folder “c++_files_for_linux”. 
This compiles the Linux version of the c++ code. 
A precompiled windows version, InputLabel.dll, is already included in the folder. 

Other files:
model_new_cross_ent_step=268200.h5: This is the best trained weights for the model
small_training_data_file.pkl: Contains some training data
Finally there are two folders containing the games from the 1000 game match and the 200 game match between the model and KataGo. 
These contain README files to further explain the matches. 
