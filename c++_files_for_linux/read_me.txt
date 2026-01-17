This is the c++ code used by the model to generate input features. 
A precomiled windows version of this c++ code is included as a .dll file in the parent directory

This code can be compiled using <g++ -shared -o libInputLabel.so -fPIC InputLabel.cpp> to form a .so file