#!/bin/bash

# # Compile server
# echo "Compiling server..."
# nvcc server.cu -o server -lrt -lpthread -I/etc/pliops -L/etc/pliops -lstorelib -g

# # Check if server compilation was successful
# if [ $? -eq 0 ]; then
#     echo "Server compiled successfully."
# else
#     echo "Server compilation failed."
#     exit 1
# fi

# # Compile client
# echo "Compiling client..."
# nvcc client.cu -o client -lrt -lpthread -g

# # Check if client compilation was successful
# if [ $? -eq 0 ]; then
#     echo "Client compiled successfully."
# else
#     echo "Client compilation failed."
# fi

# Compile two_streams
# echo "Compiling two_streams..."
# nvcc two_streams.cu -o two_streams -lrt -lpthread -I/etc/pliops -L/etc/pliops -lstorelib -g -G -arch=sm_80 

# # Check if server compilation was successful
# if [ $? -eq 0 ]; then
#     echo "two_streams compiled successfully."
# else
#     echo "two_streams compilation failed."
#     exit 1
# fi

# Compile gdrcopy_poc
# echo "Compiling gdrcopy_poc..."
# nvcc gdrcopy_poc.cu gdrcopy_common.cpp -o gdrcopy_poc -lrt -lpthread -I/etc/pliops -L/etc/pliops -lstorelib -g -G -arch=sm_80 -lcuda -lgdrapi

# # Check if server compilation was successful
# if [ $? -eq 0 ]; then
#     echo "gdrcopy_poc compiled successfully."
# else
#     echo "gdrcopy_poc compilation failed."
#     exit 1
# fi

# Compile gdrcopy_pplat
echo "Compiling gdrcopy_pplat..."
nvcc gdrcopy_pplat.cu gdrcopy_common.cpp -o gdrcopy_pplat -lrt -lpthread -I/etc/pliops -L/etc/pliops -lstorelib -g -G -arch=sm_80 -lcuda -lgdrapi

# Check if server compilation was successful
if [ $? -eq 0 ]; then
    echo "gdrcopy_pplat compiled successfully."
else
    echo "gdrcopy_pplat compilation failed."
    exit 1
fi

# Compile ucf
# echo "Compiling unified_client_server (ucf)..."
# nvcc unified_client_server.cu -o ucf -lrt -lpthread -I/etc/pliops -L/etc/pliops -lstorelib -g -G -arch=sm_80

# # Check if server compilation was successful
# if [ $? -eq 0 ]; then
#     echo "ucf compiled successfully."
# else
#     echo "ucf compilation failed."
#     exit 1
# fi