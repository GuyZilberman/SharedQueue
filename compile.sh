#!/bin/bash

# Compile server
echo "Compiling server..."
g++ server.cpp -o server -lrt -lpthread -I/etc/pliops -L/etc/pliops -lstorelib -g

# Check if server compilation was successful
if [ $? -eq 0 ]; then
    echo "Server compiled successfully."
else
    echo "Server compilation failed."
    exit 1
fi

# Compile client
echo "Compiling client..."
g++ client.cpp -o client -lrt -lpthread -g

# Check if client compilation was successful
if [ $? -eq 0 ]; then
    echo "Client compiled successfully."
else
    echo "Client compilation failed."
fi
