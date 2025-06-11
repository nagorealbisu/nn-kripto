# Compiler settings
CXX = g++
CC  = gcc
CXX_FLAGS = -Iinclude -I/home/nagor/openfhe-development/src/pke/include -D__USE_MINGW_ANSI_STDIO=1 -Wall -MMD -MP -fpermissive
C_FLAGS   = -Iinclude -D__USE_MINGW_ANSI_STDIO=1 -Wall -MMD -MP -DCPU

LIB_DIR   = -L/home/nagor/openfhe-development/build/lib
LIBS      = -lOPENFHEbinfhe -lOPENFHEcore -lOPENFHEpke -lm

# Directories
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Source files
C_SRCS   := $(wildcard $(SRC_DIR)/*.c)
CPP_SRCS := enc_inference.cpp

# Object files
OBJS_C   := $(C_SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
OBJS_CPP := $(CPP_SRCS:%.cpp=$(OBJ_DIR)/%.o)

# The main executable that links main.o, enc_inference.o and other C objects
TARGET = $(BIN_DIR)/nn

.PHONY: all clean

all: $(TARGET)

# Link executable with g++, including main.o, enc_inference.o and other C objects (except main.o)
$(TARGET): $(OBJ_DIR)/main.o $(OBJ_DIR)/enc_inference.o $(filter-out $(OBJ_DIR)/main.o, $(OBJS_C)) | $(BIN_DIR)
	$(CXX) $^ -o $@ $(LIB_DIR) $(LIBS)

# Compile C source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(C_FLAGS) -c $< -o $@

# Compile C++ source files
$(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIR)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# Create directories if they don't exist
$(OBJ_DIR) $(BIN_DIR):
	mkdir -p $@

clean:
	$(RM) -rv $(OBJ_DIR) $(BIN_DIR)/*.exe $(TARGET)

# Include dependency files if they exist
-include $(OBJS_C:.o=.d) $(OBJS_CPP:.o=.d)
