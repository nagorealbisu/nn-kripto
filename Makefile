# Compiler settings
CXX = g++
CC  = gcc
CXX_FLAGS = -Iinclude -I/home/nagor/openfhe-development/src/pke/include -D__USE_MINGW_ANSI_STDIO=1 -Wall -MMD -MP -fpermissive
C_FLAGS   = -Iinclude -D__USE_MINGW_ANSI_STDIO=1 -Wall -MMD -MP -DCPU

LIB_DIR   = -L/home/nagor/openfhe-development/build/lib
LIBS      = -lOPENFHEbinfhe -lOPENFHEcore -lOPENFHEpke -lm

# Directories
SRC_DIR = src
ENC_DIR = $(SRC_DIR)/encrypted_nn
PLA_DIR = $(SRC_DIR)/plaintext_nn
OBJ_DIR = obj
BIN_DIR = bin

# Source files
MAIN_C      := $(SRC_DIR)/main.c
PLA_C_SRCS  := $(wildcard $(PLA_DIR)/*.c)

# C++ files (encrypted_nn) — list manually
CPP_FILES := enc_inference.cpp enc_training_main.cpp enc_train_functions.cpp ckks_utils.cpp nn_crypto_context.cpp
CPP_SRCS  := $(addprefix $(ENC_DIR)/, $(CPP_FILES))

# Object files
OBJ_MAIN_C := $(MAIN_C:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
OBJS_PLA   := $(PLA_C_SRCS:$(PLA_DIR)/%.c=$(OBJ_DIR)/plaintext_nn/%.o)
OBJS_CPP   := $(CPP_SRCS:$(ENC_DIR)/%.cpp=$(OBJ_DIR)/encrypted_nn/%.o)

ALL_OBJS := $(OBJ_MAIN_C) $(OBJS_PLA) $(OBJS_CPP)

# Targets
TARGET1 = $(BIN_DIR)/test
TARGET2 = $(BIN_DIR)/train

.PHONY: all clean

all: $(TARGET1) $(TARGET2)

# Executables
$(TARGET1): $(ALL_OBJS) | $(BIN_DIR)
	$(CXX) $(ALL_OBJS) -o $@ $(LIB_DIR) $(LIBS)

$(TARGET2): $(ALL_OBJS) | $(BIN_DIR)
	$(CXX) $(ALL_OBJS) -o $@ $(LIB_DIR) $(LIBS)

# Compile main.c
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(C_FLAGS) -c $< -o $@

# Compile plaintext_nn .c files
$(OBJ_DIR)/plaintext_nn/%.o: $(PLA_DIR)/%.c | $(OBJ_DIR)/plaintext_nn
	$(CC) $(C_FLAGS) -c $< -o $@

# Compile encrypted_nn .cpp files
$(OBJ_DIR)/encrypted_nn/%.o: $(ENC_DIR)/%.cpp | $(OBJ_DIR)/encrypted_nn
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# Create directories if they don't exist
$(OBJ_DIR) $(OBJ_DIR)/plaintext_nn $(OBJ_DIR)/encrypted_nn $(BIN_DIR):
	mkdir -p $@

clean:
	$(RM) -rv $(OBJ_DIR) $(BIN_DIR)/*.exe $(TARGET1) $(TARGET2)

# Include dependency files
-include $(OBJ_MAIN_C:.o=.d) $(OBJS_PLA:.o=.d) $(OBJS_CPP:.o=.d)
