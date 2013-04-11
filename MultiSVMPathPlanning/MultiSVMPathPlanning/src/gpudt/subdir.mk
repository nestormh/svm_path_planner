################################################################################
# Automatically-generated file. Do not edit!
################################################################################

gpudt_dir = ./gpudt

-include $(gpudt_dir)/Cuda/subdir.mk

OBJS += \
$(gpudt_dir)/gpudt.o \
$(gpudt_dir)/predicates.o

# Each subdirectory must supply rules for building sources it contributes
$(gpudt_dir)/%.o: $(gpudt_dir)/%.cpp
	$(CC) $(INCLUDES) $(OPT_FLAGS) -c -o "$@" "$<"

