################################################################################
# Automatically-generated file. Do not edit!
################################################################################

cuda_dir = ./gpudt/Cuda

OBJS += \
$(cuda_dir)/cudaBoundary.cuo \
$(cuda_dir)/cudaConstraint.cuo \
$(cuda_dir)/cudaFlipping.cuo \
$(cuda_dir)/cudaMain.cuo \
$(cuda_dir)/cudaMissing.cuo \
$(cuda_dir)/cudaReconstruction.cuo \
$(cuda_dir)/cudaShifting.cuo \
$(cuda_dir)/cudaVoronoi.cuo \
$(cuda_dir)/pba2DHost.cuo 

# Each subdirectory must supply rules for building sources it contributes
$(cuda_dir)/%.cuo: $(cuda_dir)/%.cu
	$(NVCC) -c $(NVCCFLAGS) $(INCLUDES) -o "$@" "$<"
