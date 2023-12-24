/**
 * Qbit gate operations (unified memory)
 * 
 * This program simulates a single-qubit quantum gate.
 * 
 * A single-qubit gate operation can be simulated as many 2x2 matrix multiplications. 
 * In an n-qubit quantum circuit, the n-qubit quantum state is represented as a vector 
 * of length N = 2^n, i.e.,  [a0, a1, ..., aN-1]. The vector indices can be represented 
 * with binary notations. For example, a9 = a0001001 in a 7-qubit system.
 */

#include <stdio.h>
#include <fstream>
#include <sys/time.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#define GATE_ELEM_NUMS 4
#define LINES_NOT_INPUT_VECT 5  //2 for matrix, 1 space, another 1 space, and 1 for target qubit


/**
 * CUDA Kernel Device code
 *
 * Computes the multiplication of 2x2 matrix gate and input vector into an output vector.  
 */
__global__ void qbitGateOp(float *in_vect, float *gate, float *out_vect, int numOfElems)
{
    int i1 = (blockDim.x * blockIdx.x * 2) + threadIdx.x;
    int i2 = i1 + blockDim.x;

    out_vect[i1] = (gate[0] * in_vect[i1]) + (gate[1] * in_vect[i2]);
    out_vect[i2] = (gate[2] * in_vect[i1]) + (gate[3] * in_vect[i2]);
}

/**
 * Host main routine
 */
int main(int argc, char *argv[])
{
    FILE *fp;          	// File pointer.
    char *input_file; 	// This variable holds the input file name.
    int target_qbit;
    int numOfElems;
    int inout_vect_size;
    float *in_vect, *gate, *out_vect;
    int gate_size;

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Exit with an error if the number of command-line arguments is incorrect.
    if (argc != 2) {
        printf("Error: Expected 1 command-line argument but was provided %d.\n", (argc - 1));
        exit(EXIT_FAILURE);
    }


    input_file = argv[1];
    // Open the input file for reading.
    fp = fopen(input_file, "r");
    if (fp == (FILE *) NULL) {
       // Exit with an error if file open failed.
       printf("Error: Unable to open file %s\n", input_file);
       exit(EXIT_FAILURE);
    }

    int numOfLines = 0;
    //count number of lines in file
    for (char c = getc(fp); c != EOF; c = getc(fp)){
        if (c == '\n'){
            numOfLines++;
        }
    }

    numOfElems = numOfLines - LINES_NOT_INPUT_VECT;
    inout_vect_size = numOfElems * sizeof(float);
    gate_size = GATE_ELEM_NUMS * sizeof(float);

     // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&in_vect, inout_vect_size);
    cudaMallocManaged(&out_vect, inout_vect_size);
    cudaMallocManaged(&gate, gate_size);

    rewind(fp); // rewinding back at the top of fp

    // get 2x2 matrix
    fscanf(fp, "%f %f", &gate[0], &gate[1]); // storing the 2x2 matrix in a 4x1 vector to keep element contiguous in memory for cudaMemcpy
    fscanf(fp, "%f %f", &gate[2], &gate[3]);

    // get input vector 
    for(int i = 0; i < numOfElems; i++){
        fscanf(fp, "%f\n", &in_vect[i]);
    }
	
    fscanf(fp, "%d", &target_qbit);


    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = pow(2, target_qbit);
    int blocksPerGrid = (numOfElems/2) / threadsPerBlock;   // numOfElems/2 because each thread 
                                                            // will perform 2 ops and access 2 indices
    //Timing the kernel execution for performanc analysis
    struct timeval begin, end;
    gettimeofday (&begin, NULL);
    //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    qbitGateOp<<<blocksPerGrid, threadsPerBlock>>>(in_vect, gate, out_vect, numOfElems);
    //wait for all GPU threads to finish executing
    cudaDeviceSynchronize();            
    gettimeofday (&end, NULL);
    int time_in_us = 1e6 * (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    //printing contents of h_out_vect
    for(int i = 0; i < numOfElems; i++){
        printf("%.3lf\n", out_vect[i]);	
    }
    printf("\n");


    // Free device global memory
    err = cudaFree(in_vect);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(gate);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(out_vect);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return 0;
}

