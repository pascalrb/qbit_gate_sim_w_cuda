/**
 * Qbit gate operations (GPU dispatch)
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
__global__ void qbitGateOp(float *d_in_vect, float *d_gate, float *d_out_vect, int numOfElems)
{
    int i1 = (blockDim.x * blockIdx.x * 2) + threadIdx.x;
    int i2 = i1 + blockDim.x;

    d_out_vect[i1] = (d_gate[0] * d_in_vect[i1]) + (d_gate[1] * d_in_vect[i2]);
    d_out_vect[i2] = (d_gate[2] * d_in_vect[i1]) + (d_gate[3] * d_in_vect[i2]);
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
    float h_gate [GATE_ELEM_NUMS];
    float *h_in_vect, *h_out_vect;
    int gate_size = GATE_ELEM_NUMS * sizeof(float);

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

    // Alloc space for device copies of in/out vect, and gate matrix 
    float *d_in_vect = NULL;
    err = cudaMalloc((void **)&d_in_vect, inout_vect_size); 
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device mem for either in_vect (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    float *d_gate = NULL;
    err = cudaMalloc((void **)&d_gate, gate_size); 
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device mem for either gate (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    float *d_out_vect = NULL;
    err = cudaMalloc((void **)&d_out_vect, inout_vect_size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device mem for either out_vect (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Alloc space for host copies of in/out vect, and gate matrix 
    h_in_vect = (float *)malloc(inout_vect_size);  
    h_out_vect = (float *)malloc(inout_vect_size);

    // Verify that allocations succeeded
    if (h_in_vect == NULL || h_out_vect == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    rewind(fp); // rewinding back at the top of fp

    // get 2x2 matrix
    fscanf(fp, "%f %f", &h_gate[0], &h_gate[1]); // storing the 2x2 matrix in a 4x1 vector to keep element contiguous in memory for cudaMemcpy
    fscanf(fp, "%f %f", &h_gate[2], &h_gate[3]);

    // get input vector 
    for(int i = 0; i < numOfElems; i++){
        fscanf(fp, "%f\n", &h_in_vect[i]);
    }
	
    fscanf(fp, "%d", &target_qbit);


    // Copy the host input vectors A and B in host memory to the
    // device input vectors in device memory
    err = cudaMemcpy(d_in_vect, h_in_vect, inout_vect_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy input vector from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_gate, h_gate, gate_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy gate matrix from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = pow(2, target_qbit);
    int blocksPerGrid = (numOfElems/2) / threadsPerBlock;      // numOfElems/2 because each thread 
                                                                // will perform 2 ops and access 2 indices
    //Timing the kernel execution for performance analysis
    struct timeval begin, end;
    gettimeofday (&begin, NULL);
    //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    qbitGateOp<<<blocksPerGrid, threadsPerBlock>>>(d_in_vect, d_gate, d_out_vect, numOfElems);
    cudaDeviceSynchronize(); //wait for all GPU threads to finish executing
    gettimeofday (&end, NULL);
    int time_in_us = 1e6 * (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector in host memory.
    err = cudaMemcpy(h_out_vect, d_out_vect, inout_vect_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy output vector from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //printing contents of h_out_vect
    for(int i = 0; i < numOfElems; i++){
        printf("%.3lf\n", h_out_vect[i]);	
    }
    printf("\n");


    // Free device global memory
    err = cudaFree(d_in_vect);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_gate);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_out_vect);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_in_vect);
    free(h_out_vect);

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

