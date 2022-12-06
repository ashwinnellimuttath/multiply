#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.h"

const unsigned int numStream = 4;

int main (int argc, char *argv[])
{

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    float *A_h, *B_h, *C_h;

    float *A_d, *B_d, *C_d;

    size_t A_sz, B_sz, C_sz;
    unsigned matArow, matAcol;
    unsigned matBrow, matBcol;
    dim3 dim_grid, dim_block;
    unsigned VecSize;



    cudaStream_t streams[numStream];
    for (int i = 0; i < numStream; i++)
        cudaStreamCreate(&streams[i]);


    if (argc == 1) {
        matArow = 1000;
        matAcol = matBrow = 1000;
        matBcol = 1000;
    } else if (argc == 2) {
        matArow = atoi(argv[1]);
        matAcol = matBrow = atoi(argv[1]);
        matBcol = atoi(argv[1]);
    } else if (argc == 4) {
        matArow = atoi(argv[1]);
        matAcol = matBrow = atoi(argv[2]);
        matBcol = atoi(argv[3]);
    } else {
        printf("\n    Invalid input parameters!"
      "\n    Usage: ./sgemm-tiled                # All matrices are 1000 x 1000"
      "\n    Usage: ./sgemm-tiled <m>            # All matrices are m x m"
      "\n    Usage: ./sgemm-tiled <m> <k> <n>    # A: m x k, B: k x n, C: m x n"
      "\n");
        exit(0);
    }
   
    A_sz = matArow*matAcol;
    B_sz = matBrow*matBcol;
    C_sz = matArow*matBcol;

    VecSize = matArow*matBcol;
    const int segmentLen = VecSize / numStream;
    const int segmentLenA = A_sz / numStream;
    const int segmentLenB = B_sz / numStream;
    const int segmentLenC = C_sz / numStream;

    // A_h = (float*) malloc( sizeof(float)*A_sz );
    cudaHostAlloc((void**)&A_h, A_sz*sizeof(float), cudaHostAllocDefault);
    for (unsigned int i=0; i < A_sz; i++) { A_h[i] = (rand()%100)/100.00; }
    // cudaHostAlloc((void**)&a, A_sz*sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void**)&B_h, B_sz*sizeof(float), cudaHostAllocDefault);
    for (unsigned int i=0; i < B_sz; i++) { B_h[i] = (rand()%100)/100.00; }

    // cudaHostAlloc((void**)&b, A_sz*sizeof(float), cudaHostAllocDefault);
    // for (unsigned int i=0; i < B_sz; i++) { b[i] = (rand()%100)/100.00; }

    // C_h = (float*) malloc( sizeof(float)*C_sz );
    cudaHostAlloc((void**)&C_h, C_sz*sizeof(float), cudaHostAllocDefault);
    // cudaHostAlloc((void**)&c, A_sz*sizeof(float), cudaHostAllocDefault);


    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    A: %u x %u\n    B: %u x %u\n    C: %u x %u\n", matArow, matAcol,
        matBrow, matBcol, matArow, matBcol);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    /*************************************************************************/
    //INSERT CODE HERE


    cudaMalloc((float **)&A_d, sizeof(float) * A_sz);
    cudaMalloc((float **)&B_d, sizeof(float) * B_sz);
    cudaMalloc((float **)&C_d, sizeof(float) * C_sz);




    /*************************************************************************/
	
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------
    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);
	
    /*************************************************************************/

    //INSERT CODE HERE

    for (int i = 0; i < numStream; i++)
    {   
        
        int Offset = i * segmentLen;
        int OffsetA = i * segmentLenA;
        // int OffsetB = i * segmentLenB;
        int OffsetC = i * segmentLenC;
        if (i != numStream-1) {
            cudaMemcpyAsync(&A_d[OffsetA], &A_h[OffsetA], sizeof(float)*segmentLenA, cudaMemcpyHostToDevice, streams[i]);
            cudaMemcpyAsync(B_d, B_h, sizeof(float)*B_sz, cudaMemcpyHostToDevice, streams[i]);
            
            basicSgemmStream(matArow,matBcol,matBrow, &A_d[OffsetA], B_d, &C_d[OffsetC], streams[i]);

            cudaMemcpyAsync(&C_h[OffsetC], &C_d[OffsetC], sizeof(float)*segmentLenC, cudaMemcpyDeviceToHost, streams[i]);

        }
        else {
            Offset = (i * segmentLen) + (VecSize % numStream);
            OffsetA = (i * segmentLenA) + (A_sz % numStream);
            OffsetC = (i * segmentLenC) + (C_sz % numStream);
            cudaMemcpyAsync(&A_d[OffsetA], &A_h[OffsetA], sizeof(float)*(segmentLenA + (A_sz % numStream)), cudaMemcpyHostToDevice, streams[i]);
            cudaMemcpyAsync(B_d, B_h, sizeof(float)*(B_sz), cudaMemcpyHostToDevice, streams[i]);
            
            basicSgemmStream(matArow,matBcol,matBrow, &A_d[OffsetA], B_d, &C_d[OffsetC], streams[i]);
            
            cudaMemcpyAsync(&C_h[OffsetC], &C_d[OffsetC], sizeof(float)*(segmentLenC + (C_sz % numStream)), cudaMemcpyDeviceToHost, streams[i]);

        }

        // cudaStreamSynchronize(streams[i]);
    }

    /*************************************************************************/
    
    // cudaDeviceSynchronize();
    // stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel using standard sgemm interface ---------------------------
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);


    // cuda_ret = cudaDeviceSynchronize();
    // if(cuda_ret != cudaSuccess) printf("Unable to launch kernel");
    // stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------
    printf("Copying data from device to host..."); fflush(stdout);
    // startTime(&timer);

    /*************************************************************************/
    //INSERT CODE HERE
    // cudaMemcpy(C_h, C_d, sizeof(float) * C_sz, cudaMemcpyDeviceToHost);	

    /*************************************************************************/

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    // printf(C_h, "c_h");fflush(stdout);

    verify(A_h, B_h, C_h, matArow, matAcol, matBcol);


    // Free memory ------------------------------------------------------------
    // free(A_h);
    // free(B_h);
    // free(C_h);

    cudaFreeHost(A_h);
    cudaFreeHost(B_h);
    cudaFreeHost(C_h);
    /*************************************************************************/
    //INSERT CODE HERE
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    for (int i = 0; i < numStream; i++)
    {
        cudaStreamDestroy(streams[i]);
    }
    /*************************************************************************/
;
    return 0;
}

