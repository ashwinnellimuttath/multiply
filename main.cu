#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.h"

const unsigned int numStream = 3;

int main (int argc, char *argv[])
{

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    float *A_h, *B_h, *C_h;
    // float *A_d, *B_d, *C_d;
    float *A_d[numStream], *B_d[numStream], *C_d[numStream];



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

    VecSize = matArow*matAcol;
    const int segmentLen = VecSize / numStream;

    // A_h = (float*) malloc( sizeof(float)*A_sz );
    cudaHostAlloc((void**)&A_h, A_sz*sizeof(float), cudaHostAllocDefault);
    for (unsigned int i=0; i < A_sz; i++) { A_h[i] = (rand()%100)/100.00; }

    // B_h = (float*) malloc( sizeof(float)*B_sz );
    cudaHostAlloc((void**)&B_h, B_sz*sizeof(float), cudaHostAllocDefault);
    for (unsigned int i=0; i < B_sz; i++) { B_h[i] = (rand()%100)/100.00; }

    // C_h = (float*) malloc( sizeof(float)*C_sz );
    cudaHostAlloc((void**)&C_h, C_sz*sizeof(float), cudaHostAllocDefault);

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    A: %u x %u\n    B: %u x %u\n    C: %u x %u\n", matArow, matAcol,
        matBrow, matBcol, matArow, matBcol);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    /*************************************************************************/
    //INSERT CODE HERE
    // cudaMalloc((float**) &A_d, sizeof(float)*A_sz);
    // cudaMalloc((float**) &B_d, sizeof(float)*B_sz);
    // cudaMalloc((float**) &C_d, sizeof(float)*C_sz); 

    for (int i = 0; i < numStream; i++)
    {
        if (i != numStream-1)
        {
            cudaMalloc((void**) &A_d[i], sizeof(float) * segmentLen);
            cudaMalloc((void**) &B_d[i], sizeof(float) * segmentLen);
            cudaMalloc((void**) &C_d[i], sizeof(float) * segmentLen);
        }
        else    // remainder
        {
            cudaMalloc((void**) &A_d[i], sizeof(float) * (segmentLen + VecSize % numStream));
            cudaMalloc((void**) &B_d[i], sizeof(float) * (segmentLen + VecSize % numStream));
            cudaMalloc((void**) &C_d[i], sizeof(float) * (segmentLen + VecSize % numStream));
        }
    }

    // for (int i = 0; i < numStream; i++)
    // {
    //         cudaMalloc((float**) &A_d[i], sizeof(float) * VecSize);
    //         cudaMalloc((float**) &B_d[i], sizeof(float) * VecSize);
    //         cudaMalloc((float**) &C_d[i], sizeof(float) * VecSize);

    // }







    /*************************************************************************/
	
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------
    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);
	
    /*************************************************************************/
    //INSERT CODE HERE
    // cudaMemcpy(A_d, A_h, sizeof(float) * A_sz, cudaMemcpyHostToDevice);
    // cudaMemcpy(B_d, B_h, sizeof(float) * B_sz, cudaMemcpyHostToDevice);

    for (int i = 0; i < numStream; i++)
    {
        if (i != numStream-1)
        {
            cudaMemcpyAsync(A_d[i], A_h + i*segmentLen, sizeof(float)*segmentLen, cudaMemcpyHostToDevice, streams[i]);
            cudaMemcpyAsync(B_d[i], B_h + i*segmentLen, sizeof(float)*segmentLen, cudaMemcpyHostToDevice, streams[i]);
        }
        else
        {
            cudaMemcpyAsync(A_d[i], A_h + i*segmentLen, sizeof(float)*(segmentLen + VecSize % numStream), cudaMemcpyHostToDevice, streams[i]);
            cudaMemcpyAsync(B_d[i], B_h + i*segmentLen, sizeof(float)*(segmentLen + VecSize % numStream), cudaMemcpyHostToDevice, streams[i]);
        }
    }

    // for (int i = 0; i < numStream; i++)
    // {
    //     int Offset = i * segmentLen;
    //     cudaMemcpyAsync(&A_d[Offset], &A_h[Offset], sizeof(float)*segmentLen, cudaMemcpyHostToDevice, streams[i]);
    //     cudaMemcpyAsync(&B_d[Offset], &B_h[Offset], sizeof(float)*segmentLen, cudaMemcpyHostToDevice, streams[i]);
        
    //     basicSgemm(matArow/numStream, matArow/numStream, matArow/numStream, A_d[Offset], B_d[Offset], C_d[Offset], streams[i]);
    //     // else
    //     // {
    //     //     cudaMemcpyAsync(A_d[i], A_h + i*segmentLen, sizeof(float)*(segmentLen + VecSize % numStream), cudaMemcpyHostToDevice, streams[i]);
    //     //     cudaMemcpyAsync(B_d[i], B_h + i*segmentLen, sizeof(float)*(segmentLen + VecSize % numStream), cudaMemcpyHostToDevice, streams[i]);
    //     // }
    //     cudaMemcpyAsync(&C_h[Offset], &C_d[Offset], sizeof(float)*segmentLen, cudaMemcpyDeviceToHost, streams[i]);
    // }

    for (int i = 0; i < numStream; i++)
        cudaStreamSynchronize(streams[i]);





    /*************************************************************************/
    
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel using standard sgemm interface ---------------------------
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);


    // basicSgemm(matArow, matBcol, matBrow, A_d, B_d, C_d);
    for (int i = 0; i < numStream; i++)
    {
        if (i != numStream-1)
        {
            basicSgemm(matArow/numStream, matArow/numStream, matArow/numStream, A_d[i], B_d[i], C_d[i], streams[i]);
        }
        else
        {
            basicSgemm(matArow/numStream + VecSize % numStream, matArow/numStream + VecSize % numStream,matArow/numStream + VecSize % numStream,A_d[i], B_d[i], C_d[i], streams[i]);
        }
    }










    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) printf("Unable to launch kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------
    printf("Copying data from device to host..."); fflush(stdout);
    // startTime(&timer);

    /*************************************************************************/
    //INSERT CODE HERE
    // cudaMemcpy(C_h, C_d, sizeof(float) * C_sz, cudaMemcpyDeviceToHost);	


    for (int i = 0; i < numStream; i++)
    {
        if (i != numStream-1)
        {
            cudaMemcpyAsync(C_h + i*segmentLen, C_d[i], sizeof(float)*segmentLen, cudaMemcpyDeviceToHost, streams[i]);
        }
        else
        {
            cudaMemcpyAsync(C_h + i*segmentLen, C_d[i], sizeof(float)*(segmentLen + VecSize % numStream), cudaMemcpyDeviceToHost, streams[i]);
        }
    }









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
    // cudaFree(A_d);
    // cudaFree(B_d);
    // cudaFree(C_d);
    for (int i = 0; i < numStream; i++)
    {
        cudaFree(A_d[i]);
        cudaFree(B_d[i]);
        cudaFree(C_d[i]);
        cudaStreamDestroy(streams[i]);
    }
    /*************************************************************************/
;
    return 0;
}

