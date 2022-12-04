#include <stdio.h>

#define TILE_SIZE 16

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    /*************************************************************************/
    // INSERT KERNEL CODE HERE
    __shared__ float A_T[TILE_SIZE][TILE_SIZE];
    __shared__ float B_T[TILE_SIZE][TILE_SIZE];



    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int p,i;
    float Pvalue = 0.0;
    for (p=0; p< (k-1)/TILE_SIZE + 1; p++) {
        
        if (row < m && p*TILE_SIZE + tx < k) {
            A_T[ty][tx] = A[row*k + p*TILE_SIZE+tx];
        }
        else {
            A_T[ty][tx] = 0.0;
        }
        if (col < n && p*TILE_SIZE+ty < k) {
            B_T[ty][tx] = B[(p*TILE_SIZE+ty)*n + col];
        }
        else {
            B_T[ty][tx] = 0.0;
        }
        __syncthreads();
        
        if (row < m && col < n)
        {
            for (i = 0; i < TILE_SIZE; i++)
                Pvalue += A_T[ty][i] * B_T[i][tx];
        }
        __syncthreads();
    }
    if (row < m && col < n){
        C[row*n + col] = Pvalue;
    }    
    /*************************************************************************/
}

void basicSgemm(int m, int n, int k, const float *A, const float *B, float *C)
{
    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;
	
    /*************************************************************************/
    //INSERT CODE HERE
    dim3 DimGrid((n-1)/BLOCK_SIZE + 1,(m-1)/BLOCK_SIZE + 1,1); 
    dim3 DimBlock(BLOCK_SIZE,BLOCK_SIZE,1);
    /*************************************************************************/

    // Invoke CUDA kernel -----------------------------------------------------

    /*************************************************************************/
    //INSERT CODE HERE
    mysgemm<<<DimGrid,DimBlock>>>(m,n,k,A,B,C);  	
    /*************************************************************************/
}

void basicSgemmStream(int m, int n, int k, const float *A, const float *B, float *C, cudaStream_t stream)
{
    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;
	
    /*************************************************************************/
    //INSERT CODE HERE
    dim3 DimGrid((n-1)/BLOCK_SIZE + 1,(m-1)/BLOCK_SIZE + 1,1); 
    dim3 DimBlock(BLOCK_SIZE,BLOCK_SIZE,1);
    /*************************************************************************/

    // Invoke CUDA kernel -----------------------------------------------------

    /*************************************************************************/
    //INSERT CODE HERE
    mysgemm<<<DimGrid,DimBlock, 0, stream>>>(m,n,k,A,B,C);  	
    /*************************************************************************/
}


