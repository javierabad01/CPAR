%%cu


//ABAD HERNANDEZ, JAVIER
//CASTRO GARCIA, JAIME


/**
 * Matrix Multiplication: C = A * B.
 *
 * This file contains both device and host code to compute a matrix multiplication.
 *
 */

#include <math.h>
#include <stdio.h>

#define MATRIX_DIM	 16
#define SEGMENT_SIZE 32

// --------------------
// Device Kernels
// --------------------
///////////////////////////////////////////////////////////
//
// Computes the Transpose of a Matrix
//
///////////////////////////////////////////////////////////
__global__ void transposeMatrix(float *d_data, int mat_dim) {

	// Array in Shared Memory
	extern __shared__ float sdata[];
	
	// Calcula el ID del hilo en el eje x y en el y
	int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
	int tid_y = threadIdx.y + blockIdx.y * blockDim.y;

	// Transpone cada segmento de la matriz
	if (tid_x < mat_dim && tid_y < mat_dim) {
		// Copia los datos de la matriz original a la memoria compartida en una posición transpuesta
		sdata[threadIdx.y * blockDim.x + threadIdx.x] = d_data[tid_y * mat_dim + tid_x];
	}

	__syncthreads();


	// Copia los datos transpuestos de la memoria compartida a la matriz original
	if (tid_x < mat_dim && tid_y < mat_dim) {
		d_data[tid_x * mat_dim + tid_y] = sdata[threadIdx.y * blockDim.x + threadIdx.x];
	}
}

///////////////////////////////////////////////////////////////////////////////
//
// Computes the scalar product of two vectors of N elements on GPU.
//
///////////////////////////////////////////////////////////////////////////////
__global__ void scalarProd(float *C, const float *A, const float *B, int nElem) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nElem) {
        // Calcula el producto escalar y lo guarda en el vector C
        C[tid] = A[tid] * B[tid];
    }
}

/////////////////////////////////////////////////////////////////
//
// Computes a standard parallel reduction on GPU.
//
/////////////////////////////////////////////////////////////////
__global__ void vectorReduce(float *R, const float *C, int nElem)
{
    // Array in Shared Memory
    extern __shared__ float sdata[];

    // Calcula el ID del hilo dentro del bloque
    unsigned int tidb = threadIdx.x;

    // Calcula el ID global del hilo en el grid
    unsigned int tidg = blockIdx.x * blockDim.x + tidb;

    // Mueve los datos del vector C desde la memoria global a la memoria compartida
    sdata[tidb] = (tidg < nElem) ? C[tidg] : 0;

    __syncthreads();

    // Realiza la reducción en la memoria compartida
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tidb < s)
        {
            // Suma los elementos adyacentes y guarda el resultado en la primera mitad del array
            sdata[tidb] += sdata[tidb + s];
        }
        __syncthreads();
    }
    
    // Suma el resultado final de este bloque al resultado global R[0]
    if (tidb == 0)
    {
        R[blockIdx.x] = sdata[0];
    }
}

// ---------------------
// Host Utility Routines
// ---------------------
void matrixMul(const float *A, const float *B, float *C, const int n)
{
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			float acum = 0.0f;
			for (int k = 0; k < n; k++) {
				acum += A[i * n + k] * B[k * n + j];
			}
			C[i * n + j] = acum;
		}
	}
}

bool compareData(float *h_C, float *d_C, int n)
{
  // Hemos modificado el epsilon, para que tuviera menor indice de precision, ya que en los float puede variar.
	double eps = 1.E-4;
	for (int i = 0; i < n * n; i++) {
		if (fabsf(h_C[i] - d_C[i]) > eps) {
			return false;
		}
	}
	return true;
}

float randFloat(float low, float high) {
	float t = (float) rand() / (float) RAND_MAX;
	return (1.0f - t) * low + (t * high);
}

// ------------
// Main Program
// ------------
int main( void ) {

    // Matrix Dimensions
    int dim_x = MATRIX_DIM;
    int dim_y = dim_x;
    
    // Matrix Size
    int mat_size = dim_x * dim_y;
    
    // Block Dimension
    int block_dim = SEGMENT_SIZE;
    
    // Number of Blocks
    int n_block = ( dim_x % block_dim == 0 ) ? dim_x / block_dim : dim_x / block_dim + 1;
    
    // Execution Configuration Parameters
    dim3 blocksPerGrid  ( n_block, n_block );
    dim3 threadsPerBlock(block_dim, block_dim);
    
    // Size Required to Store the Matrix
    size_t n_bytes = (mat_size * sizeof(float));
    
    // Allocate Pinned Host Memory
    float *h_A, *h_B, *h_C, *h_R;
    
    cudaMallocHost((void**)&h_A, n_bytes);
    cudaMallocHost((void**)&h_B, n_bytes);
    cudaMallocHost((void**)&h_C, n_bytes);
    cudaMallocHost((void**)&h_R, n_bytes);

    
    // Initialize Host Data
    srand(123);
    
    // Generating input data on CPU
    for (int i=0; i < mat_size; i++) {
      h_A[i] = randFloat(0.0f, 1.0f);
      h_B[i] = randFloat(0.0f, 1.0f);
    }
    
    // Compute Reference Matrix Multiplication
    matrixMul(h_A, h_B, h_C, dim_x);

    // CUDA Streams
    cudaStream_t stream;
    
    // Create Stream
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    
    // Performance Data
    float kernel_time, kernel_bandwidth;
    
    // Allocate Device Memory
    float *d_A, *d_B, *d_C, *d_R;

    cudaMalloc((void**)&d_A, n_bytes);
    cudaMalloc((void**)&d_B, n_bytes);
    cudaMalloc((void**)&d_C, n_bytes);
    cudaMalloc((void**)&d_R, n_bytes);

    // CUDA Events
    cudaEvent_t start, stop;


    // Init Events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start Time Measurement
    cudaEventRecord(start, stream);

    // Copy Host Data to Device
    cudaMemcpyAsync(d_A, h_A, n_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, h_B, n_bytes, cudaMemcpyHostToDevice, stream);

    //Se le pasa el tamaño de cada bloque y el numero de hilos por cada bloque.
    transposeMatrix<<<blocksPerGrid, threadsPerBlock, block_dim * block_dim * sizeof(float)>>>(d_B, dim_x);
    cudaStreamSynchronize(stream);

	  for(int i = 0; i < dim_y; i++) {
        for(int j = 0; j < dim_x; j++) {
            scalarProd<<<blocksPerGrid, threadsPerBlock,0,stream>>>(d_C, d_A + i * dim_x, d_B + j * dim_x, dim_x);
            cudaStreamSynchronize(stream);

            //La reducción se hace sobre un único bloque y tendra como numero de hilos el tamaño del bloque.
            vectorReduce<<<1, block_dim, block_dim * sizeof(float),stream>>>(d_R + i * dim_x + j, d_C, dim_x * dim_x);

        }
	  }
	  cudaDeviceSynchronize();

    // Copy Device Data to Host
	  cudaMemcpyAsync(h_R, d_R, n_bytes, cudaMemcpyDeviceToHost, stream);

   	cudaStreamSynchronize(stream);


    // End Time Measurement
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&kernel_time, start, stop);


    // Este bucle se utilizó para comprobar que las transposiciones fueran correctas valor por valor para los 10 primeros.
    /*
    for (int i=0; i<10;i++)
      printf("%f, %f\n",h_C[i],h_R[i]);
    */

    bool res = compareData(h_C, h_R, dim_x);
    
    if (res == true) {
        // Report Effective Bandwidth
        kernel_bandwidth = (2.0f * 1000.0f * n_bytes) / (1024 * 1024 * 1024);
        kernel_bandwidth /= kernel_time;

        printf("Throughput = %.4f GB/s, Time = %.5f ms, Size = %u fp32 elements, \n",
            kernel_bandwidth, kernel_time, (dim_x * dim_y));
    }

    // Free Host Memory
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFreeHost(h_R);

    // Free Device Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_R);

    // Destroy Events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Destroy Stream
    cudaStreamDestroy(stream);

    if (res == false) {
        printf("Test Failed!\n");
        exit(EXIT_FAILURE);
    }
    printf("Test Passed\n");
    exit(EXIT_SUCCESS);
}

