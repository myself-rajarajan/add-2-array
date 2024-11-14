
// __global__ is the function specifier to define the cuda kernal that run the actual GPU
__global__ void add(int *a, int *b, int *c){                                     
  int i = threadIdx.x + blockDim.x * blockIdx.x;                              // to add 2 vectors or arrays together
  c[i] = a[i] + b[i];
}
__managed__ int vector_a[256], vector_b[256], vector_c[256];                 // __managed__ is used for cuda to accessed from both CPU and device GPU

int main(){
  for(int i = 0; i < 256; i++){
    vector_a[i] = i;
    vector_b[i] = 256 - i;
  }

  // <<<1,256>>> represents the <<<block,threads per block>>> are used to run this code in parallel 
  add<<<1,256>>>(vector_a, vector_b, vector_c);                                

  cudaDeviceSynchronize();                                        // will pass the execution of thos code and wait for it to complete on the GPU

  int result_sum = 0;
  for(i = 0; i < 256; i++){
    result_sum += vector_c[i];                                   // result to store the output. 
  }

  printf("result: sum = %d", result_sum);
}