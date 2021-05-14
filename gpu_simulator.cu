#define _USE_MATH_DEFINES

#include <iostream>
#include <cmath>
#include <vector>
#include <complex>
#include <cuComplex.h>
#include <assert.h>
#include <cusparse.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

using namespace std;

static inline int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__global__ void
scan_upsweep_kernel(int N, int twod, int twod1, int* device_data) {
    int index = twod1 * (blockIdx.x * blockDim.x + threadIdx.x);
    device_data[index+twod1-1] += device_data[index+twod-1];
    
}

__global__ void
scan_lastIndex_kernel(int N, int* device_data) {
    device_data[N-1] = 0;
}

__global__ void
scan_downsweep_kernel(int N, int twod, int twod1, int* device_data) {
    int index = twod1 * (blockIdx.x * blockDim.x + threadIdx.x);
    int t = device_data[index+twod-1];
    device_data[index+twod-1] = device_data[index+twod1-1];
    device_data[index+twod1-1] += t;
}

void exclusive_scan(int* device_data, int length)
{
    int N = nextPow2(length);

    int threadsPerBlock = 512;
    int blocks = max(N / threadsPerBlock, 1);

    // upsweep
    for (int twod = 1; twod < N; twod *= 2)
    {
        if (blocks > 1)
            blocks /= 2;
        else
            threadsPerBlock /= 2;
        int twod1 = twod * 2;

        scan_upsweep_kernel<<<blocks, threadsPerBlock>>>(N, twod, twod1, device_data);
        cudaDeviceSynchronize();
    }
    scan_lastIndex_kernel<<<1, 1>>>(N, device_data);

    // currently, threadsPerBlock = 1 and blocks = 1

    // downsweep
    for (int twod = N / 2; twod >= 1; twod /= 2)
    {
        int twod1 = twod * 2;
        scan_downsweep_kernel<<<blocks, threadsPerBlock>>>(N, twod, twod1, device_data);
        cudaDeviceSynchronize();
        if (threadsPerBlock < 512)
            threadsPerBlock *= 2;
        else
            blocks *= 2;
    }
}

void cudaScan(int* inarray, int length, int* resultarray)
{
    int* device_data;
    int rounded_length = nextPow2(length);
    cudaMalloc((void **)&device_data, sizeof(int) * rounded_length);

    cudaMemcpy(device_data, inarray, length * sizeof(int),
               cudaMemcpyDeviceToDevice);
    gpuErrchk(cudaDeviceSynchronize());
    exclusive_scan(device_data, length);

    // Wait for any work left over to be completed.
    gpuErrchk(cudaDeviceSynchronize());

    cudaMemcpy(resultarray, device_data, length * sizeof(int),
               cudaMemcpyDeviceToDevice);
}


typedef enum OneQubitGateType{
    IDENTITY,
    NOT,
    HADAMARD,
    PHASE
} OneQubitGateType;

#define NO_PARAM 0.0 

typedef enum TwoQubitGateType{
    CNOT,
    SWAP,
    CPHASE
} TwoQubitGateType;

float toBW(int bytes, float sec) {
    return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}

// Sparse matrix reserves the first element for dimension, assuming matrix is square.
// It is implemented using a vector.
struct sparse_elt {
    long u;
    long v;
    cuDoubleComplex amp;
};

__device__ sparse_elt
createElt(long u, long v, cuDoubleComplex amp) {
    sparse_elt elt;
    elt.u = u;
    elt.v = v;
    elt.amp = amp;
    return elt;
}

__device__ double
norm(cuDoubleComplex amp) {
    return cuCreal(cuCmul(amp, cuConj(amp)));
}

double normHost(cuDoubleComplex amp) {
    return cuCreal(cuCmul(amp, cuConj(amp)));
}

__device__ bool
equalsZero(cuDoubleComplex amplitude) {
    return (norm(amplitude) < 1e-15);
}

bool equalsZeroHost(cuDoubleComplex amplitude) {
    return (normHost(amplitude) < 1e-15);
}

__device__ void
atomicComplexAdd(cuDoubleComplex* ptr, cuDoubleComplex valToAdd) {
    double2 *newPtr = (double2*) ptr;
    double2 newVal = (double2) valToAdd;
    atomicAdd(&(newPtr->x), newVal.x);
    atomicAdd(&(newPtr->y), newVal.y);
}



__global__ void
assignAll(cuDoubleComplex *vec, const long N, cuDoubleComplex val) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int numPerThread = max(N / (blockDim.x * gridDim.x), 1L);
    for(int i = id * numPerThread; i < ((id + 1) * numPerThread) && i < N; ++i) {
        vec[id] = val;
    }
}


__device__ void
assignAllDevice(cuDoubleComplex *vec, const long N, cuDoubleComplex val) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int numPerThread = max(N / (blockDim.x * gridDim.x), 1L);
    for(int i = id * numPerThread; i < ((id+1) * numPerThread) && i < N; ++i) {
        vec[id] = val;
    }
}

__device__ void
assignAllDoublesDevice(double *vec, const long N, double val) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int numPerThread = max(N / (blockDim.x * gridDim.x), 1L);
    for(int i = id * numPerThread; i < ((id+1) * numPerThread) && i < N; ++i) {
        vec[id] = val;
    }
}

__global__ void
assignOne(cuDoubleComplex *vec, const long N, cuDoubleComplex val, const long which) {
    if(blockIdx.x == 0 && threadIdx.x == 0) {
        vec[which] = val;
    }
}

/* These reduction helpers require that vec is of length blockDim * gridDim. */
__device__ void
sumReductionHelper(int id, double *vec, double *ans) {
    __shared__ double sum;
    sum = 0.0;
    atomicAdd(&sum, vec[id]);
    __syncthreads();
    if(threadIdx.x == 0) {
        atomicAdd(ans, sum);
    }
}



__global__ void
totalMag (cuDoubleComplex *state, const long N, double *temp, double *ans) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int numPerThread = max(N / (blockDim.x * gridDim.x), 1L);
    assignAllDoublesDevice(temp, blockDim.x * gridDim.x, 0.0);
    double sum = 0;
    for (int i = id * numPerThread; i < ((id+1) * numPerThread) && i < N; ++i) {
        sum += norm(state[i]);
    }

    temp[id] = sum;
    __syncthreads();
    if(id < N) {
        sumReductionHelper(id, temp, ans);
    }
}

double isNormalized(cuDoubleComplex *state, const long N, const int blocks, const int threadsPerBlock) {
    double* temp;
    double* ansDevice;
    cudaMalloc((void **)&temp, sizeof(double) * blocks * threadsPerBlock);
    cudaMalloc((void **)&ansDevice, sizeof(double));
    double ansHost = 0.0;
    cudaMemcpy(ansDevice, &ansHost, sizeof(double), cudaMemcpyHostToDevice);
    totalMag<<<blocks, threadsPerBlock>>> (state, N, temp, ansDevice);
    cudaDeviceSynchronize();
    cudaMemcpy(&ansHost, ansDevice, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(temp);
    cudaFree(ansDevice);
    return ansHost;//(abs(ansHost - 1.0) < 1e-13);
}

cuDoubleComplex* uniform(const long N, const int blocks, const int threadsPerBlock) {
    cuDoubleComplex* state;
    cudaMalloc((void **)&state, sizeof(cuDoubleComplex) * N);
    assignAll<<<blocks, threadsPerBlock>>> (state, N, make_cuDoubleComplex(sqrt(1.0/N),0.0));
    cudaDeviceSynchronize();
    return state;
}
cuDoubleComplex* zero(const long N, const int blocks, const int threadsPerBlock) {
    cuDoubleComplex* state;
    cudaMalloc((void **)&state, sizeof(cuDoubleComplex) * N);
    assignAll<<<blocks, threadsPerBlock>>>(state, N, make_cuDoubleComplex(0.0, 0.0));
    cudaDeviceSynchronize();
    return state;
}

cuDoubleComplex* classical(const long N, const long which, const int blocks, const int threadsPerBlock) {
    cuDoubleComplex* state;
    cudaMalloc((void **)&state, sizeof(cuDoubleComplex) * N);
    assignOne<<<blocks, threadsPerBlock>>>(state, N, make_cuDoubleComplex(1.0, 0.0), which);
    cudaDeviceSynchronize();
    return state;
}


/* Print vector features */

void printComplex(cuDoubleComplex amplitude) {
    cout << "(" << cuCreal(amplitude) << "+" << cuCimag(amplitude) << "i)";
}
void printVec(cuDoubleComplex* state, const int n, const long N, bool printAll) {
    cuDoubleComplex* stateHost = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex) * N);
    cudaMemcpy(stateHost, state, sizeof(cuDoubleComplex) * N, cudaMemcpyDeviceToHost);
    cout << "State:\n";
    for (int i = 0; i < N; ++i) {
        if (!printAll && equalsZeroHost(stateHost[i]))
            continue;
        printComplex(stateHost[i]);
        cout << "|";
        for (int j = n-1; j >= 0; --j) {
            cout << (0x1 & (i>>j));
        }
        cout << "> + ";
        cout << "\n";
    }
    free(stateHost);
}
void printVecProbs(cuDoubleComplex* state, const int n, const long N, bool printAll) {
    cuDoubleComplex* stateHost = (cuDoubleComplex*) malloc(sizeof(cuDoubleComplex) * N);
    cudaMemcpy(stateHost, state, sizeof(cuDoubleComplex) * N, cudaMemcpyDeviceToHost);
    cout << "Probability distribution:\n";
    for (int i = 0; i < N; ++i) {
        if (!printAll && equalsZeroHost(stateHost[i]))
            continue;

        cout << "|<";
        for (int j = n-1; j >= 0; --j) {
            cout << (0x1 & (i>>j));
        }
        cout << "|Psi>|^2 = " << normHost(stateHost[i]) << "\n";
    }
    free(stateHost);
}


__global__ void
getMostLikely(cuDoubleComplex* state, long* answers, double* probs, const int n, const long N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int numPerThread = max(N / (blockDim.x * gridDim.x), 1L);
    double prob = 0.0;

    for (long i = id * numPerThread; i < ((id+1) * numPerThread) && i < N; i++) {
        if (norm(state[i]) > prob) {
            prob = norm(state[i]);
            answers[id] = i;
        }
    }
    probs[id] = prob;
}
long getMostLikelyHost(cuDoubleComplex* state, const int n, const long N, const int blocks, const int threadsPerBlock) {
    /*cudaMalloc((void**)&answers, sizeof(long)*blocks*threadsPerBlock);
    cudaMalloc((void**)&probs, sizeof(double)*blocks*threadsPerBlock);
    getMostLikely<<<blocks, threadsPerBlock>>>(state, answers, probs, n, N);
    cudaDeviceSynchronize();

    double *probsHost = (double*)malloc(sizeof(double)*blocks*threadsPerBlock);
    cudaMemcpy(probsHost, probs, sizeof(double)*blocks*threadsPerBlock, cudaMemcpyDeviceToHost);
    
    */
    cuDoubleComplex* stateHost = (cuDoubleComplex*)malloc(N * sizeof(cuDoubleComplex));
    cudaMemcpy(stateHost, state, N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    double maxProb = 0.0;
    long answer = 0;
    for(long i = 0; i < N; ++i) {
        if(normHost(stateHost[i]) > maxProb) {
            answer = i;
            maxProb = normHost(stateHost[i]);
        }
    }
    /*long *answer = (long*)malloc(sizeof(long) * blocks * threadsPerBlock);
    cudaMemcpy(&answer, answers, sizeof(long) * blocks * threadsPerBlock, cudaMemcpyDeviceToHost);
    free(probsHost);
    cudaFree(answers);
    cudaFree(probs);*/
    free(stateHost);
    return answer;
}

__device__ void 
printOp(sparse_elt* U) {
    printf("Operator\n");
    for(int i = 0; i < U[0].u; ++i) {
        printf("(%d, %d), %f + %f i\n", (int)U[i].u, (int)U[i].v, cuCreal(U[i].amp), cuCimag(U[i].amp));
    }
}

/* One qubit gates */ 
__device__ sparse_elt*
identity(){
    __shared__ sparse_elt id[3];
    // dimensions
    id[0] = createElt(3, 2, make_cuDoubleComplex(0.0, 0.0));
    // values
    id[1] = createElt(0, 0, make_cuDoubleComplex(1.0, 0.0));
    id[2] = createElt(1, 1, make_cuDoubleComplex(1.0, 0.0));
    return id;
}

__device__ sparse_elt*
naught() {
    __shared__ sparse_elt cx[3];
    // dimensions
    cx[0] = createElt(3, 2, make_cuDoubleComplex(0.0, 0.0));
    // values
    cx[1] = createElt(1, 0, make_cuDoubleComplex(1.0, 0.0));
    cx[2] = createElt(0, 1, make_cuDoubleComplex(1.0, 0.0));
    return cx;
}

__device__ sparse_elt*
hadamard() {
    __shared__ sparse_elt had[5];
    // dimensions
    had[0] = createElt(5, 2, make_cuDoubleComplex(0.0, 0.0));
    // values
    double c = 1.0 / sqrt(2.0);
    had[1] = createElt(0, 0, make_cuDoubleComplex(c, 0.0));
    had[2] = createElt(0, 1, make_cuDoubleComplex(c, 0.0));
    had[3] = createElt(1, 0, make_cuDoubleComplex(c, 0.0));
    had[4] = createElt(1, 1, make_cuDoubleComplex(-c, 0.0));
    return had;
}

__device__ sparse_elt*
phase(double phi) {
    __shared__ sparse_elt phase[3];
    // dimensions
    phase[0] = createElt(3, 2, make_cuDoubleComplex(0.0, 0.0));
    // values
    phase[1] = createElt(0, 0, make_cuDoubleComplex(1.0, 0.0));
    phase[2] = createElt(1, 1, make_cuDoubleComplex(cos(phi), sin(phi)));
    return phase;
}


/* The answer gets written into gateScratch */
__device__ void
tensor(sparse_elt* gate1, sparse_elt* gate2, sparse_elt* gateScratch, int gate_n, long gateLen, bool *successDevice) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    long num1 = gate1[0].u;
    long num2 = gate2[0].u;
    long dim1 = gate1[0].v;
    long dim2 = gate2[0].v;
    int numPerThread = max(num1 / (blockDim.x * gridDim.x), 1L);

    if(gate_n != 0 && (num1-1) * (num2 - 1) > gateLen) {
        if(id == 0) { *successDevice = false; }
        __syncthreads();
        return;
    }
    else{
        if(id == 0) { *successDevice = true; }
        __syncthreads();
    }

    if(id == 0) {
        gateScratch[0] = createElt((num1-1)*(num2-1)+1, dim1 * dim2, make_cuDoubleComplex(0.0, 0.0));
    }

    for(long i = (id * numPerThread) + 1; i < (((id + 1) * numPerThread) + 1) && i < num1; ++i) {
        for(long j = 1; j < num2; ++j) {
            gateScratch[(i-1)*(num2-1) + j] = 
                createElt(gate1[i].u * dim2 + gate2[j].u,
                          gate1[i].v * dim2 + gate2[j].v,
                          cuCmul(gate1[i].amp, gate2[j].amp));
        }
    }
}

__device__ void
copyGate(sparse_elt* dest, sparse_elt* U1){
    long num_gates = U1[0].u;
    for(int i = 0; i < num_gates; ++i){
        dest[i] = U1[i];
    }
}

__global__ void
tensorKernel(sparse_elt* gate, sparse_elt* gateScratch, long gateLen, 
                             OneQubitGateType whichGate, double param, const int u, int gate_n, bool *successDevice) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ sparse_elt* U1;
    __shared__ sparse_elt* I;
    if(threadIdx.x == 0) {
        I = identity();
        switch(whichGate){
            case HADAMARD:
            U1 = hadamard();
            break;
            
            case NOT:
            U1 = naught();
            break;
            
            case IDENTITY:
            U1 = identity();
            break;

            case PHASE:
            U1 = phase(param);
            break;
        }
    }
    __syncthreads();
    if(gate_n == 0) {
        if(id == 0) {
            *successDevice = true;
            if(u == 0) { copyGate(gateScratch, U1); }
            else { copyGate(gateScratch, I); }
        }
    }
    else {
        if(gate_n == u) { tensor(gate, U1, gateScratch, gate_n, gateLen, successDevice); }
        else { tensor(gate, I, gateScratch, gate_n, gateLen, successDevice); }
    }
    //if(id == 0) printOp(gateScratch);
}
/* Two qubit gates */

/* Does not need to check if gate has capacity enough, because CNOT has exactly 2^n nonzero
 * elements and that is the initial (and also minimal) size of the gate array. */
__global__ void
CNOTExpanded(sparse_elt* gate, const int n, const long N, const int u, const int v) { 
    gate[0] = createElt(N+1, N, make_cuDoubleComplex(0.0, 0.0));
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int numPerThread = max(N / (blockDim.x * gridDim.x), 1L);
    int j;
    for(int i = id * numPerThread; i < (id+1) * numPerThread && i < N; ++i) {
        if (i>>(n-u-1) & 0x1) {
            j = i ^ (0x1<<(n-v-1));
            gate[i+1] = createElt(i, j, make_cuDoubleComplex(1.0, 0.0));
        } else {
            gate[i+1] = createElt(i, i, make_cuDoubleComplex(1.0, 0.0));
        }
    }
}

__device__ void
mult(sparse_elt* U1, int whichIndex, cuDoubleComplex* ans) {
    ans[0] = make_cuDoubleComplex(0.0, 0.0);
    ans[1] = make_cuDoubleComplex(0.0, 0.0);
    for(int i = 1; i < U1[0].u; ++i) {
        if(U1[i].v == whichIndex){
            ans[U1[i].u] = U1[i].amp;
        }
    }
}
__global__ void
CU1ExpandedSetup(OneQubitGateType whichGate, double param, int* eltsPerThread, const int n, const long N, const int u, const int v, long gateLen, bool* successDevice) {
    
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int numPerThread = max(N / (blockDim.x * gridDim.x), 1L);

    __shared__ sparse_elt* U1;
    if(threadIdx.x == 0) {
        switch(whichGate){
            case HADAMARD:
            U1 = hadamard();
            break;
            
            case NOT:
            U1 = naught();
            break;
            
            case IDENTITY:
            U1 = identity();
            break;

            case PHASE:
            U1 = phase(param);
            break;
        }
    }

    if((N/4) * ((U1[0].u) + 1) > gateLen) {
        if(id == 0) { *successDevice = false; }
        return;
    } else {
        if(id == 0) { *successDevice = true; }
    }
    cuDoubleComplex stateOfV[2];
    int numEltsForThisThread = 0;
    for(long i = id * numPerThread; i < (id + 1) * numPerThread && i < N; ++i) {
        if (i>>(n-u-1) & 0x1) {
            mult(U1, (i>>(n-v-1)) & 0x1, stateOfV);
            if(!equalsZero(stateOfV[0])) {
                numEltsForThisThread++;
            }
            if(!equalsZero(stateOfV[1])) {
                numEltsForThisThread++;
            }
        } else {
            numEltsForThisThread++;
        }
    }
    eltsPerThread[id] = numEltsForThisThread;
}

__global__ void
CU1Expanded(sparse_elt* gate, OneQubitGateType whichGate, double param, int* startIndices, const int n, const long N, const int u, const int v) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int numPerThread = max(N / (blockDim.x * gridDim.x), 1L);
    
    __shared__ sparse_elt* U1;
    if(threadIdx.x == 0) {
        switch(whichGate){
            case HADAMARD:
            U1 = hadamard();
            break;
            
            case NOT:
            U1 = naught();
            break;
            
            case IDENTITY:
            U1 = identity();
            break;

            case PHASE:
            U1 = phase(param);
            break;
        }
    }
    
    cuDoubleComplex stateOfV[2];
    gate[0] = createElt(N+1, N, make_cuDoubleComplex(0.0, 0.0));
    int j = 0;
    for(long i = id * numPerThread; i < (id + 1) * numPerThread && i < N; ++i) {
        if (i>>(n-u-1) & 0x1) {
            mult(U1, (i>>(n-v-1)) & 0x1, stateOfV);
            if(!equalsZero(stateOfV[0])) {
                gate[startIndices[id]+j+1] = createElt(i, i & ~(0x1L<<(n-v-1)), stateOfV[0]);
                j++;
            }
            if(!equalsZero(stateOfV[1])) {
                gate[startIndices[id]+j+1] = createElt(i, i | (0x1L<<(n-v-1)), stateOfV[1]);
                j++;
            }
        } else {
            gate[startIndices[id]+j+1] = createElt(i, i, make_cuDoubleComplex(1.0, 0.0));
            j++;
        }
    }
}

__global__ void
SWAPExpanded(sparse_elt* gate, const int n, const long N, const int u, const int v) {
    assert(u != v);
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int numPerThread = max(N / (blockDim.x * gridDim.x), 1L);
    //dimensions
    gate[0] = createElt(N+1, N, make_cuDoubleComplex(0.0, 0.0));
    // values
    int j;
    for(int i = id * numPerThread; i < (id+1) * numPerThread && i < N; ++i) {
        if ((i>>(n-u-1) & 0x1) ^ (i>>(n-v-1) & 0x1)) {
            j = (i ^ (0x1<<(n-u-1))) ^ (0x1<<(n-v-1));
            gate[i+1] = createElt(i, j, make_cuDoubleComplex(1.0, 0.0));
        } else {
            gate[i+1] = createElt(i, i, make_cuDoubleComplex(1.0, 0.0));
        }
    }
}


/* Call swapPtrs after every applyOperator and every tensor product. */
void swapPtrsState(cuDoubleComplex** state, cuDoubleComplex** stateScratch, const int N, const int blocks, const int threadsPerBlock) {
    cudaDeviceSynchronize();
    cuDoubleComplex* temp = *state;
    *state = *stateScratch;
    *stateScratch = temp;
    assignAll<<<blocks, threadsPerBlock>>>(*stateScratch, N, make_cuDoubleComplex(0.0,0.0));
    cudaDeviceSynchronize();
}
void swapPtrsGate(sparse_elt** state, sparse_elt** stateScratch) {
    cudaDeviceSynchronize();
    sparse_elt* temp = *state;
    *state = *stateScratch;
    *stateScratch = temp;
}

__global__ void
applyOperatorKernel(cuDoubleComplex* state, cuDoubleComplex* stateScratch, sparse_elt* gate) {

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    long numElts = gate[0].u;
    int numPerThread = max(numElts / (blockDim.x * gridDim.x), 1L);
    for(int i = (id * numPerThread) + 1; i < ((id+1)*numPerThread) + 1 && i < numElts; ++i) {
        atomicComplexAdd(&(stateScratch[gate[i].u]), cuCmul(gate[i].amp, state[gate[i].v]));
    }
}

void applyOneQubitOperator(cuDoubleComplex** state, cuDoubleComplex** stateScratch, sparse_elt** gate, sparse_elt** gateScratch,
                           const int n, const long N, long* gateLen,
                           const int u, OneQubitGateType whichGate, double param,
                           const int blocks, const int threadsPerBlock) {
    bool *successDevice;
    cudaMalloc((void**)&successDevice, sizeof(bool));
    bool success = true;
    cudaMemcpy(successDevice, &success, sizeof(bool), cudaMemcpyHostToDevice);
    int gate_n = 0;
    do {
        if (!success) {
            cudaFree(*gate);
            cudaFree(*gateScratch);
            *gateLen = (*gateLen) * 2;
            cudaMalloc((void**)gate, (*gateLen + 1) * sizeof(sparse_elt));
            cudaMalloc((void**)gateScratch, (*gateLen + 1) * sizeof(sparse_elt));
            gate_n = 0;
        }
        tensorKernel<<<blocks, threadsPerBlock>>> (*gate, *gateScratch, *gateLen, whichGate, param, u, gate_n, successDevice);
        cudaDeviceSynchronize();
        swapPtrsGate(gate, gateScratch);
        cudaMemcpy(&success, successDevice, sizeof(bool), cudaMemcpyDeviceToHost);
        gate_n++;
    } while (!(success && gate_n >= n));
    applyOperatorKernel<<<blocks, threadsPerBlock>>> (*state, *stateScratch, *gate);
    cudaDeviceSynchronize();
    swapPtrsState(state, stateScratch, N, blocks, threadsPerBlock);
    cudaFree(successDevice);
}

void applyTwoQubitOperator(cuDoubleComplex** state, cuDoubleComplex** stateScratch, sparse_elt** gate, sparse_elt** gateScratch,
                           const int n, const long N, long* gateLen,
                           const int u, const int v, TwoQubitGateType whichGate, double param,
                           const int blocks, const int threadsPerBlock) {
    bool *successDevice;
    cudaMalloc((void**)&successDevice, sizeof(bool));
    bool success = true;
    cudaMemcpy(successDevice, &success, sizeof(bool), cudaMemcpyHostToDevice);
    
    switch(whichGate){
        case CNOT:
        CNOTExpanded<<<blocks, threadsPerBlock>>>(*gate, n, N, u, v);
        gpuErrchk(cudaDeviceSynchronize());
        break;
        
        case CPHASE:
        int *numPerThread;
        cudaMalloc((void**)&numPerThread, sizeof(int) * blocks * threadsPerBlock);
        do {
            if (!success) {
                cudaFree(*gate);
                cudaFree(*gateScratch);
                *gateLen = (*gateLen) * 2;
                cudaMalloc((void**)gate, (*gateLen + 1) * sizeof(sparse_elt));
                cudaMalloc((void**)gateScratch, (*gateLen + 1) * sizeof(sparse_elt));
            }
            CU1ExpandedSetup<<<blocks, threadsPerBlock>>> (PHASE, param, numPerThread, n, N, u, v, *gateLen, successDevice);
            gpuErrchk(cudaDeviceSynchronize());
            cudaMemcpy(&success, successDevice, sizeof(bool), cudaMemcpyDeviceToHost);
        } while (!success);

        int *startIndices;
        cudaMalloc((void**)&startIndices, sizeof(int) * blocks * threadsPerBlock);
        cudaScan(numPerThread, (blocks * threadsPerBlock), startIndices);
        gpuErrchk(cudaDeviceSynchronize());
        CU1Expanded<<<blocks, threadsPerBlock>>>(*gate, PHASE, param, startIndices, n, N, u, v);
        gpuErrchk(cudaDeviceSynchronize());
        cudaFree(numPerThread);
        cudaFree(startIndices);
        break;
        
        case SWAP:
        SWAPExpanded<<<blocks, threadsPerBlock>>>(*gate, n, N, u, v);
        break;
    }
    applyOperatorKernel<<<blocks, threadsPerBlock>>> (*state, *stateScratch, *gate);
    gpuErrchk(cudaDeviceSynchronize());
    swapPtrsState(state, stateScratch, N, blocks, threadsPerBlock);
    cudaFree(successDevice);
}



/* Pi estimation functions for benchmarking performance. */

/* Inverse quantum fourier transform */

void qft_dagger(cuDoubleComplex** state, cuDoubleComplex** stateScratch, sparse_elt** gate, sparse_elt** gateScratch,
                long* gateLen, const int n, const int blocks, const int threadsPerBlock) {
    const int n_prime = n+1;
    const long N = pow(2, n_prime);
    for (int i = 0; i < n/2; ++i) {
        applyTwoQubitOperator(state, stateScratch, gate, gateScratch, n_prime, N, gateLen, i, n-i-1, SWAP, NO_PARAM, blocks, threadsPerBlock);   
        gpuErrchk(cudaDeviceSynchronize());
    }


    for (int j = 0; j < n; ++j) {
        for (int m = 0; m < j; ++m) {
            double phi = -M_PI / ((double)pow(2, j - m));
            applyTwoQubitOperator(state, stateScratch, gate, gateScratch, n_prime, N, gateLen, j, m, CPHASE, phi, blocks, threadsPerBlock); 
            gpuErrchk(cudaDeviceSynchronize());
            cout << "IsNormalized inner " << isNormalized(*state, N, blocks, threadsPerBlock) << endl;
        }
        
        applyOneQubitOperator(state, stateScratch, gate, gateScratch, n_prime, N, gateLen, j, HADAMARD, NO_PARAM, blocks, threadsPerBlock);
        gpuErrchk(cudaDeviceSynchronize()); 
        //cout << "IsNormalized after had " << isNormalized(*state, N, blocks, threadsPerBlock) << endl;
    }
}

// Setup for quantum phase estimation. 
void qpe_pre(cuDoubleComplex** state, cuDoubleComplex** stateScratch, sparse_elt** gate, sparse_elt** gateScratch,
                         long* gateLen, const int n, const int blocks, const int threadsPerBlock) {
    const int n_prime = n+1;
    const long N = pow(2, n_prime);

    for (int i = 0; i < n; ++i) {
        applyOneQubitOperator(state, stateScratch, gate, gateScratch, n_prime, N, gateLen, i, HADAMARD, NO_PARAM, blocks, threadsPerBlock); 
        gpuErrchk(cudaDeviceSynchronize());
    }
    applyOneQubitOperator(state, stateScratch, gate, gateScratch, n_prime, N, gateLen, n, NOT, NO_PARAM, blocks, threadsPerBlock); 
    gpuErrchk(cudaDeviceSynchronize());

    for (int i = n-1; i >= 0; --i) {
        for (int j = 0; j < pow(2, n-i-1); ++j) {
            applyTwoQubitOperator(state, stateScratch, gate, gateScratch, n_prime, N, gateLen, n, n-i-1, CPHASE, 1.0, blocks, threadsPerBlock); 
            gpuErrchk(cudaDeviceSynchronize());
        }
    }
}

/* The bits we want for this task are from 1 to n inclusive, since the 0 bit is our extra for
 * setting up the problem. Additionally, we want to read them in reverse. */
long getCorrectBitsForPiEstimate(long bits, int n){
    long answer = 0;
    for(int i = 1; i <= n; i++){
        answer = answer<<1;
        answer += (bits>>i) & 0x1L;
    }
    return answer;
}

double gpu_get_pi_estimate(const int n, const int blocks, const int threadsPerBlock){
    const int n_prime = n+1;
    const long N = pow(2, n_prime);
    cuDoubleComplex* state = classical(N, 0, blocks, threadsPerBlock);
    cuDoubleComplex* stateScratch = zero(N, blocks, threadsPerBlock);
    sparse_elt* gate;
    sparse_elt* gateScratch;
    cudaMalloc((void**)&gate, (N + 1) * sizeof(sparse_elt));
    cudaMalloc((void**)&gateScratch, (N + 1) * sizeof(sparse_elt));
    long gateLen = N;

    qpe_pre(&state, &stateScratch, &gate, &gateScratch, &gateLen, n, blocks, threadsPerBlock);
    qft_dagger(&state, &stateScratch, &gate, &gateScratch, &gateLen, n, blocks, threadsPerBlock);
    
    long mostLikely = getMostLikelyHost(state, n_prime, N, blocks, threadsPerBlock);
    cout << mostLikely << endl;
    double theta = (double)(getCorrectBitsForPiEstimate(mostLikely, n)) / pow(2.0, n);
    
    //printVec(state, n_prime, N, false);
    //printVecProbs(state, n_prime, N, false);
    
    cout << "IsNormalized " << isNormalized(state, N, blocks, threadsPerBlock) << endl;
    cudaDeviceSynchronize();
    cudaFree(state);
    cudaFree(stateScratch);
    cudaFree(gate);
    cudaFree(gateScratch);
    return 1.0 / (2 * theta);
}

int main(){
    int n = 5;
    long N = pow(2, n+1); //Just for this purpose since we have an extra qubit in the circuit.
    int threadsPerBlock = 64;
    int blocks = (N + threadsPerBlock - 1)/threadsPerBlock;

    //gpu time includes time for allocation, transfer, and execution
    double gpu_start_time = CycleTimer::currentSeconds();
    cout << gpu_get_pi_estimate(n, blocks, threadsPerBlock) << endl;
    double gpu_end_time = CycleTimer::currentSeconds();
    
    //TODO Add Bandwidth Calculation
    printf("Total GPU Time: %.3f ms\n", 1000.f * (gpu_end_time-gpu_start_time));

    return 0;
}
