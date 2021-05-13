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


using namespace std;

typedef enum OneQubitGateType{
    IDENTITY,
    NOT,
    HADAMARD,
    PHASE
} OneQubitGateType;

#define NO_PARAM 0.0 

typedef enum TwoQubitGateType{
    CNOT,
    CU1,
    CPhase
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
__device__ void
maxReductionHelper(int id, unsigned long long int *vec, unsigned long long int *ans) {
    __shared__ unsigned long long int max;
    max = 0;
    atomicMax(&max, vec[id]);
    __syncthreads();
    if(threadIdx.x == 0) {
        atomicMax(ans, max);
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
    totalMag<<<blocks, threadsPerBlock>>> (state, N, temp, ansDevice);
    double ansHost = 0.0;
    cudaMemcpy(&ansHost, ansDevice, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(temp);
    cudaFree(ansDevice);
    return ansHost;//(abs(ansHost - 1.0) < 1e-13);
}

cuDoubleComplex* uniform(const long N, const int blocks, const int threadsPerBlock) {
    cuDoubleComplex* state;
    cudaMalloc((void **)&state, sizeof(cuDoubleComplex) * N);
    assignAll<<<blocks, threadsPerBlock>>> (state, N, make_cuDoubleComplex(sqrt(1.0/N),0.0));
    
    return state;
}
cuDoubleComplex* zero(const long N, const int blocks, const int threadsPerBlock) {
    cuDoubleComplex* state;
    cudaMalloc((void **)&state, sizeof(cuDoubleComplex) * N);
    assignAll<<<blocks, threadsPerBlock>>>(state, N, make_cuDoubleComplex(0.0, 0.0));
    return state;
}

cuDoubleComplex* classical(const long N, const long which, const int blocks, const int threadsPerBlock) {
    cuDoubleComplex* state;
    cudaMalloc((void **)&state, sizeof(cuDoubleComplex) * N);
    assignOne<<<blocks, threadsPerBlock>>>(state, N, make_cuDoubleComplex(1.0, 0.0), which);
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
getMostLikely(cuDoubleComplex* state, unsigned long long int* temp, const int n, const long N, unsigned long long int *ans) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int numPerThread = max(N / (blockDim.x * gridDim.x), 1L);
    if(id == 0) {
        *ans = 0;
    }
    double prob = 0.0;

    for (long i = id * numPerThread; i < ((id+1) * numPerThread) && i < N; i++) {
        if (norm(state[i]) > prob) {
            prob = norm(state[i]);
            temp[id] = i;
        }
    }
    __syncthreads();
    if(id < N) {
        maxReductionHelper(id, temp, ans);
    }
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
    phase[0] = createElt(2, 2, make_cuDoubleComplex(0.0, 0.0));
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
    if(gate_n != 0 && (num1-1) * (num2 - 1) > gateLen) {
        *successDevice = false;
        __syncthreads();
        return;
    }
    else{
        *successDevice = true;
        __syncthreads();
    }

    if(id == 0) {
        gateScratch[0] = createElt((num1-1) * (num2-1) + 1, dim1 * dim2, make_cuDoubleComplex(0.0, 0.0));
    }

    int numPerThread = max(num1 / (blockDim.x * gridDim.x), 1L);

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
    if(id == 0) {
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
}

/* Two qubit gates */

__device__ void
CNOTExpanded(sparse_elt* gate, const int n, const long N, const int u, const int v) { 
    gate[0] = createElt(N, N, make_cuDoubleComplex(0.0, 0.0));
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int numPerThread = max(N / (blockDim.x * gridDim.x), 1L);
    int j;
    for(int i = id * numPerThread; i < (id+1) * numPerThread && i < N; ++i) {
        if (i>>(n-u-1) & 0x1) {
            j = i ^ (0x1<<(n-v-1));
            gate[i] = createElt(i, j, make_cuDoubleComplex(1.0, 0.0));
        } else {
            gate[i] = createElt(i, i, make_cuDoubleComplex(1.0, 0.0));
        }
    }
}
/*
vector<sparse_elt> CU1Expanded(vector<sparse_elt> U1, const int n, const int u, const int v) {
    vector<sparse_elt> CU1;
    long N = pow(2, n);
    CU1.push_back(createElt(N, N, 0.0));
    vector<complex<double> > stateOfV;
    for(long i = 0; i < N; ++i) {
        if (i>>(n-u-1) & 0x1) {
            stateOfV = classical(2, (i>>(n-v-1)) & 0x1);
            stateOfV = applyOperator(stateOfV, U1);
            if(!equalsZero(stateOfV.data()[0])) {
                CU1.push_back(createElt(i, i & ~(0x1L<<(n-v-1)), stateOfV.data()[0]));
            }
            if(!equalsZero(stateOfV.data()[1])) {
                CU1.push_back(createElt(i, i | (0x1L<<(n-v-1)), stateOfV.data()[1]));
            }
        } else {
            CU1.push_back(createElt(i, i, 1.0));
        }
    }
    return CU1;
}
vector<sparse_elt> CPhase(const int n, const int u, const int v, double phi) {
    return CU1Expanded(phase(phi), n, u, v);
}

vector<sparse_elt> swapExpanded(const int n, const int u, const int v) {
    assert(u != v);
    vector<sparse_elt> swp;
    long N = pow(2, n);
    //dimensions
    swp.push_back(createElt(N, N, 0.0));
    // values
    int j;
    for(int i = 0; i < N; ++i) {
        if ((i>>(n-u-1) & 0x1) ^ (i>>(n-v-1) & 0x1)) {
            j = (i ^ (0x1<<(n-u-1))) ^ (0x1<<(n-v-1));
            swp.push_back(createElt(i, j, 1.0));
        } else {
            swp.push_back(createElt(i, i, 1.0));
        }
    }
    return swp;
}
*/

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
        swapPtrsGate(gate, gateScratch);
        cudaMemcpy(&success, successDevice, sizeof(bool), cudaMemcpyDeviceToHost);
        gate_n++;
    } while (!(success && gate_n >= n));
    applyOperatorKernel<<<blocks, threadsPerBlock>>> (*state, *stateScratch, *gate);
    swapPtrsState(state, stateScratch, N, blocks, threadsPerBlock);
}

/*void applyTwoQubitOperator(cuDoubleComplex** state, cuDoubleComplex** stateScratch, sparse_elt** gate,
                           const int n, const long N, long* gateLen,
                           const int u, TwoQubitGateType whichGate, double param,
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
            *gateLen = 2 * (*gateLen);
            cudaMalloc((void**)gate, (*gateLen + 1) * sizeof(sparse_elt));
            cudaMalloc((void**)gateScratch, (*gateLen + 1) * sizeof(sparse_elt));
            success = true;
            gate_n = 0;
        }
        tensorKernel<<<blocks, threadsPerBlock>>> (*gate, *gateScratch, *gateLen, whichGate, param, u, gate_n, successDevice);
        swapPtrsGate(gate, gateScratch);
        cudaMemcpy(&success, successDevice, sizeof(bool), cudaMemcpyDeviceToHost);
        gate_n++;
    } while (!(success && gate_n >= n));
    applyOperatorKernel<<<blocks, threadsPerBlock>>> (*state, *stateScratch, *gate);
    swapPtrsState(state, stateScratch, N, blocks, threadsPerBlock);
}*/



/* Pi estimation functions for benchmarking performance. */

/* Inverse quantum fourier transform */
/*
vector<complex<double> > qft_dagger(vector<complex<double> > state, const int n) {
    const int n_prime = n+1;
    for (int i = 0; i < n/2; ++i) {
        state = applyOperator(state, swapExpanded(n_prime, i, n-i-1));    
    }
    for (int j = 0; j < n; ++j) {
        for (int m = 0; m < j; ++m) {
            double phi = -M_PI / ((double)pow(2, j - m));
            state = applyOperator(state, CPhase(n_prime, j, m, phi)); 
        }
        state = applyOperator(state, oneQubitGateExpand(hadamard(), n_prime, j));
    }
    return state;
}*/

// Setup for quantum phase estimation. 
cuDoubleComplex* qpe_pre(const int n, const int blocks, const int threadsPerBlock){
    const int n_prime = n+1;
    const long N = pow(2, n_prime);
    cuDoubleComplex* state = classical(N, 0, blocks, threadsPerBlock);
    /*for (int i = 0; i < n; ++i) {
        state = applyOperator(state, oneQubitGateExpand(hadamard(), n_prime, i));    
    }
    state = applyOperator(state, oneQubitGateExpand(naught(), n_prime, n));

    for (int i = n-1; i >= 0; --i) {
        for (int j = 0; j < pow(2, n-i-1); ++j) {
            state = applyOperator(state, CPhase(n_prime, n, n-i-1, 1.0));
        }
    }*/
    return state;
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

int gpu_get_pi_estimate(const int n, const int N, const int blocks, const int threadsPerBlock){
    //cuDoubleComplex* state = uniform(N, blocks, threadsPerBlock);
    cuDoubleComplex* state = classical(N, 0, blocks, threadsPerBlock);
    cuDoubleComplex* stateScratch = zero(N, blocks, threadsPerBlock);
    sparse_elt* gate;
    sparse_elt* gateScratch;
    cudaMalloc((void**)&gate, (N + 1) * sizeof(sparse_elt));
    cudaMalloc((void**)&gateScratch, (N + 1) * sizeof(sparse_elt));
    long gateLen = N;
    //cuDoubleComplex* state = qpe_pre(n, N, blocks, threadsPerBlock);
    cudaDeviceSynchronize();
    applyOneQubitOperator(&state, &stateScratch, &gate, &gateScratch, n, N, &gateLen, 2, HADAMARD, NO_PARAM, blocks, threadsPerBlock); 
    
    printVec(state, n, N, false);
    //printVecProbs(state, n, N, false);
    
    
    //cout << isNormalized(state, N, blocks, threadsPerBlock) << endl;
    cudaFree(state);
    cudaFree(stateScratch);
    cudaFree(gate);
    cudaFree(gateScratch);
    cudaDeviceSynchronize();
    return 0;
}

int main(){
    int n = 4;
    long N = pow(2, n);
    int threadsPerBlock = 256;
    int blocks = (threadsPerBlock + N - 1)/threadsPerBlock;
    
    //gpu time includes time for allocation, transfer, and execution
    double gpu_start_time = CycleTimer::currentSeconds();
    gpu_get_pi_estimate(n, N, blocks, threadsPerBlock);
    double gpu_end_time = CycleTimer::currentSeconds();
    
    //TODO Add Bandwidth Calculation
    printf("Total GPU Time: %.3f ms\n", 1000.f * (gpu_end_time-gpu_start_time));
    /*
    double cpu_start_time = CycleTimer::currentSeconds();
    get_pi_estimate(n, N, blocks, threadsPerBlock);
    double cpu_end_time = CycleTimer::currentSeconds();

    printf("Total CPU Time: %.3f ms\t\t", 1000.f * (cpu_end_time-cpu_start_time));
    */

    return 0;
}
