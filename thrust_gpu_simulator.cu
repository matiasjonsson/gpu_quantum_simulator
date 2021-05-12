#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <complex>
#include <cusparse.h>
using namespace std;

// Sparse matrix reserves the first element for dimension, assuming matrix is square.
// It is implemented using a vector.
struct sparse_elt {
    long u;
    long v;
    complex<double> amp;
};

typedef enum GateName{
    HADAMARD,
    IDENTITY,
    CNOT,
    NOT,
    CU11,
    SWAP
    // Add more gates as necessary
} GateName;


struct gate {
    GateName name;
    vector<int> operands;
};


struct my_complex_norm {
    template <typename T>
    __host__ __device__
    T operator()(thrust::complex<T> &d){
         return thrust::norm(d);
    }
};

class increment
{
    private:
            int num;
    public:
            increment(int n) : num(n) {  }

                    // This operator overloading enables calling
                    // operator function () on objects of increment
            int operator () (int arr_num) const {
                return num + arr_num;
            }
};


__global__
void createElt(int u, int v, complex<double> amp, sparse_elt &elt) {
    elt.u = u;
    elt.v = v;
    elt.amp = amp;
}


bool equalsZero(complex<double> amplitude) {
    return (norm(amplitude) < 1e-15);
}
/*
//__global__
bool isNormalized (thrust::device_vector<complex<double> > state) {
    double sum = 0;
    for (auto i = state.begin(); i != state.end(); ++i) {
        sum += norm(*i);
    }
    return (abs(1.0 - sum) < 1e-15);
//}

//TODO FIX Norm
bool isNormalized (thrust::device_vector<complex<double> > state) {
    double sum = 0;
    thrust::device_vector<double> out (state.size());

    thrust::transform(state.begin(), state.end(), out.begin(), my_complex_norm());

    thrust::counting_iterator<int> first(0);

    thrust::counting_iterator<int> last = first + state.size();

    thrust::reduce(first, last);
    //for (int i = 0; i < state.size(); i++) sum += my_complex_norm(state[i]);

    return (abs(1.0 - sum) < 1e-15);
}


__global__
void uniform(thrust::device_vector<complex<double> > &state, const long N) {
    thrust::reduce(first, last)
    thrust::fill(state.begin(), state.end(), sqrt(1.0/N));
}

__global__
void zero(thrust::device_vector<complex<double> > state,  const long N) {
    thrust::fill(state.begin(), state.end(), 0);
}

__global__
void classical(thrust::device_vector<complex<double> > state, const long N, const int init) {
    thrust::fill(state.begin(), state.end(), 0);
    state.data()[init] = 1.0;
}

//__global__
thrust::device_vector<complex<double> > applyOperator(thrust::device_vector<complex<double> > state, thrust::device_vector<sparse_elt> unitary) {
    //assert(unitary.begin()->u == state.size());
    thrust::device_vector<complex<double> > newstate = zero(state.size());
    for(auto i = unitary.begin() + 1; i != unitary.end(); ++i) {
        newstate.data()[i->u] += i->amp * state.data()[i->v];
    }
    return newstate;
}

//__global__
thrust::device_vector<sparse_elt> tensor(thrust::device_vector<sparse_elt> u1, thrust::device_vector<sparse_elt> u2) {
    int dim1 = u1.begin()->u;
    int dim2 = u2.begin()->u;
    thrust::device_vector<sparse_elt> u3;
    u3.push_back(createElt(dim1*dim2, dim1*dim2, 0.0));
    for(auto i = u1.begin() + 1; i != u1.end(); ++i) {
        for(auto j = u2.begin() + 1; j != u2.end(); ++j) {
            u3.push_back(createElt(i->u * dim2 + j->u, i->v * dim2 + j->v, i->amp * j->amp));
        }
    }
    return u3;
}

// Print features

__global__
void printComplex(complex<double> amplitude) {
    cout << "(" << real(amplitude) << "+" << imag(amplitude) << "i)";
}

__global__
void printVec(thrust::host_vector<complex<double> > state, const int n, const long N, bool printAll) {
    cout << "State:\n";
    for (int i = 0; i < N; ++i) {
        if (!printAll && equalsZero(state.at(i)))
            continue;
        printComplex(state.at(i));
        cout << "|";
        for (int j = n-1; j >= 0; --j) {
            cout << (0x1 & (i>>j));
        }
        cout << "> + ";
        cout << "\n";
    }
}

__global__
void printVecProbs(thrust::host_vector<complex<double> > state, const int n, const long N, bool printAll) {
    cout << "Probability distribution:\n";
    for (int i = 0; i < N; ++i) {
        if (!printAll && equalsZero(state.at(i)))
            continue;

        cout << "|<";
        for (int j = n-1; j >= 0; --j) {
            cout << (0x1 & (i>>j));
        }
        cout << "|Psi>|^2 = " << norm(state.at(i)) << "\n";
    }
}


__global__
void printOp(thrust::host_vector<sparse_elt> U) {
    thrust::vector<thrust::vector<complex<double> > > densified;
    int dim = U.begin()->u;
    densified.assign(dim, zero(dim));
    for (auto x = U.begin()+1; x != U.end(); ++x) {
        densified.data()[x->u].data()[x->v] += x->amp;
    }
    for (int x = 0; x < dim; ++x) {
        for (int y = 0; y < dim; ++y) {
            cout << densified.data()[x].data()[y] << " ";
        }
        cout << "\n";
    }
}


__global__
long getMostLikely(thrust::device_vector<complex<double> > state, int n) {
    long ans = -1;
    double prob = 0.0;
    long N = pow(2, n);
    for (long i = 0; i < N; i++) {
        if (norm(state.data()[i]) > prob) {
            prob = norm(state.data()[i]);
            ans = i;
        }
    }
    return ans;
}


// One qubit gates
__global__
thrust::device_vector<sparse_elt> identity(){
    thrust::vector<sparse_elt> id;
    // dimensions
    id.push_back(createElt(2, 2, 0.0));
    // values
    id.push_back(createElt(0, 0, 1.0));
    id.push_back(createElt(1, 1, 1.0));
    return id;
}

__global__
thrust::device_vector<sparse_elt> naught() {
    thrust::vector<sparse_elt> naught;
    // dimensions
    naught.push_back(createElt(2, 2, 0.0));
    // values
    naught.push_back(createElt(1, 0, 1.0));
    naught.push_back(createElt(0, 1, 1.0));
    return naught;
}

__global__
thrust::device_vector<sparse_elt> hadamard() {
    thrust::vector<sparse_elt> had;
    // dimensions
    had.push_back(createElt(2, 2, 0.0));
    // values
    double norm = 1.0 / sqrt(2);
    had.push_back(createElt(0, 0, norm));
    had.push_back(createElt(0, 1, norm));
    had.push_back(createElt(1, 0, norm));
    had.push_back(createElt(1, 1, -norm));
    return had;
}

__global__
thrust::device_vector<sparse_elt> phase(double phi){
    thrust::vector<sparse_elt> phase;
    // dimensions
    phase.push_back(createElt(2,2, 0.0));
    // values
    phase.push_back(createElt(0,0, 1.0));
    complex<double> ex (cos(phi), sin(phi));
    phase.push_back(createElt(1,1,ex));
    return phase;
}


// Expanding operations for 1 qubit gates.

// It is much more efficient to do many 1 qubit gates sequentially than to compute their tensor product
__global__
thrust::device_vector<sparse_elt> oneQubitGateExpand(thrust::device_vector<sparse_elt> oneQubitGate, const int n, const int which) {
    thrust::vector<sparse_elt> total = (which == 0 ? oneQubitGate : identity());
    for(int i = 1; i < n; ++i) {
        if(which == i)
            total = tensor(total, oneQubitGate);
        else
            total = tensor(total, identity());
    }
    return total;
}


// Two qubit gates

__global__
thrust::device_vector<sparse_elt> CNOTExpanded(const int n, const int u, const int v) {
    thrust::vector<sparse_elt> cnot;
    long N = pow(2, n);
    cnot.push_back(createElt(N, N, 0.0));
    int j;
    for(int i = 0; i < N; ++i) {
        if (i>>(n-u-1) & 0x1) {
            j = i ^ (0x1<<(n-v-1));
            cnot.push_back(createElt(i, j, 1.0));
        } else {
            cnot.push_back(createElt(i, i, 1.0));
        }
    }
    return cnot;
}
__global__
thrust::device_vector<sparse_elt> CU1Expanded(thrust::device_vector<sparse_elt> U1, const int n, const int u, const int v) {
    thrust::device_vector<sparse_elt> CU1;
    long N = pow(2, n);
    CU1.push_back(createElt(N, N, 0.0));
    thrust::device_vector<complex<double> > stateOfV;
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
__global__
thrust::vector<sparse_elt> CPhase(const int n, const int u, const int v, double phi) {
    return CU1Expanded(phase(phi), n, u, v);
}


__global__
thrust::vector<sparse_elt> swapExpanded(const int n, const int u, const int v) {
    assert(u != v);
    thrust::vector<sparse_elt> swp;
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

// Pi estimation functions for benchmarking performance.

// Inverse quantum fourier transform

__global__
thrust::vector<complex<double> > qft_dagger(thrust::vector<complex<double> > state, const int n) {
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
}

// Setup for quantum phase estimation.
__global__
thrust::vector<complex<double> > qpe_pre(const int n){
    const long N = pow(2, n+1);
    const int n_prime = n+1;
    thrust::vector<complex<double> > state = classical(N, 0);
    for (int i = 0; i < n; ++i) {
        state = applyOperator(state, oneQubitGateExpand(hadamard(), n_prime, i));
    }
    state = applyOperator(state, oneQubitGateExpand(naught(), n_prime, n));

    for (int i = n-1; i >= 0; --i) {
        for (int j = 0; j < pow(2, n-i-1); ++j) {
            state = applyOperator(state, CPhase(n_prime, n, n-i-1, 1.0));
        }
    }
    return state;
}

// The bits we want for this task are from 1 to n inclusive, since the 0 bit is our extra for
// setting up the problem. Additionally, we want to read them in reverse.

__global__
long getCorrectBitsForPiEstimate(long bits, int n){
    long answer = 0;
    for(int i = 1; i <= n; i++){
        answer = answer<<1;
        answer += (bits>>i) & 0x1L;
    }
    return answer;
}

__global__
double get_pi_estimate(const int n) {
    thrust::vector<complex<double> > state = qpe_pre(n);
    state = qft_dagger(state, n);
    long mostLikely = getMostLikely(state, n+1);
    double theta = (double)(getCorrectBitsForPiEstimate(mostLikely, n)) / pow(2.0,n);
    return 1.0 / (2 * theta);
}

int main() {
    const int n = 10;
    //const int N = pow(2, n);

    cout << get_pi_estimate(n) << endl;



    //Testing functions out
    vector<complex<double> > state = classical(N, 0);


    // Uniform superposition:
    for (int i = 0; i < n; ++i) {
        state = applyOperator(state, oneQubitGateExpand(hadamard(), n, i));
    }
    //state = applyOperator(state, oneQubitGateExpand(hadamard(), n, 0));
    state = applyOperator(state, oneQubitGateExpand(naught(), n, 0));
    state = applyOperator(state, CPhase(n, 0,1, 0.1));

    printVec(state, n, N, false);
    //printVecProbs(state, n, N, false);
}

double get_pi_estimate(const int n) {
    thrust::vector<complex<double> > state = qpe_pre(n);
    state = qft_dagger(state, n);
    long mostLikely = getMostLikely(state, n+1);
    double theta = (double)(getCorrectBitsForPiEstimate(mostLikely, n)) / pow(2.0,n);
    return 1.0 / (2 * theta);
}
*/

int main(){
    printf("Trying something anything");


    cudaStat1 = cudaMalloc((void**)&z, 2*(n+1)*sizeof(z[0]));
    if (cudaStat1 != cudaSuccess) {
            CLEANUP("Device malloc failed (z)");
                return 1;
    }
    cudaStat1 = cudaMemset((void *)z,0, 2*(n+1)*sizeof(z[0]));
    if (cudaStat1 != cudaSuccess) {
            CLEANUP("Memset on Device failed");
                return 1;
    }
    status= cusparseDcsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, 2, n,
                                   nnz, &dfive, descr, cooVal, csrRowPtr, cooColIndex,
                                                          y, n, &dzero, z, n+1);
    if (status != CUSPARSE_STATUS_SUCCESS) {
            CLEANUP("Matrix-matrix multiplication failed");
                return 1;
    }







    return 0;
}


