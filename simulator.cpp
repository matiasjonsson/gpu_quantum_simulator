#include <iostream>
#include <cmath>
#include <vector>
#include <complex>
#include <assert.h>
#include <bitset>
using namespace std;

// Sparse matrix reserves the first element for dimension, assuming matrix is square.
// It is implemented using a vector.
struct sparse_elt {
    int u;
    int v;
    complex<double> amp;
};

sparse_elt createElt(int u, int v, complex<double> amp) {
    sparse_elt elt;
    elt.u = u;
    elt.v = v;
    elt.amp = amp;
    return elt;
}

bool equalsZero(complex<double> amplitude) {
    return (norm(amplitude) < 1e-15);
}

bool isNormalized (vector<complex<double> > state) {
    double sum = 0;
    for (auto i = state.begin(); i != state.end(); ++i) {
        sum += norm(*i);
    }
    return (abs(1.0 - sum) < 1e-15);
}

vector<complex<double> > uniform(const int N) {
    vector<complex<double> > state;
    state.assign(N, sqrt(1.0/N));
    return state;
}
vector<complex<double> > zero(const int N) {
    vector<complex<double> > state;
    state.assign(N, 0);
    return state;
}
vector<complex<double> > classical(const int N, const int init) {
    vector<complex<double> > state = zero(N);
    state.data()[init] = 1.0;
    return state;
}

vector<complex<double> > applyOperator(vector<complex<double> > state, vector<sparse_elt> unitary) {
    assert(unitary.begin()->u == state.size());
    vector<complex<double> > newstate = zero(state.size());
    for(auto i = unitary.begin() + 1; i != unitary.end(); ++i) {
        newstate.data()[i->u] += i->amp * state.data()[i->v];
    }
    return newstate;
}

vector<sparse_elt> tensor(vector<sparse_elt> u1, vector<sparse_elt> u2) {
    int dim1 = u1.begin()->u;
    int dim2 = u2.begin()->u;
    vector<sparse_elt> u3;
    u3.push_back(createElt(dim1*dim2, dim1*dim2, 0.0));
    for(auto i = u1.begin() + 1; i != u1.end(); ++i) {
        for(auto j = u2.begin() + 1; j != u2.end(); ++j) {
            u3.push_back(createElt(i->u * dim2 + j->u, i->v * dim2 + j->v, i->amp * j->amp));
        }
    }
    return u3;
}

/* Print features */

void printComplex(complex<double> amplitude) {
    cout << "(" << real(amplitude) << "+" << imag(amplitude) << "i)";
}
void printVec(vector<complex<double> > state, const int n, const int N, bool printAll) {
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
void printVecProbs(vector<complex<double> > state, const int n, const int N, bool printAll) {
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
void printOp(vector<sparse_elt> U) {
    vector<vector<complex<double> > > densified;
    int dim = U.begin()->u;
    densified.assign(dim, zero(dim));
    for (auto x = U.begin()+1; x != U.end(); x++) {
        densified.data()[x->u].data()[x->v] += x->amp;
    }
    for (int x = 0; x < dim; x++) {
        for (int y = 0; y < dim; y++) {
            cout << densified.data()[x].data()[y] << " ";
        }
        cout << "\n";
    }
}

/* One qubit gates */ 
vector<sparse_elt> identity(){
    vector<sparse_elt> id;
    // dimensions
    id.push_back(createElt(2, 2, 0.0));
    // values
    id.push_back(createElt(0, 0, 1.0));
    id.push_back(createElt(1, 1, 1.0));
    return id;
}

vector<sparse_elt> naught() {
    vector<sparse_elt> naught;
    // dimensions
    naught.push_back(createElt(2, 2, 0.0));
    // values
    naught.push_back(createElt(1, 0, 1.0));
    naught.push_back(createElt(0, 1, 1.0));
    return naught;
}
vector<sparse_elt> hadamard() {
    vector<sparse_elt> had;
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


/* Expanding operations for 1 qubit gates. */

/* It is much more efficient to do many 1 qubit gates sequentially than to compute their tensor product */
vector<sparse_elt> oneQubitGateExpand(vector<sparse_elt> oneQubitGate, const int n, const int which) {
    vector<sparse_elt> total = (which == 0 ? oneQubitGate : identity());
    for(int i = 1; i < n; i++) {
        if(which == i) 
            total = tensor(total, oneQubitGate);
        else
            total = tensor(total, identity());
    }
    return total;
}


/* Two qubit gates */
vector<sparse_elt> CNOTExpanded(const int n, const int u, const int v) { 
    vector<sparse_elt> cnot;
    int N = pow(2, n);
    cnot.push_back(createElt(N, N, 0.0));
    int j;
    for(int i = 0; i < N; i++) {
        if (i>>(n-u-1) & 0x1) {
            j = i ^ (0x1<<(n-v-1));
            cnot.push_back(createElt(i, j, 1.0));
        } else {
            cnot.push_back(createElt(i, i, 1.0));
        }
    }
    return cnot;
}
vector<sparse_elt> CU1Expanded(vector<sparse_elt> U1, const int n, const int u, const int v) {
    vector<sparse_elt> CU1;
    int N = pow(2, n);
    CU1.push_back(createElt(N, N, 0.0));
    vector<complex<double> > stateOfV;
    for(long i = 0; i < N; i++) {
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



int main() {
    const int n = 2;
    const int N = pow(2, n);
    vector<complex<double> > state = classical(N, 0);
    
    /* Uniform superposition: */
    /*for (int i = 0; i < n; ++i) {
        state = applyOperator(state, oneQubitGateExpand(hadamard(), n, i));    
    }*/
    state = applyOperator(state, oneQubitGateExpand(naught(), n, 0));
    state = applyOperator(state, oneQubitGateExpand(naught(), n, 1));
    state = applyOperator(state, CU1Expanded(hadamard(), n, 0, 1));

    printVec(state, n, N, false);
    //printVecProbs(state, n, N, false);
}
