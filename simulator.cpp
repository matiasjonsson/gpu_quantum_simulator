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
    int x;
    int y;
    complex<double> val;
};

float normSquared(complex<double> amplitude) {
    return real(amplitude * conj(amplitude));
}

bool isNormalized (vector<complex<double> > state) {
    double sum = 0;
    for (auto i = state.begin(); i != state.end(); ++i) {
        sum += normSquared(*i);
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
    assert(unitary.begin().x == state.size());
    vector<complex<double> > newstate = zero(state.size());
    for(auto i = unitary.begin() + 1; i != unitary.end(); ++i) {
        newstate.data()[i->x] += i->val * state.data()[i->y];
    }
    return newstate;
}

sparse_elt createElt(int x, int y, complex<double> val) {
    sparse_elt elt;
    elt.x = x;
    elt.y = y;
    elt.val = val;
    return elt;
}

vector<sparse_elt> tensor(vector<sparse_elt> u1, vector<sparse_elt> u2) {
    int dim1 = u1.begin()->x;
    int dim2 = u2.begin()->x;
    vector<sparse_elt> u3;
    u3.push_back(createElt(dim1*dim2, dim1*dim2, 0.0));
    for(auto i = u1.begin() + 1; i != u1.end(); ++i) {
        for(auto j = u2.begin() + 1; j != u2.end(); ++j) {
            u3.push_back(createElt(i->x * dim2 + j->x, i->y * dim2 + j->y, i->val * j->val));
        }
    }
    return u3;
}


vector<sparse_elt> hadamard() {
    vector<sparse_elt> hadamard;
    // Dimensions
    hadamard.push_back(createElt(2, 2, 0.0));
    // Values
    double temp = 1.0 / sqrt(2);
    hadamard.push_back(createElt(0, 0, temp));
    hadamard.push_back(createElt(0, 1, temp));
    hadamard.push_back(createElt(1, 0, temp));
    hadamard.push_back(createElt(1, 1, -temp));
    return hadamard;
}

vector<sparse_elt> hadamardAll(const int n) {
    vector<sparse_elt> had1 = hadamard();
    vector<sparse_elt> hadAll = had1;
    for(int i = 1; i < n; i++) {
        hadAll = tensor(had1, hadAll);
    }
    return hadAll;
}

void printComplex(complex<double> amplitude) {
    cout << "(" << real(amplitude) << "+" << imag(amplitude) << "i)";
}
void printVec(vector<complex<double> > state, const int n, const int N) {
    cout << "State:\n";
    for (int i = 0; i < N; ++i) {
        printComplex(state.at(i));
        cout << "|";
        for (int j = n-1; j >= 0; --j) {
            cout << (0x1 & (i>>j));
        }
        cout << "> + ";
        cout << "\n";
    }
}
void printVecProbs(vector<complex<double> > state, const int n, const int N) {
    cout << "Probability distribution:\n";
    for (int i = 0; i < N; ++i) {
        cout << "|<";
        for (int j = n-1; j >= 0; --j) {
            cout << (0x1 & (i>>j));
        }
        cout << "|Psi>|^2 = " << normSquared(state.at(i)) << "\n";
    }
}

int main() {
    const int n = 7;
    const int N = pow(2, n);
    vector<complex<double> > state = classical(N, 0);
    state = applyOperator(state, hadamardAll(n));
    printVec(state, n, N);
}
