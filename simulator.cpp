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
    vector<sparse_elt> had;
    // Dimensions
    had.push_back(createElt(2, 2, 0.0));
    // Values
    double temp = 1.0 / sqrt(2);
    had.push_back(createElt(0, 0, temp));
    had.push_back(createElt(0, 1, temp));
    had.push_back(createElt(1, 0, temp));
    had.push_back(createElt(1, 1, -temp));
    return had;
}
vector<sparse_elt> identity(){
    vector<sparse_elt> id;
    // Dimensions
    id.push_back(createElt(2, 2, 0.0));
    // Values
    id.push_back(createElt(0, 0, 1.0));
    id.push_back(createElt(1, 1, 1.0));
    return id;
}

vector<sparse_elt> hadamardAll(const int n) {
    vector<sparse_elt> had1 = hadamard();
    vector<sparse_elt> hadAll = had1;
    for(int i = 1; i < n; i++) {
        hadAll = tensor(had1, hadAll);
    }
    return hadAll;
}
vector<sparse_elt> hadamardOne(const int n, const int which) {
    vector<sparse_elt> had = hadamard();
    vector<sparse_elt> id = identity();
    vector<sparse_elt> total = (which == 0 ? had : id);
    for(int i = 1; i < n; i++) {
        if(which == i) 
            total = tensor(total, had);
        else
            total = tensor(total, id);
    }
    return total;
}

void printComplex(complex<double> amplitude) {
    cout << "(" << real(amplitude) << "+" << imag(amplitude) << "i)";
}
bool equalsZero(complex<double> amplitude) {
    return (norm(amplitude) < 1e-15);
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

int main() {
    const int n = 3;
    const int N = pow(2, n);
    vector<complex<double> > state = classical(N, 7);
    //Two ways of applying hadamard to all qubits:
    //Method 1: much faster, do each hadamard independently.
    for (int i = 0; i < n; ++i) {
        state = applyOperator(state, hadamardOne(n, i));    
    }

    //Method 2: much slower due to not being sparse, creates
    //one single dense matrix that hadamards all the qubits at once.
    //state = applyOperator(state, hadamardAll(n));
    
    //Only print if the number of qubits is small, else will be very long.
    printVec(state, n, N, false);
    //printVecProbs(state, n, N, false);
}
