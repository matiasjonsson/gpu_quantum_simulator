#include <iostream>
#include <cmath>
#include <vector>
#include <complex>
#include <assert.h>
#include <bitset>
using namespace std;

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
vector<complex<double> > classical(const int N, const int init) {
    vector<complex<double> > state;
    state.assign(N, 0);
    state.data()[init] = 1.0;
    return state;
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
    const int n = 3;
    const int N = pow(2, n);
    vector<complex<double> > state = uniform(N);
    printVecProbs(state, n, N);
    return 0;
}
