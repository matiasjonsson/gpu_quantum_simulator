







void Qvec::Qvec(uint32_t n) :
    n {n}

{
    this->N = pow(2,n);
    
}





void Qvec::printVec(bool printAll){
    cout << "State:\n";
    for (int i = 0; i < this->N; ++i) {
        if (!printAll && equalsZero(this->state.at(i)))
            continue;
        printComplex(this->state.at(i));
        cout << "|";
        for (int j = n-1; j >= 0; --j) {
            cout << (0x1 & (i>>j));
        }
        cout << "> + ";
        cout << "\n";
    }

}



