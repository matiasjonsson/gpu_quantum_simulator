













class Qvec{

	
	
    private:
	uint32_t n;
	unsigned long N;

    public:
	vector<complex<double> > state;
	void printVec();
	void printProbs();
        long getMostLikely();

	Qvec(uint32_t num_qubits);


}
