# Honors-Thesis
My Honors Thesis project that uses a neural network to recommend songs based on sound similarity

# How to Use
Download the repo and install the required python3 libraries. 
Required libraries: pylab, numpy, pydub, keras  
Then, run `python3 scriptname.py e` where e is the number of epochs to test on and scriptname is one of the scripts you'd like to used depending on how you'd like to encode the data.  
`ae-normal.py` does a basic autoencoder.  
`ae-sparse.py` adds a sparsity constraint.  
`ae-deep.py` adds more layers to the basic autoencoder.  
