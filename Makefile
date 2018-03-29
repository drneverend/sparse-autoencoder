CC=gcc
CFlags=-Wall

all:
	$(CC) $(CFlags) autoencoder.c layer.c network.c optimizer.c dataset.c linearalgebra.c util.c -o autoencoder

test:
	$(CC) $(CFlags) testlayer.c layer.c linearalgebra.c util.c -o testlayer

clean:
	rm autoencoder
