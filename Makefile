CC=gcc
CFlags=-Wall

all:
	$(CC) $(CFlags) autoencoder.c layer.c network.c optimizer.c dataset.c linearalgebra.c -o autoencoder

clean:
	rm autoencoder
