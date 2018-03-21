CC=gcc
CFlags=-Wall

all:
	$(CC) $(CFlags) autoencoder.c layer.c network.c optimizer.c dataset.c -o autoencoder

clean:
	rm *o
	rm autoencoder
