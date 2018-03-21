all:
	gcc autoencoder.c layer.c network.c optimizer.c dataset.c -o autoencoder

clean:
	rm *.o
	rm autoencoder
