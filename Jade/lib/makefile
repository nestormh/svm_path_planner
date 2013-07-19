
all: sibtra_shm_shmInterface.so

sibtra_shm_shmInterface.so : sibtra_shm_shmInterface.c sibtra_shm_shmInterface.h shmCInterface.o
	gcc -shared -I"/usr/lib/jvm/java-1.5.0-gcj-4.3-1.5.0.0/include" -I"/usr/lib/jvm/java-1.5.0-gcj-4.3-1.5.0.0/include/linux" -o sibtra_shm_shmInterface.so sibtra_shm_shmInterface.c shmCInterface.o

 shmCInterface.o:  shmCInterface.c  shmCInterface.h
	-@echo 'Generando libreria nativa'
	gcc -c shmCInterface.c shmCInterface.h

clean:
	-@echo 'Eliminando libreria nativa shmInterface.so'
	-rm sibtra_shm_shmInterface.so