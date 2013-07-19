
CC=gcc
AUTOGENERADOS=shmInterface.h shmInterface.c sibtra_shm_ShmInterface.c ../../src/sibtra/shm/ShmInterface.java
JAVA_HOME=/usr/lib/jvm/java-6-sun

sibtra_shm_ShmInterface.so : sibtra_shm_ShmInterface.c sibtra_shm_ShmInterface.h shmInterface.o Makefile
	$(CC) -Wall sibtra_shm_ShmInterface.c shmInterface.o \
	-o sibtra_shm_ShmInterface.so \
	-I $(JAVA_HOME)/include/ -I $(JAVA_HOME)/include/linux/ \
        -I $(JAVA_HOME)/../include/ -I $(JAVA_HOME)/../include/linux/ \
	-shared \
	-fPIC \
	-Wl,-soname,sibtra_shm_ShmInterface.so \

shmInterface.o:  shmInterface.c  shmInterface.h
	-@echo 'Generando libreria nativa'
	gcc -c shmInterface.c -o shmInterface.o -fPIC  

../../bin/sibtra/shm/ShmInterface.class : ../../src/sibtra/shm/ShmInterface.java
	$(JAVA_HOME)/bin/javac -d ../../bin/ ../../src/sibtra/shm/ShmInterface.java

sibtra_shm_ShmInterface.h : ../../bin/sibtra/shm/ShmInterface.class
	$(JAVA_HOME)/bin/javah -classpath .:../../bin/ sibtra.shm.ShmInterface

$(AUTOGENERADOS) : Variables.lst GeneraShm.sh
	./GeneraShm.sh


clean:
	-@echo 'Eliminando libreria nativa shmInterface.so'
	-rm sibtra_shm_ShmInterface.so
	-rm sibtra_shm_ShmInterface.h
	-rm shmInterface.o
	-rm $(AUTOGENERADOS)
