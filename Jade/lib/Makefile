CC=gcc

sibtra_lms_ManejaTelegramasJNI.so : sibtra_lms_ManejaTelegramasJNI.c Makefile
	$(CC) -Wall sibtra_lms_ManejaTelegramasJNI.c \
	-o sibtra_lms_ManejaTelegramasJNI.so \
	-I $(JAVA_HOME)/include/ -I $(JAVA_HOME)/include/linux/ \
	-I $(JAVA_HOME)/../include/ -I $(JAVA_HOME)/../include/linux/ \
	-shared  \
	-fPIC \
	-Wl,-soname,sibtra_lms_ManejaTelegramasJNI.so \

# No funciona en amd64 y no parece necesario
#	-static -lc \
	 
#Para generar cabecera
# javah -classpath ../bin -jni sibtra.lms.ManejaTelegramasJNI