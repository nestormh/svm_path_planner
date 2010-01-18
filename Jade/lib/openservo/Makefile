CC=gcc

openservo_OsifJNI.so : openservo_OsifJNI.c Makefile
	$(CC) -Wall openservo_OsifJNI.c \
	-o openservo_OsifJNI.so \
	-I $(JAVA_HOME)/include/ -I $(JAVA_HOME)/include/linux/ \
	-I $(JAVA_HOME)/../include/ -I $(JAVA_HOME)/../include/linux/ \
	-shared  \
	-Ldll \
	-Idll \
	 -lOSIFlib \
	-fPIC \
	-Wl,-soname,openservo_OsifJNI.so \

# No funciona en amd64 y no parece necesario
#	-static -lc \
	 
#Para generar cabecera
# javah -classpath ../../bin -jni openservo.OsifJNI
