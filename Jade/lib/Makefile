

sibtra_lms_ManejaTelegramasJNI.so : sibtra_lms_ManejaTelegramasJNI.c Makefile
	gcc sibtra_lms_ManejaTelegramasJNI.c \
	-o sibtra_lms_ManejaTelegramasJNI.so \
	-shared -Wl,-soname,libnative.so -static -lc \
	-I $(JAVA_HOME)/include/ -I $(JAVA_HOME)/include/linux/ \
	-I $(JAVA_HOME)/../include/ -I $(JAVA_HOME)/../include/linux/