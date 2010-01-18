CC=gcc
CFLAGS=-I.
DEPS = OSIFdll.h OSIFflash.h OSIFlib.h
OBJ = OSIFdll.o OSIFflash.o OSIFhex.o OSIFio.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) -fPIC

libOSIFlib: $(OBJ)
	gcc -shared -Wl,-soname,libOSIFlib.so.1 -o libOSIFlib.so.1.0.1 $^ $(CFLAGS) -lusb

testapp_Directo : testapp_Directo.c
	gcc testapp_Directo.c -L. -lOSIFlib -o testapp_Directo


clean:
	rm *.o
	rm libOSIFlib.so.1.0.1
	rm testapp_Directo
