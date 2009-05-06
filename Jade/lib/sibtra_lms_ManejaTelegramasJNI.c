
/* Si queremos mensajes de depuración, descomentar */
/*#define  INFO*/
/* Si queremos los mensaje de error, descomentar la siguiente */
#define ERR

#include "sibtra_lms_ManejaTelegramasJNI.h"


#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <termios.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <ctype.h>

#include <errno.h>
#include <time.h>

#ifdef ERR
#define ERROR(men)	{fprintf(stderr,men); fflush(stderr);}	
#else
#define ERROR(men)	{}
#endif

#define BAUDRATE B38400
#define MODEMDEVICE "/dev/ttyS0"

#define MAX_LEN 812  /*Maxima longitud de telegrama*/

#define TAMBUF (MAX_LEN+1)

/*10 sg*/
/*#define TimeOut (10*1000) */ 
#define TIME_OUT (200) /*200 ms*/

/*Descriptor de archivo de la serial*/
int fdSer;


/*Buffer */
unsigned char Buffer[MAX_LEN];

/*Esta inicializado*/
int Inicializado=0;

/*Guardarán las configuración del puerto*/
struct termios oldtio,newtio;

/*
Funcion para calcular el checksum

 */

unsigned short CalculaCRC(unsigned char *PtBuff, unsigned int Len) {

#define CRC16_GEN_POL 0x8005

  unsigned short uCRC16;
  int i;

  uCRC16=PtBuff[0];
  i=0;
  while(i<Len-1) {
    if(uCRC16 & 0x8000) {
      uCRC16=(uCRC16 & 0x7fff)<<1;
      uCRC16^=CRC16_GEN_POL;
    } else 
      uCRC16<<=1;

    uCRC16 ^= (unsigned short)PtBuff[i+1] | (((unsigned short)PtBuff[i])<<8) ;
    i++;
  }

  return uCRC16;
}


/* Idem que read pero con time out indicado en miliTO
 */
ssize_t LeeTimeOut(int fd, void *bufer, size_t nbytes, unsigned long miliTO) {

  if(miliTO>0) {
	  fd_set readfs;
	  int r;
	  struct timeval tv;
	  
	  FD_ZERO(&readfs);
	  FD_SET(fd,&readfs);
	
	  tv.tv_sec=miliTO/1000; /*nos quedamos con los segundos*/
	  tv.tv_usec=(miliTO%1000)*1000; /*nos quedamos con los mili.sg y pasamos a micro.sg*/
	
	  do {
	    r=select(fd+1,&readfs,NULL,NULL,&tv);
	  }  while(r==-1 && errno == EINTR);
	  if(r!=1)
	    return 0; /*ha habido timeout o otro error*/
  }
  /*Debe haber algo disponible o se ha pasado time out 0*/
  return read(fd,bufer,nbytes);
}

/*
 * Class:     sibtra_ManejaTelegramasJNI
 * Method:    ConectaPuerto
 * Signature: (Ljava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL Java_sibtra_lms_ManejaTelegramasJNI_ConectaPuerto
(JNIEnv *env, jobject obj, jstring jsNombrePuerto) {

   Inicializado=JNI_FALSE;

   /*Tenemos que obtener el nombre del puerto del String pasado*/
   const char *NombrePuerto = (*env)->GetStringUTFChars(env, jsNombrePuerto, 0);

   fdSer = open(NombrePuerto, O_RDWR | O_NOCTTY );
   if (fdSer <0) {  
     perror(NombrePuerto); 
     return JNI_FALSE;  
     
   }
   /*liberamos ya que no la necesitamos más*/
   (*env)->ReleaseStringUTFChars(env, jsNombrePuerto, NombrePuerto);


   tcgetattr(fdSer,&oldtio); /* salva configuracion actual del puerto  */

   bzero((void *)&newtio, sizeof(newtio));/* limpiamos struct para recibir los
                                        nuevos parametros del puerto */

/*
    BAUDRATE: Fija la tasa bps. Podria tambien usar cfsetispeed y cfsetospeed.
    CRTSCTS : control de flujo de salida por hardware (usado solo si el cable 
    tiene todas las lineas necesarias Vea sect. 7 de Serial-HOWTO)
    CS8     : 8n1 (8bit,no paridad,1 bit de parada)
    CLOCAL  : conexion local, sin control de modem
    CREAD   : activa recepcion de caracteres
*/


   newtio.c_cflag = BAUDRATE | CS8 | CLOCAL | CREAD;

/*
   IGNPAR  : ignora los bytes con error de paridad
   ICRNL   : mapea CR a NL (en otro caso una entrada CR del otro ordenador 
   no terminaria la entrada) en otro caso hace un dispositivo en bruto 
   (sin otro proceso de entrada)
*/
   newtio.c_iflag = IGNPAR;

   newtio.c_oflag = 0;

/* pone el modo entrada (no-canonico, sin eco,...) */

   newtio.c_lflag = 0;

   newtio.c_cc[VTIME]    = 0;   /* temporizador entre caracter, máximo 100 ms */
   newtio.c_cc[VMIN]     = 1;   /* bloquea lectura hasta recibir 1 chars  */

   tcflush(fdSer, TCIFLUSH);
   tcsetattr(fdSer,TCSANOW,&newtio);

   Inicializado=JNI_TRUE;
   return JNI_TRUE;
}

/*
 * Class:     sibtra_lms_ManejaTelegramasJNI
 * Method:    setBaudrate
 * Signature: (I)Z
 * Tenemos que comprobar que la velocidad es valida y tratar de hacer el cambio.
 * Debe estar inicializado
 */
JNIEXPORT jboolean JNICALL Java_sibtra_lms_ManejaTelegramasJNI_setBaudrate
  (JNIEnv *env, jobject obj, jint bautrate) {

  speed_t br;
  /*Si no esta inicializado, salimos */
  if(Inicializado!=JNI_TRUE)
  	return JNI_FALSE;
  switch (bautrate) {
    case 0: br=B0; break;
    case 50: br=B50; break;
    case 75: br=B75; break;
    case 110: br=B110; break;
    case 134: br=B134; break;
    case 150: br=B150; break;
    case 200: br=B200; break;
    case 300: br=B300; break;
    case 600: br=B600; break;
    case 1200: br=B1200; break;
    case 1800: br=B1800; break;
    case 2400: br=B2400; break;
    case 4800: br=B4800; break;
    case 9600: br=B9600; break;
    case 19200: br=B19200; break;
    case 38400: br=B38400; break;
    case 57600: br=B57600; break;
    case 115200: br=B115200; break;
    case 230400: br=B230400; break;
    default: 
     {      /*No es velocidad válida*/
        jclass newExcCls = (*env)->FindClass(env, 
                      "java/lang/IllegalArgumentException");
        if (newExcCls == NULL) {
            /* Unable to find the exception class, give up. */
             return JNI_FALSE;
         }
         (*env)->ThrowNew(env, newExcCls, "Baudrate inválido");
      }
      return JNI_FALSE;
  	
    }
  cfsetispeed(&newtio,br);
  cfsetospeed(&newtio,br);

  /* no parce necesario 
	 tcflush(fdSer, TCIFLUSH);
  */
  if(tcsetattr(fdSer,TCSANOW,&newtio)<0) {
	   ERROR("\n\t\t\tJNI: Al establecer la nueva velocidad");
	   fflush(stderr);
       return JNI_FALSE;
   }
  return JNI_TRUE;
}


/*
 * Class:     sibtra_lms_ManejaTelegramasJNI
 * Method:    LeeMensaje
 * Signature: (I)[B
 */
JNIEXPORT jbyteArray JNICALL Java_sibtra_lms_ManejaTelegramasJNI_LeeMensaje
  (JNIEnv *env, jobject obj, jint milisTOut) {

  unsigned char *buf;
  int res;
  int len;  /*valor campo longitud*/
  jbyteArray jarray;

  if(!Inicializado)
    return JNI_FALSE;

  	buf=Buffer;
	buf[0]=0x06;

	//Buffer para quitar los posibles confirmaciones iniciales
  while((buf[0]==0x15) || (buf[0]==0x06) ) {
		res=LeeTimeOut(fdSer,buf,1,milisTOut);   
	  	if(res!=1) {
#ifdef ERR
	    fprintf(stderr,"\n\t\t\tJNI: Ha habido timeout (%d) esperando inicio mensaje: %d",milisTOut,res);
	    fflush(stderr);
#endif
	    return JNI_FALSE;
	  }
	
	  if(buf[0]==0x15) {
	    ERROR("\n\t\t\tJNI: Es NO RECONOCIMIENTO (0x15)");
	    continue;
	  }
	  if(buf[0]==0x06) {
	    ERROR("\n\t\t\tJNI: Es reconocimiento (0x06)");
	    continue;
	  }
  }
  if(buf[0]!=0x02) {
    ERROR("\n\t\t\tJNI: NO es comienzo de telegrama: ALGO RARO");
    return JNI_FALSE;
  }
#ifdef INFO
  fprintf(stdout,"\n\t\t\tJNI:  STX: %02hhX;",buf[0]);
  fflush(stdout);
#endif

  /*Tenemos que conseguir todo el mensaje*/
  /*Primero hasta saber el tamaño*/
  while(res<4) {
    int bl;
    if(!(bl=LeeTimeOut(fdSer,buf+res,4-res,milisTOut))) {
#ifdef ERR
    fprintf(stderr,"\n\t\t\tJNI: Ha habido timeout (%d) esperando tamaño",milisTOut);
    fflush(stderr);
#endif
      return 0;
    }
    res+=bl;
  }         
#ifdef INFO
  fprintf(stdout,"\t\t\tJNI:  Addr: %02hhX;",buf[1]);
  fflush(stdout);
#endif

  len=(unsigned short)(buf[2] | (buf[3]<<8));

#ifdef INFO
  fprintf(stdout,"\t\t\tJNI:  Len: %02hhX%02hhX (%d)",buf[3],buf[2],len);
  fflush(stdout);
#endif

  /*Conseguimos el resto del mensaje*/
  while(res<(len+6)) {
    int bl;
    if(!(bl=LeeTimeOut(fdSer,buf+res,(len+6)-res,milisTOut))) {
#ifdef ERR
      fprintf(stderr,"\n\t\t\tJNI: Ha habido timeout (%d) esperando resto del mensaje: %d de %d"
        ,milisTOut,res, len+6);
      fflush(stderr);
#endif
      return 0;
    }
    res+=bl;
  }         

#ifdef INFO
  { int i;
   fprintf(stdout,"\n\t\t\tJNI: Telegrama completo:"); 

   /*Volcamos en hexadecimal como mucho 20 bytes */
   for (i=0; i<res && i<20; i++)
     fprintf(stdout," %02hhX",buf[i]);
/*     fprintf(stdout," => ");  */

/*   /\*Volcamos caracter *\/ */
/*   for (i=0; i<res; i++) */
/*     fprintf(stdout,"%c",buf[i]);  */


/*   fprintf(stdout,"\n Mensaje: "); */
/*   /\*Volcamos en hexadecimal*\/ */
/*   for (i=4; i<(res-3); i++) */
/*     fprintf(stdout," %02hhX",buf[i]); */

/*   fprintf(stdout," => "); */
/*   /\*Volcamos caracter*\/ */
/*   for (i=4; i<(res-3); i++) */
/*     fprintf(stdout,"%c",buf[i]); */


  fprintf(stdout,"\n Comando: %02hhX; Status: %02hhX;",buf[4],buf[res-3]);

  fprintf(stdout,"\nCHKS: %02hhX%02hhX;",buf[res-1],buf[res-2]);

  fprintf(stdout," Calculado: %04hX",CalculaCRC(buf,res-2));
  fflush(stdout);
  }
#endif


  /* Creamos el byte[] para depositar mensaje */
  jarray = (*env)->NewByteArray(env,len);
  (*env)->SetByteArrayRegion(env,jarray, 0, len, buf+4);

  return jarray;


}

/*
 * Class:     sibtra_lms_ManejaTelegramasJNI
 * Method:    EnviaMensajeSinConfirmacion
 * Signature: ([B)Z
 */
JNIEXPORT jboolean JNICALL Java_sibtra_lms_ManejaTelegramasJNI_EnviaMensajeSinConfirmacion
(JNIEnv *env, jobject obj, jbyteArray jBA) {

  unsigned char *buf=Buffer, *ptjBA;
  unsigned short CRC;
  int res;
  jsize TamMen;

  if(!Inicializado)
    return JNI_FALSE;

  /*Copiamos el array pasado*/
  TamMen=(*env)->GetArrayLength(env, jBA);
  
  ptjBA=(*env)->GetByteArrayElements(env,jBA,0);

  memcpy((void*)(buf+4), (const void*)ptjBA,(size_t)(sizeof(unsigned char)*TamMen));

  /*liberamos sin copiar ya que no ha habido cambios*/
  (*env)->ReleaseByteArrayElements(env,jBA,ptjBA,JNI_ABORT);

  /*Completamos telegrama*/
  buf[0]=2;
  buf[1]=0;
  buf[2]=TamMen%256;
  buf[3]=TamMen/256;

  CRC=CalculaCRC(buf,TamMen+4);
  buf[4+TamMen]=CRC%256;
  buf[4+TamMen+1]=CRC/256;

#ifdef INFO
  { 
  	int i;
    fprintf(stdout,"\n\t\t\tJNI: Telegrama Completo:");
    /*Volcamos en hexadecimal*/
    for (i=0; i<TamMen+6; i++)
      fprintf(stdout," %02hhX",buf[i]);

    fprintf(stdout,"\n\t\t\tJNI: Procedemos a enviarlo:");
    fflush(stdout);
  }
#endif

  /*Antes de enviar purgamos todo lo que hay a la entrada
   para ello aprovechamos el buffer no usado*/
//  while(LeeTimeOut(fdSer,buf+TamMen+7,TAMBUF-TamMen-7,1)>0);

  res=write(fdSer,buf,TamMen+6);
  if(res<TamMen+6) {
#ifdef ERR
    fprintf(stderr,"\n\t\t\tJNI: No se envió todo el mensaje, solo %d caracteres",res);
    fflush(stderr);
#endif
    return JNI_FALSE;
  }
  
#ifdef INFO
    fprintf(stdout,"\n\t\t\tJNI: Enviado Correctamente");
    fflush(stdout);
#endif
  return JNI_TRUE;
}
/*
 * Class:     sibtra_lms_ManejaTelegramasJNI
 * Method:    esperaConfirmacion
 * Signature: (I)Z
 */
JNIEXPORT jboolean JNICALL Java_sibtra_lms_ManejaTelegramasJNI_esperaConfirmacion
  (JNIEnv *nev, jobject obj, jint milisTOut) {
  int res;
  unsigned char conf;

#ifdef INFO
    fprintf(stdout,"\n\t\t\tJNI:  Esperamos confirmación de mensaje");
    fflush(stdout);
#endif
     
  if(!(res = LeeTimeOut(fdSer,&conf,1,milisTOut))) {
#ifdef ERR
    fprintf(stderr,"\n\t\t\tJNI: Ha habido timeout (%d) esperando confirmación",milisTOut);
    fflush(stderr);
#endif
    return JNI_FALSE;
  }

  if(conf==0x15) {
    ERROR("\n\t\t\tJNI: Telgrama NO ENTENDIDO");
    return JNI_FALSE;
  } else if(conf==0x06) {
#ifdef INFO
    fprintf(stdout,"\n\t\t\tJNI:  SI se confirmó el mensaje");
    fflush(stdout);
#endif
    return JNI_TRUE;
  }
  ERROR("\n\t\t\tJNI: Situación rara, lo recibido no es ni confirmación ni fallo");
  return JNI_FALSE;

}

/*
 * Class:     sibtra_ManejaTelegramasJNI
 * Method:    cierraPuerto
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_sibtra_lms_ManejaTelegramasJNI_cierraPuerto
(JNIEnv *env, jobject obj) {

  if(!Inicializado)
    return JNI_FALSE;
  close(fdSer);
  Inicializado=JNI_FALSE;
  return JNI_FALSE;
}


/*
 * Class:     sibtra_lms_ManejaTelegramasJNI
 * Method:    purgaBufferEntrada
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_sibtra_lms_ManejaTelegramasJNI_purgaBufferEntrada
  (JNIEnv *env, jobject obj) {

//  while(LeeTimeOut(fdSer,buf+TamMen+7,TAMBUF-TamMen-7,1)>0);
  /* no parce necesario 
	 tcflush(fdSer, TCIFLUSH);
  */
//	 tcflush(fdSer, TCIFLUSH);
   while(LeeTimeOut(fdSer,Buffer,TAMBUF,1)>0);

  	
}



/*
 * Class:     sibtra_lms_ManejaTelegramasJNI
 * Method:    isInicializado
 * Signature: ()Z
 * Sencillamente devolvemos Inicializado
 */
JNIEXPORT jboolean JNICALL Java_sibtra_lms_ManejaTelegramasJNI_isInicializado
  (JNIEnv *env, jobject obj) {
  	return Inicializado;
}



