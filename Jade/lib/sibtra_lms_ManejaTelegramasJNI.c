
/* Si queremos mensajes de depuración, descomentar
#define  DEPURA */

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

#define BAUDRATE B38400
#define MODEMDEVICE "/dev/ttyS0"

#define MAX_LEN 812  /*Maxima longitud de telegrama*/

#define TAMBUF (MAX_LEN+1)

#define TimeOut (10*1000) /*10 sg*/
/*#define TimeOut (200) /*10 sg*/

/*Descriptor de archivo de la serial*/
int fdSer;


/*Buffer */
unsigned char Buffer[MAX_LEN];

/*Esta inicializado*/
int Inicializado=0;

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
ssize_t LeeTimeOut(int fd, void *buf, size_t nbytes, unsigned long miliTO) {

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

  /*Debe haber algo disponible*/
  return read(fd,buf,nbytes);
}

/*
 * Class:     sibtra_ManejaTelegramasJNI
 * Method:    ConectaPuerto
 * Signature: (Ljava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL Java_sibtra_lms_ManejaTelegramasJNI_ConectaPuerto
(JNIEnv *env, jobject obj, jstring jsNombrePuerto) {
   struct termios oldtio,newtio;

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
 * Class:     sibtra_ManejaTelegramasJNI
 * Method:    LeeMensaje
 * Signature: ()[B
 */
JNIEXPORT jbyteArray JNICALL Java_sibtra_lms_ManejaTelegramasJNI_LeeMensaje
(JNIEnv *env, jobject obj) {

  unsigned char *buf=Buffer;
  int res,i;
  int len;  /*valor campo longitud*/
  jbyteArray jarray;

  if(!Inicializado)
    return 0;
     
  if(!(res=LeeTimeOut(fdSer,buf,1,TimeOut))) {
    printf("\nHa habido timeout esperando inicio mensaje");
    return 0;
  }

  if(buf[0]==0x06) {
    printf("\nEs reconocimiento");
    return 0;
  }
  if(buf[0]==0x15) {
    printf("\nNO RECONOCIMIENTO");
    return 0;
  }

  if(buf[0]!=0x02) {
    printf("\nNO es comeinzo de telegrama: ALGO RARO");
    return 0;
  }

#ifdef DEPURA
  printf("\n STX: %02hhX;",buf[0]);
#endif

  /*Tenemos que conseguir todo el mensaje*/
  /*Primero hasta saber el tamaño*/
  while(res<4) {
    int bl;
    if(!(bl=LeeTimeOut(fdSer,buf+res,4-res,TimeOut))) {
      printf("\nHa habido timeout esperando tamaño");
      return 0;
    }
    res+=bl;
  }         
#ifdef DEPURA
  printf(" Addr: %02hhX;",buf[1]);
#endif

  len=(unsigned short)(buf[2] | (buf[3]<<8));

#ifdef DEPURA
  printf(" Len: %02hhX%02hhX (%d)",buf[3],buf[2],len);
#endif

  /*Conseguimos el resto del mensaje*/
  while(res<(len+6)) {
    int bl;
    if(!(bl=LeeTimeOut(fdSer,buf+res,(len+6)-res,TimeOut))) {
      printf("\nHa habido timeout esperando resto del mensaje: %d de %d",res, len+6);
      return 0;
    }
    res+=bl;
  }         

#ifdef DEPURA
   printf("\nTelegrama completo:\n"); 

   /*Volcamos en hexadecimal como mucho 20 bytes */
   for (i=0; i<res && i<20; i++)
     printf(" %02hhX",buf[i]);
/*     printf(" => ");  */

/*   /\*Volcamos caracter *\/ */
/*   for (i=0; i<res; i++) */
/*     printf("%c",buf[i]);  */


/*   printf("\n Mensaje: "); */
/*   /\*Volcamos en hexadecimal*\/ */
/*   for (i=4; i<(res-3); i++) */
/*     printf(" %02hhX",buf[i]); */

/*   printf(" => "); */
/*   /\*Volcamos caracter*\/ */
/*   for (i=4; i<(res-3); i++) */
/*     printf("%c",buf[i]); */


  printf("\n Comando: %02hhX; Status: %02hhX;",buf[4],buf[res-3]);

  printf("\nCHKS: %02hhX%02hhX;",buf[res-1],buf[res-2]);

  printf(" Calculado: %04hX",CalculaCRC(buf,res-2));
#endif


  /* Creamos el byte[] para depositar mensaje */
  jarray = (*env)->NewByteArray(env,len);
  (*env)->SetByteArrayRegion(env,jarray, 0, len, buf+4);

  return jarray;


}

/*
 * Class:     sibtra_ManejaTelegramasJNI
 * Method:    EnviaMensaje
 * Signature: ([B)Z
 */
JNIEXPORT jboolean JNICALL Java_sibtra_lms_ManejaTelegramasJNI_EnviaMensaje
(JNIEnv *env, jobject obj, jbyteArray jBA) {

  unsigned char *buf=Buffer, *ptjBA;
  unsigned short CRC;
  int res,i;
  unsigned char conf;
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

#ifdef DEPURA
  printf("\nTelegrama Completo:");
  /*Volcamos en hexadecimal*/
  for (i=0; i<TamMen+6; i++)
    printf(" %02hhX",buf[i]);

  printf("\nProcedemos a enviarlo:");
#endif

  /*Antes de enviar purgamos todo lo que hay a la entrada
   para ello aprovechamos el buffer no usado*/
  while(LeeTimeOut(fdSer,buf+TamMen+7,TAMBUF-TamMen-7,1)>0);

  res=write(fdSer,buf,TamMen+6);
  if(res<TamMen+6) {
    printf("\nNo se envió todo el mensaje, solo %d caracteres\n",res);
    return JNI_FALSE;
  } else

#ifdef DEBUG
    printf("\nEnviado Correctamente");
#endif

     
  if(!(res = LeeTimeOut(fdSer,&conf,1,TimeOut))) {
    printf("\nHa habido timeout esperando confirmación");
    return JNI_FALSE;
  }

  if(conf==0x15) {
    printf("\nTelgrama NO ENTENDIDO");
    return JNI_FALSE;
  } else if(conf==0x06) {
#ifdef DEBUG
    printf("\n SI se confirmó el mensaje");
#endif
    return JNI_TRUE;
  }
  printf("\nSituación rara");
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

