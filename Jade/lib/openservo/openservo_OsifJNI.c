/* Implementacion de la JNI */

#include "openservo_OsifJNI.h"
#include <string.h>
#include <stdio.h>
#include <OSIFdll.h>

/*
 * Class:     openservo_OsifJNI
 * Method:    OSIF_init
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_openservo_OsifJNI_OSIF_1init
  (JNIEnv *env, jclass clase) {
   return OSIF_init();
}

/*
 * Class:     openservo_OsifJNI
 * Method:    OSIF_deinit
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_openservo_OsifJNI_OSIF_1deinit
  (JNIEnv *env, jclass clase) {
   return OSIF_deinit();
}

/*
 * Class:     openservo_OsifJNI
 * Method:    OSIF_get_libversion
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_openservo_OsifJNI_OSIF_1get_1libversion
  (JNIEnv *env, jclass clase) {
	unsigned char data[8];
	return OSIF_get_libversion(data);
}

/*
 * Class:     openservo_OsifJNI
 * Method:    OSIF_write_data
 * Signature: (IIB[BIZ)I
 */
JNIEXPORT jint JNICALL Java_openservo_OsifJNI_OSIF_1write_1data
  (JNIEnv *env, jclass clase, jint adapter, jint i2c_addr, jbyte addr, jbyteArray jdata, jint buflen, jboolean issue_stop) {
	jsize TamMen=(*env)->GetArrayLength(env, jdata);
	if (TamMen<buflen) return -1; //array demasiado pequeño

	//Obtenemos array C de jarray de bytes
	jbyte* data=(*env)->GetByteArrayElements(env,jdata,NULL);

	int resultado=OSIF_write_data(adapter,i2c_addr, addr, (unsigned char*)data, buflen, issue_stop );
	//liberamos el array sin copiarlo
	(*env)->ReleaseByteArrayElements(env,jdata,data,JNI_ABORT);
	return resultado;
}

/*
 * Class:     openservo_OsifJNI
 * Method:    OSIF_writeonly
 * Signature: (II[BIZ)I
 */
JNIEXPORT jint JNICALL Java_openservo_OsifJNI_OSIF_1writeonly
  (JNIEnv *env, jclass clase, jint adapter, jint i2c_addr, jbyteArray jdata, jint buflen, jboolean issue_stop){
	jsize TamMen=(*env)->GetArrayLength(env, jdata);
	if (TamMen<buflen) return -1; //array demasiado pequeño

	//Obtenemos array C de jarray de bytes
	jbyte* data=(*env)->GetByteArrayElements(env,jdata,NULL);

	int resultado=OSIF_writeonly(adapter, i2c_addr, (unsigned char *) data, buflen, issue_stop );
	//liberamos el array sin copiarlo
	(*env)->ReleaseByteArrayElements(env,jdata,data,JNI_ABORT);
	return resultado;

}

/*
 * Class:     openservo_OsifJNI
 * Method:    OSIF_read_data
 * Signature: (IIB[BIZ)I
 */
JNIEXPORT jint JNICALL Java_openservo_OsifJNI_OSIF_1read_1data
  (JNIEnv *env, jclass clase, jint adapter, jint i2c_addr, jbyte addr, jbyteArray jdata, jint buflen, jboolean issue_stop) {
	jsize TamMen=(*env)->GetArrayLength(env, jdata);
	if (TamMen<buflen) return -1; //array demasiado pequeño

	//Obtenemos array C de jarray de bytes
	jbyte* data=(*env)->GetByteArrayElements(env,jdata,NULL);

	int resultado=OSIF_read_data(adapter, i2c_addr, addr, (unsigned char *) data, buflen, issue_stop );
	//liberamos el array COPIANDO
	(*env)->ReleaseByteArrayElements(env,jdata,data,0);
	return resultado;

}

/*
 * Class:     openservo_OsifJNI
 * Method:    OSIF_readonly
 * Signature: (II[BIZ)I
 */
JNIEXPORT jint JNICALL Java_openservo_OsifJNI_OSIF_1readonly
(JNIEnv *env, jclass clase, jint adapter, jint i2c_addr, jbyteArray jdata, jint buflen, jboolean issue_stop) {
	jsize TamMen=(*env)->GetArrayLength(env, jdata);
	if (TamMen<buflen) return -1; //array demasiado pequeño

	//Obtenemos array C de jarray de bytes
	jbyte* data=(*env)->GetByteArrayElements(env,jdata,NULL);

	int resultado=OSIF_readonly(adapter, i2c_addr, (unsigned char *) data, buflen, issue_stop );
	//liberamos el array COPIANDO
	(*env)->ReleaseByteArrayElements(env,jdata,data,0);
	return resultado;

}

/*
 * Class:     openservo_OsifJNI
 * Method:    OSIF_scan
 * Signature: (I)[I
 */
JNIEXPORT jintArray JNICALL Java_openservo_OsifJNI_OSIF_1scan
  (JNIEnv *env, jclass clase, jint adapter) {
	int dev_count;
	int devices[MAX_I2C_DEVICES];
	int resultado=OSIF_scan(adapter, devices, &dev_count );

	if(resultado<0)
		return NULL;
	jintArray jdevices=(*env)->NewIntArray(env,dev_count);
	(*env)->SetIntArrayRegion(env,jdevices, 0, dev_count, devices);

	return jdevices;

}

/*
 * Class:     openservo_OsifJNI
 * Method:    OSIF_probe
 * Signature: (II)Z
 */
JNIEXPORT jboolean JNICALL Java_openservo_OsifJNI_OSIF_1probe
  (JNIEnv *env, jclass clase, jint adapter, jint i2c_addr) {
	return OSIF_probe(adapter, i2c_addr );
}

/*
 * Class:     openservo_OsifJNI
 * Method:    OSIF_command
 * Signature: (IIB)I
 */
JNIEXPORT jint JNICALL Java_openservo_OsifJNI_OSIF_1command
  (JNIEnv *env, jclass clase, jint adapter, jint i2c_addr, jbyte command) {
	return OSIF_command(adapter, i2c_addr, command);
}

/*
 * Class:     openservo_OsifJNI
 * Method:    OSIF_get_adapter_count
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_openservo_OsifJNI_OSIF_1get_1adapter_1count
  (JNIEnv *env, jclass clase) {
	return OSIF_get_adapter_count();
}

/*
 * Class:     openservo_OsifJNI
 * Method:    OSIF_get_adapter_name
 * Signature: (I)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_openservo_OsifJNI_OSIF_1get_1adapter_1name
  (JNIEnv *env, jclass clase, jint adapter) {
  	char name[80];
  	OSIF_get_adapter_name(adapter, name);
	return (*env)->NewStringUTF(env, name);
}

/*
 * Class:     openservo_OsifJNI
 * Method:    OSIF_io_set_ddr
 * Signature: (III)I
 */
JNIEXPORT jint JNICALL Java_openservo_OsifJNI_OSIF_1io_1set_1ddr
  (JNIEnv *env, jclass clase, jint adapter_no, jint ddr, jint enabled) {
  	return OSIF_io_set_ddr(adapter_no, ddr, enabled);
}

/*
 * Class:     openservo_OsifJNI
 * Method:    OSIF_io_set_out
 * Signature: (II)I
 */
JNIEXPORT jint JNICALL Java_openservo_OsifJNI_OSIF_1io_1set_1out
  (JNIEnv *env, jclass clase, jint adapter_no, jint io) {
  	return OSIF_io_set_out(adapter_no, io);
}

/*
 * Class:     openservo_OsifJNI
 * Method:    OSIF_io_get_in
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_openservo_OsifJNI_OSIF_1io_1get_1in
  (JNIEnv *env, jclass clase, jint adapter_no) {
  	return OSIF_io_get_in(adapter_no);
}

/*
 * Class:     openservo_OsifJNI
 * Method:    OSIF_io_set_out1
 * Signature: (III)I
 */
JNIEXPORT jint JNICALL Java_openservo_OsifJNI_OSIF_1io_1set_1out1
  (JNIEnv *env, jclass clase, jint adapter_no, jint gpio, jint state) {
	return   	OSIF_io_set_out1( adapter_no, gpio, state);
}

/*
 * Class:     openservo_OsifJNI
 * Method:    OSIF_io_get_current
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_openservo_OsifJNI_OSIF_1io_1get_1current
  (JNIEnv *env, jclass clase, jint adapter_no) {
	return OSIF_io_get_current(adapter_no);
}

/*
 * Class:     openservo_OsifJNI
 * Method:    OSIF_disable_i2c
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_openservo_OsifJNI_OSIF_1disable_1i2c
  (JNIEnv *env, jclass clase, jint adapter_no){
	return OSIF_disable_i2c(adapter_no);
}

/*
 * Class:     openservo_OsifJNI
 * Method:    OSIF_enable_i2c
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_openservo_OsifJNI_OSIF_1enable_1i2c
  (JNIEnv *env, jclass clase, jint adapter_no) {
	return OSIF_enable_i2c(adapter_no);
}

/*
 * Class:     openservo_OsifJNI
 * Method:    OSIF_set_bitrate
 * Signature: (II)I
 */
JNIEXPORT jint JNICALL Java_openservo_OsifJNI_OSIF_1set_1bitrate
  (JNIEnv *env, jclass clase, jint adapter_no, jint bitrate_hz) {
	return OSIF_set_bitrate(adapter_no, bitrate_hz);
}

/*
 * Class:     openservo_OsifJNI
 * Method:    OSIF_set_twbr
 * Signature: (III)I
 */
JNIEXPORT jint JNICALL Java_openservo_OsifJNI_OSIF_1set_1twbr
  (JNIEnv *env, jclass clase, jint adapter_no, jint twbr, jint twps) {
	return   	OSIF_set_twbr( adapter_no, twbr, twps);
}


