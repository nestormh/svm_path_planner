
#include <jni.h>
#include <stdio.h>
#include "shmCInterface.h"
#include "shmInterface.h"

JNIEXPORT void JNICALL Java_sibtra_shm_shmInterface_shmSafeGet (JNIEnv *env, jobject obj) {

jint shmid;
jclass cls;
jfieldID fid;
	
	cls = (*env)->GetObjectClass(env,obj);
	fid = (*env)->GetFieldID(env,cls,"shmid","I");
	shmid = shmSafeGet();
	//printf ("shmid en C (Get)%d\n", shmid);
	(*env)->SetIntField(env,obj,fid,shmid);
	
return;
}

JNIEXPORT void JNICALL Java_sibtra_shm_shmInterface_shmSafeErase (JNIEnv *env, jobject obj) {

jint shmid;
jclass cls;
jfieldID fid;
	
	cls = (*env)->GetObjectClass(env,obj);
	fid = (*env)->GetFieldID(env,cls,"shmid","I");
	shmid = (*env)->GetIntField(env,obj,fid);
	shmSafeErase(shmid);	
	//printf ("shmid en C (Erase)%d\n", shmid);
	
}

JNIEXPORT void JNICALL Java_sibtra_shm_shmInterface_shmReadGPSOrientation (JNIEnv *env, jobject obj) {
	
jint shmid, gpsOrientation;
jclass cls;
jfieldID fid1,fid2;

shmStruct *p;

	cls = (*env)->GetObjectClass(env,obj);
	fid1 = (*env)->GetFieldID(env,cls,"gpsOrientation","I");
	fid2 = (*env)->GetFieldID(env,cls,"shmid","I");
	shmid = (*env)->GetIntField(env,obj,fid2);
	
	gpsOrientation = shmReadGPSOrientation(shmid);
	
	(*env)->SetIntField(env,obj,fid1,gpsOrientation);
	//printf ("orientacion gps en C %d\n", gpsOrientation);
	
	
}

JNIEXPORT void JNICALL Java_sibtra_shm_shmInterface_shmWriteGPSOrientation (JNIEnv *env, jobject obj) {

jint shmid, gpsOrientation;
jclass cls;
jfieldID fid1,fid2;

shmStruct *p;

	cls = (*env)->GetObjectClass(env,obj);
	fid1 = (*env)->GetFieldID(env,cls,"gpsOrientation","I");
	gpsOrientation = (*env)->GetIntField(env,obj,fid1);
	fid2 = (*env)->GetFieldID(env,cls,"shmid","I");
	shmid = (*env)->GetIntField(env,obj,fid2);
	
	shmWriteGPSOrientation(shmid,gpsOrientation);
	
