package sibtra.shm;

public class shmInterface {

	int shmid;
	int gpsOrientation;
	
	
	public static void main(String[] args) {
		shmInterface i = new shmInterface();
		i.shmSafeGet();
		
		//System.out.println("shmid en java "+i.shmid);
		i.gpsOrientation = 10;
		i.shmWriteGPSOrientation();
		i.shmSafeErase();
	}
	
	
	
	public native void shmSafeGet();
	public native void shmSafeErase();
	public native void shmWriteGPSOrientation();
	public native void shmReadGPSOrientation();
	
	
	static {
		System.load("/home/rafael/workspace/shareMemory/lib/sibtra_shm_shmInterface.so");
	}
}
