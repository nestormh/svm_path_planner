/**
 * 
 */
package sibtra.imu;

import java.io.Serializable;

/**
 * Clase para contener los angulos recibidos de la IMU
 * @author alberto
 */
public class AngulosIMU implements Serializable {
	
	
	/**
	 * Versión para comtaibilidad al serializar
	 */
	private static final long serialVersionUID = 1L;
	double roll;
	double pitch;
	double yaw;
	int	contador;
	/**
	 * Hora del ordenador cuando se obtuvo el punto (en milisegundos).
	 * Como lo devuelve la llamada <code>System.currentTimeMillis()</code>.
	 */
	long sysTime=0;
	
	public AngulosIMU() {
		this(Double.NaN,Double.NaN,Double.NaN,Integer.MIN_VALUE);
	}
	
	public AngulosIMU(double roll,double pitch,double yaw, int contador) {
		this.roll=roll;
		this.pitch=pitch;
		this.yaw=yaw;
		this.contador=contador;
		sysTime=System.currentTimeMillis();
	}

	/** constructor de copia */
	public AngulosIMU(AngulosIMU aiOrig) {
		this.copy(aiOrig);
	}
	
	/** copia contenido a este objeto */
	public void copy(AngulosIMU aiOrig) {
		roll=aiOrig.roll;
		pitch=aiOrig.pitch;
		yaw=aiOrig.yaw;
		contador=aiOrig.contador;
		sysTime=aiOrig.sysTime;
	}

	/**
	 * @return the roll en grados
	 */
	public double getRoll() {
		return roll;
	}

	/**
	 * @return the pitch en grados
	 */
	public double getPitch() {
		return pitch;
	}

	/**
	 * @return the yaw en grados
	 */
	public double getYaw() {
		return yaw;
	}

	/**
	 * @return the contador
	 */
	public int getContador() {
		return contador;
	}

	/**
	 * @return the sysTime
	 */
	public long getSysTime() {
		return sysTime;
	}
	

	public String toString() {
		return String.format("[(%5d)R=%+8.2f P=%+8.2f Y=%+8.2f",contador,roll,pitch,yaw );
	}

}
