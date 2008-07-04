/**
 * 
 */
package sibtra.imu;

import java.util.EventObject;

/**
 * Evento que se lanza al recibir unos nuevos angulos.
 * @author alberto
 */
public class IMUEvent  extends EventObject {

	private AngulosIMU angulos;
	/**
	 * @param arg0
	 */
	public IMUEvent(Object arg0, AngulosIMU angulos) {
		super(arg0);
		this.angulos=angulos;
	}
	/**
	 * @return the nuevoPunto
	 */
	public AngulosIMU getAngulos() {
		return angulos;
	}

}
