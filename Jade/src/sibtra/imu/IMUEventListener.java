/**
 * 
 */
package sibtra.imu;

/**
 * Interfaz que deben cumplir aquellos que quiran recibir {@link IMUEvent}
 * @author alberto
 */
public interface IMUEventListener {
	
	public void handleIMUEvent(IMUEvent ev);


}
