/**
 * 
 */
package sibtra.gps;

/**
 * Interfaz que deben cumplir aquellos objetos que quieran recibir {@link GpsEvent}
 * @author alberto
 */
public interface GpsEventListener {
	
	public void handleGpsEvent(GpsEvent ev);

}
