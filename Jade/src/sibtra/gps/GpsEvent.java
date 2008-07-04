/**
 * 
 */
package sibtra.gps;

import java.util.EventObject;

/**
 * @author alberto
 *
 */
public class GpsEvent extends EventObject {

	private GPSData nuevoPunto;
	/**
	 * @param arg0
	 */
	public GpsEvent(Object arg0, GPSData nuevoPto) {
		super(arg0);
		this.nuevoPunto=nuevoPto;
	}
	/**
	 * @return the nuevoPunto
	 */
	public GPSData getNuevoPunto() {
		return nuevoPunto;
	}

}
