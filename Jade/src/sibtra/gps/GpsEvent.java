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
	private boolean espacial;
	/**
	 * @param arg0
	 * @param seAñadeEspacial 
	 */
	public GpsEvent(Object arg0, GPSData nuevoPto, boolean seAñadeEspacial) {
		super(arg0);
		this.nuevoPunto=nuevoPto;
		this.espacial=seAñadeEspacial;
	}
	/**
	 * @return the nuevoPunto
	 */
	public GPSData getNuevoPunto() {
		return nuevoPunto;
	}
	
	/**
	 * @return si el punto se añadió a buffer espacial
	 */
	public boolean isEspacial() {
		return espacial;
	}

}
