package sibtra.odometria;

import java.util.EventObject;

public class OdometriaEvent extends EventObject{
	
	private DatosOdometria odo;
	/**
	 * @param arg0
	 */
	public OdometriaEvent(Object arg0, DatosOdometria odom) {
		super(arg0);
		this.odo=odom;
	}
	/**
	 * @return the nuevoPunto
	 */
	public DatosOdometria getDatosOdometria() {
		return odo;
	}

}
