/**
 * 
 */
package sibtra.ui.modulos;

import sibtra.ui.VentanasMonitoriza;

/**
 * @author alberto
 *
 */
public class RFyRuta implements DetectaObstaculos {

	static final String NOMBRE="RF y Ruta";
	static final String DESCRIPCION="Detecta obstaculos basandose en el RF y la Ruta";

	public RFyRuta() {};
	/* (sin Javadoc)
	 * @see sibtra.ui.modulos.Modulo#setVentanaMonitoriza(sibtra.ui.VentanasMonitoriza)
	 */
	public void setVentanaMonitoriza(VentanasMonitoriza ventMonitoriza) {
		// TODO Apéndice de método generado automáticamente

	}

	/* (sin Javadoc)
	 * @see sibtra.ui.modulos.DetectaObstaculos#getDistanciaLibre()
	 */
	public double getDistanciaLibre() {
		// TODO Apéndice de método generado automáticamente
		return 0;
	}

	public String getDescripcion() {
		return DESCRIPCION;
	}

	public String getNombre() {
		return NOMBRE;
	}


	/* (sin Javadoc)
	 * @see sibtra.ui.modulos.Modulo#terminar()
	 */
	public void terminar() {
		// TODO Apéndice de método generado automáticamente

	}

}
