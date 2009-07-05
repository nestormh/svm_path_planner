/**
 * 
 */
package sibtra.ui.modulos;

import sibtra.ui.VentanasMonitoriza;

/**
 * @author alberto
 *
 */
public class EligeDestino implements SeleccionRuta {

	String NOMBRE="Elige Destino";
	String DESCRIPCION="Crea trayectoria seleccionando el destino";

	/* (non-Javadoc)
	 * @see sibtra.ui.modulos.SeleccionRuta#getTrayectoria()
	 */
	public double[][] getTrayectoria() {
		// TODO Auto-generated method stub
		return null;
	}

	/* (non-Javadoc)
	 * @see sibtra.ui.modulos.Modulo#getDescripcion()
	 */
	public String getDescripcion() {
		return DESCRIPCION;
	}

	public String getNombre() {
		return NOMBRE;
	}

	/* (non-Javadoc)
	 * @see sibtra.ui.modulos.Modulo#setVentanaMonitoriza(sibtra.ui.VentanasMonitoriza)
	 */
	public boolean setVentanaMonitoriza(VentanasMonitoriza ventMonitoriza) {
		// TODO Auto-generated method stub
		return false;
	}

	/* (non-Javadoc)
	 * @see sibtra.ui.modulos.Modulo#terminar()
	 */
	public void terminar() {
		// TODO Auto-generated method stub

	}

}
