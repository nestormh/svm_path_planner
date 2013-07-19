/**
 * 
 */
package sibtra.ui.modulos;

import sibtra.gps.Trayectoria;
import sibtra.ui.VentanasMonitoriza;
import sibtra.ui.defs.ModificadorTrayectoria;
import sibtra.ui.defs.Motor;

/**
 * Modulo de prueba. No modifica la ruta nunca
 * 
 * @author alberto
 *
 */
public class ModificadorNada implements ModificadorTrayectoria {

	/** No la recogemos porque no la vamos a modificar
	 * @see sibtra.ui.defs.ModificadorTrayectoria#setTrayectoriaInicial(sibtra.gps.Trayectoria)
	 */
	public void setTrayectoriaInicial(Trayectoria tra) {
	}

	/** No lo recogemos porque no lo necesitamos. 
	 * @see sibtra.ui.defs.SubModulo#setMotor(sibtra.ui.defs.Motor)
	 */
	public void setMotor(Motor mtr) {
	}

	public String getDescripcion() {
		return "Modificador de prueba que no hace nada";
	}

	public String getNombre() {
		return "Modificador Nada";
	}

	/** No hacemos nada */
	public boolean setVentanaMonitoriza(VentanasMonitoriza ventMonitoriza) {
		return true;
	}

	/** No hacemos nada */
	public void terminar() {
	}

	public void actuar() {
		// TODO Auto-generated method stub
		
	}

	public void parar() {
		// TODO Auto-generated method stub
		
	}

}
