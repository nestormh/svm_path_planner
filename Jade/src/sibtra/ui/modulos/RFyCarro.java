/**
 * 
 */
package sibtra.ui.modulos;

import sibtra.rfycarro.FuturoObstaculo;
import sibtra.ui.VentanasMonitoriza;

/**
 * @author alberto
 *
 */
public class RFyCarro implements Obstaculos {
	
	static final String NONBRE="Range Finder y direccion";
	static final String DESCRIPCION="Detecta la distancia libre basándose en el Range finder y la posición de la dirección";
	
	private VentanasMonitoriza ventMonitoriza;
	private FuturoObstaculo futObstaculo;
	
	public RFyCarro(VentanasMonitoriza ventMonitoriza) {
		this.ventMonitoriza=ventMonitoriza;
		
		futObstaculo=new FuturoObstaculo();
		
		//creamos panel
	}

	/* (non-Javadoc)
	 * @see sibtra.ui.modulos.Obstaculos#getDistanciaLibre()
	 */
	public double getDistanciaLibre() {
		// TODO Auto-generated method stub
		return 0;
	}

	/* (non-Javadoc)
	 * @see sibtra.ui.modulos.Modulo#getDescripcion()
	 */
	public String getDescripcion() {
		// TODO Auto-generated method stub
		return DESCRIPCION;
	}

	/* (non-Javadoc)
	 * @see sibtra.ui.modulos.Modulo#getNombre()
	 */
	public String getNombre() {
		return NOMBRE;
	}

	/* (non-Javadoc)
	 * @see sibtra.ui.modulos.Modulo#terminar()
	 */
	public void terminar() {
		// TODO Auto-generated method stub

	}

}
