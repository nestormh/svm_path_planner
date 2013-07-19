/**
 * 
 */
package sibtra.imu;

/**
 * Panel que muestra los angulo de la imu pasda. 
 * Se actuliza periodicamente usando {@link #actualiza()}
 * @author alberto
 *
 */
@SuppressWarnings("serial")
public class PanelIMU extends PanelMuestraAngulosIMU {
	
	ConexionSerialIMU conSerIMU;
	
	public PanelIMU(ConexionSerialIMU conSerIMU) {
		super(); //solo espaciales
		if(conSerIMU==null)
			throw new IllegalArgumentException("Conexion a IMU no puede ser null");
		this.conSerIMU=conSerIMU;
	}
	
	/** En la actulización periodica usamos el último ángulo */
	protected void actualiza() {
		actualizaAngulo(conSerIMU.getAngulo());
		super.actualiza();
	}

}
