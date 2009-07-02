/**
 * 
 */
package sibtra.ui.modulos;

import sibtra.lms.BarridoAngular;
import sibtra.rfycarro.FuturoObstaculo;
import sibtra.rfycarro.PanelFuturoObstaculo;
import sibtra.ui.VentanasMonitoriza;
import sibtra.util.ThreadSupendible;

/**
 * Módulo obstáculo que utiliza {@link FuturoObstaculo} y añade panel {@link PanelFuturoObstaculo} a la derecha
 * 
 * @author alberto
 *
 */
public class RFyCarro implements DetectaObstaculos {
	
	static final String NOMBRE="Range Finder y direccion";
	static final String DESCRIPCION="Detecta la distancia libre basándose en el Range finder y la posición de la dirección";
	
	private final VentanasMonitoriza ventMonitoriza;
	private FuturoObstaculo futObstaculo;
	private PanelFuturoObstaculo panelFutObstaculo;
	private ThreadSupendible thActulizacion;
	private double distanciaLibre;
	
	public RFyCarro(final VentanasMonitoriza ventanaMonitoriza) {
		this.ventMonitoriza=ventanaMonitoriza;
		
		futObstaculo=new FuturoObstaculo();
		panelFutObstaculo=new PanelFuturoObstaculo(futObstaculo);
		//lo añadimos al lado derecho sin scroll
		ventMonitoriza.añadePanel(panelFutObstaculo, NOMBRE, true, false);
		
		//Definimos el thread de actualizacion
		thActulizacion=new ThreadSupendible() {
			BarridoAngular ba=null;
			@Override
			protected void accion() {
				ba=ventMonitoriza.conexionRF.esperaNuevoBarrido(ba);
				double alfa=ventMonitoriza.conexionCarro.getAnguloVolante();
				distanciaLibre=futObstaculo.distanciaAObstaculo(alfa, ba);
				panelFutObstaculo.setBarrido(ba);
				panelFutObstaculo.actualiza();
			}
		};
		thActulizacion.setName(NOMBRE);
	}

	/**
	 * @see sibtra.ui.modulos.DetectaObstaculos#getDistanciaLibre()
	 */
	public double getDistanciaLibre() {
		return distanciaLibre;
	}

	/**
	 * @see sibtra.ui.modulos.Modulo#getDescripcion()
	 */
	public String getDescripcion() {
		return DESCRIPCION;
	}

	/**
	 * @see sibtra.ui.modulos.Modulo#getNombre()
	 */
	public String getNombre() {
		return NOMBRE;
	}

	/**
	 * Sólo suspende {@link #thActulizacion}.
	 * @see sibtra.ui.modulos.Modulo#terminar()
	 */
	public void terminar() {
		thActulizacion.suspender();
	}

}
