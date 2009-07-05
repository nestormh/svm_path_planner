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
	
	private VentanasMonitoriza ventanaMonitoriza=null;
	private FuturoObstaculo futObstaculo;
	private PanelFuturoObstaculo panelFutObstaculo;
	private ThreadSupendible thActulizacion;
	private double distanciaLibre;
	
	public RFyCarro() {};
	
	public boolean setVentanaMonitoriza(VentanasMonitoriza ventMonitoriza) {
		if(ventanaMonitoriza!=null) {
			throw new IllegalStateException("Modulo ya inicializado, no se puede volver a inicializar");
		}
		ventanaMonitoriza=ventMonitoriza;
		
		futObstaculo=new FuturoObstaculo();
		panelFutObstaculo=new PanelFuturoObstaculo(futObstaculo);
		//lo añadimos al lado derecho sin scroll
		ventMonitoriza.añadePanel(panelFutObstaculo, NOMBRE, true, false);
		
		//Definimos el thread de actualizacion
		thActulizacion=new ThreadSupendible() {
			BarridoAngular ba=null;
			@Override
			protected void accion() {
				ba=ventanaMonitoriza.conexionRF.esperaNuevoBarrido(ba);
				double alfa=ventanaMonitoriza.conexionCarro.getAnguloVolante();
				distanciaLibre=futObstaculo.distanciaAObstaculo(alfa, ba);
				panelFutObstaculo.setBarrido(ba);
				panelFutObstaculo.actualiza();
			}
		};
		thActulizacion.setName(NOMBRE);
		thActulizacion.activar();
		return true;
	}

	/**
	 * @see sibtra.ui.modulos.DetectaObstaculos#getDistanciaLibre()
	 */
	public double getDistanciaLibre() {
		if(ventanaMonitoriza==null)
			throw new IllegalStateException("Aun no inicializado");
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
	 * Termina el {@link #thActulizacion} y quita el {@link #panelFutObstaculo}
	 * @see sibtra.ui.modulos.Modulo#terminar()
	 */
	public void terminar() {
		if(ventanaMonitoriza==null)
			throw new IllegalStateException("Aun no inicializado");
		thActulizacion.terminar();
		ventanaMonitoriza.quitaPanel(panelFutObstaculo);
	}

}
