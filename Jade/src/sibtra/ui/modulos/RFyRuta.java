/**
 * 
 */
package sibtra.ui.modulos;

import javax.swing.JOptionPane;

import sibtra.gps.GPSData;
import sibtra.gps.Trayectoria;
import sibtra.lms.BarridoAngular;
import sibtra.rfyruta.MiraObstaculo;
import sibtra.rfyruta.PanelMiraObstaculo;
import sibtra.rfyruta.PanelMiraObstaculoSubjetivo;
import sibtra.ui.VentanasMonitoriza;
import sibtra.util.ThreadSupendible;

/**
 * @author alberto
 *
 */
public class RFyRuta implements DetectaObstaculos {

	static final String NOMBRE="RF y Ruta";
	static final String DESCRIPCION="Detecta obstaculos basandose en el RF y la Ruta";

	private VentanasMonitoriza ventanaMonitoriza=null;
	private MiraObstaculo miraObstaculo;
	private PanelMiraObstaculo panelMiraObs;
	private PanelMiraObstaculoSubjetivo panelMiraObsSub;
	private ThreadSupendible thActulizacion;
	private double distanciaLibre;
	private Trayectoria Tr;

	public RFyRuta() {};
	/* (sin Javadoc)
	 * @see sibtra.ui.modulos.Modulo#setVentanaMonitoriza(sibtra.ui.VentanasMonitoriza)
	 */
	public boolean setVentanaMonitoriza(VentanasMonitoriza ventMonitoriza) {
		if(ventanaMonitoriza!=null) {
			throw new IllegalStateException("Modulo ya inicializado, no se puede volver a inicializar");
		}
		ventanaMonitoriza=ventMonitoriza;
		Tr=ventanaMonitoriza.getTrayectoriaSeleccionada();
		if(Tr==null) {
			JOptionPane.showMessageDialog(ventanaMonitoriza.ventanaPrincipal,
				    "El módulo "+NOMBRE+" necesita ruta para continuar.",
				    "Sin ruta",
				    JOptionPane.ERROR_MESSAGE);
			ventanaMonitoriza=null;
			return false;

		}
		miraObstaculo=new MiraObstaculo(Tr);
		
		//creamos los paneles y los añadimos
		panelMiraObs=new PanelMiraObstaculo(miraObstaculo);
		ventanaMonitoriza.añadePanel(panelMiraObs, NOMBRE, true, false);
		panelMiraObsSub=new PanelMiraObstaculoSubjetivo(miraObstaculo,(short)80);
		ventanaMonitoriza.añadePanel(panelMiraObsSub, NOMBRE+" Sub", true, false);
		
		//Definimos el thread que estará pendiente de los nuevos barridos 
		thActulizacion=new ThreadSupendible() {
			BarridoAngular ba=null;
			protected void accion() {
				double dl=Double.POSITIVE_INFINITY;
				ba=ventanaMonitoriza.conexionRF.esperaNuevoBarrido(ba);
				GPSData pa = ventanaMonitoriza.conexionGPS.getPuntoActualTemporal();                            
	            double angAct=Double.NaN;
	            if(pa!=null) {
	            	angAct = Math.toRadians(pa.getAngulosIMU().getYaw()) + ventanaMonitoriza.getDesviacionMagnetica();
	            	Tr.situaCoche(pa.getXLocal(), pa.getYLocal());
					//calculamos distancia a obstáculo más cercano
					dl = miraObstaculo.masCercano(angAct, ba);
				} else {
					//no podemos calcular nada
					dl = Double.POSITIVE_INFINITY; 
				}
	            if(Double.isNaN(dl))
	            	dl=Double.POSITIVE_INFINITY;
				//actualizamos paneles
	            distanciaLibre=dl;
				panelMiraObs.actualiza();
				panelMiraObsSub.actualiza();
			}
		};
		thActulizacion.setName(NOMBRE);
		thActulizacion.activar();
		return true; //inicialización correcta
	}

	public double getDistanciaLibre() {
		if(ventanaMonitoriza==null)
			throw new IllegalStateException("Aun no inicializado");
		return distanciaLibre;
	}

	public String getDescripcion() {
		return DESCRIPCION;
	}

	public String getNombre() {
		return NOMBRE;
	}

	/**
	 * Termina el {@link #thActulizacion} y quita los dos paneles
	 */
	public void terminar() {
		if(ventanaMonitoriza==null)
			throw new IllegalStateException("Aun no inicializado");
		thActulizacion.terminar();
		ventanaMonitoriza.quitaPanel(panelMiraObs);
		ventanaMonitoriza.quitaPanel(panelMiraObsSub);
	}

}
