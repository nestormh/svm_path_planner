/**
 * 
 */
package sibtra.ui.modulos;

import sibtra.gps.GPSData;
import sibtra.gps.Trayectoria;
import sibtra.lms.BarridoAngular;
import sibtra.rfyruta.MiraObstaculo;
import sibtra.rfyruta.PanelMiraObstaculo;
import sibtra.rfyruta.PanelMiraObstaculoSubjetivo;
import sibtra.ui.VentanasMonitoriza;
import sibtra.ui.defs.DetectaObstaculos;
import sibtra.ui.defs.Motor;
import sibtra.ui.defs.SubModuloUsaTrayectoria;
import sibtra.util.ThreadSupendible;

/**
 * @author alberto
 *
 */
public class RFyRuta implements DetectaObstaculos, SubModuloUsaTrayectoria {

	static final String NOMBRE="RF y Ruta";
	static final String DESCRIPCION="Detecta obstaculos basandose en el RF y la Ruta";

	private VentanasMonitoriza ventanaMonitoriza=null;
	private MiraObstaculo miraObstaculo;
	private PanelMiraObstaculo panelMiraObs;
	private PanelMiraObstaculoSubjetivo panelMiraObsSub;
	private ThreadSupendible thActulizacion;
	private double distanciaLibre;
	private Trayectoria Tr=null;
	private Motor motor=null;

	public RFyRuta() {};
	/* (sin Javadoc)
	 * @see sibtra.ui.modulos.Modulo#setVentanaMonitoriza(sibtra.ui.VentanasMonitoriza)
	 */
	public boolean setVentanaMonitoriza(VentanasMonitoriza ventMonitoriza) {
		if(ventanaMonitoriza!=null) {
			throw new IllegalStateException("Modulo ya inicializado, no se puede volver a inicializar");
		}
		ventanaMonitoriza=ventMonitoriza;
		//Debemos tener el motor ya
		if(motor==null)
			throw new IllegalStateException("Se debe haber fijado el motor antes de fijar VentanaMonitoriza");
			
		motor.apuntaNecesitaTrayectoria(this);
		miraObstaculo=new MiraObstaculo();
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
	            	angAct = ventanaMonitoriza.declinaMag.rumboVerdadero(pa.getAngulosIMU());
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
	 * Termina el {@link #thActulizacion}, quita los dos paneles y libera trayectoria
	 */
	public void terminar() {
		if(ventanaMonitoriza==null)
			throw new IllegalStateException("Aun no inicializado");
		thActulizacion.terminar();
		ventanaMonitoriza.quitaPanel(panelMiraObs);
		ventanaMonitoriza.quitaPanel(panelMiraObsSub);
	}
	
	/** Indicamos el cambio a {@link #miraObstaculo} */
	private void nuevaTrayectoria(Trayectoria tra) {
		miraObstaculo.nuevaTrayectoria(tra);
		panelMiraObs.setTrayectoria(tra);
		panelMiraObs.actualiza();
		Tr=tra;
	}
	public void setMotor(Motor mtr) {
		motor=mtr;
	}
	public void setTrayectoriaInicial(Trayectoria tra) {
		nuevaTrayectoria(tra);
	}
	public void setTrayectoriaModificada(Trayectoria tra) {
		nuevaTrayectoria(tra);
	}

}
