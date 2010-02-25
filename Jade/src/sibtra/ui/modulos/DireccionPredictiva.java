/**
 * 
 */
package sibtra.ui.modulos;

import sibtra.gps.Trayectoria;
import sibtra.predictivo.Coche;
import sibtra.predictivo.ControlPredictivo;
import sibtra.predictivo.PanelMuestraPredictivo;
import sibtra.ui.VentanasMonitoriza;
import sibtra.ui.defs.CalculoDireccion;
import sibtra.ui.defs.Motor;
import sibtra.ui.defs.SubModuloUsaTrayectoria;
import sibtra.util.LabelDatoFormato;
import sibtra.util.PanelFlow;

/**
 * @author alberto
 *
 */
public class DireccionPredictiva implements CalculoDireccion, SubModuloUsaTrayectoria {
	
	String NOMBRE="Direccion Predictiva";
	String DESCRIPCION="Calcula sólo la dirección usando control predictivo";
	private VentanasMonitoriza ventanaMonitoriza=null;
	private Trayectoria Tr=null;
	private ControlPredictivo controlPredictivo;
	private Coche modCoche=null;
	private PanelMuestraPredictivo panelPredictivo;
	private int periodoMuestreoMili=200;
	// Variables ===========================================================
	private double consigna;
	private PanelFlow panelPropio;
	private Motor motor=null;
	
	public DireccionPredictiva() {}


	/* (non-Javadoc)
	 * @see sibtra.ui.modulos.Modulo#setVentanaMonitoriza(sibtra.ui.VentanasMonitoriza)
	 */
	public boolean setVentanaMonitoriza(VentanasMonitoriza ventMonitoriza) {
		if(ventanaMonitoriza!=null)
			throw new IllegalStateException("Modulo ya inicializado, no se puede volver a inicializar");
		ventanaMonitoriza=ventMonitoriza;
		//Debemos tener el motor ya
		if(motor==null)
			throw new IllegalStateException("Se debe haber fijado el motor antes de fijar VentanaMonitoriza");
			
		motor.apuntaNecesitaTrayectoria(this);

		modCoche=motor.getModeloCoche();
        //Inicializamos modelos predictivos
        controlPredictivo = new ControlPredictivo(modCoche, Tr, 13, 4, 2.0
        		, (double) periodoMuestreoMili / 1000);
		//Definimos panel y ponemos ajuste para los parámetros y etiquetas con las variables
        panelPredictivo=new PanelMuestraPredictivo(controlPredictivo);
        panelPropio=new PanelFlow();
        panelPropio.añadeAPanel(new LabelDatoFormato(this.getClass(),"getConsignaGrados","%5.2f º"), "Consigna");
        panelPredictivo.add(panelPropio);
        ventanaMonitoriza.añadePanel(panelPredictivo, "Predictivo",true,false);


		return true;
	}

	/* (non-Javadoc)
	 * @see sibtra.ui.modulos.CalculoDireccion#getConsignaDireccion()
	 */
	public double getConsignaDireccion() {
		if(ventanaMonitoriza==null)
			throw new IllegalStateException("Aun no inicializado");
		if(Tr==null)
			throw new IllegalStateException("La trayectoria es null");
        consigna = controlPredictivo.calculaComando();       
        panelPredictivo.actualiza();
        panelPropio.actualizaDatos(this);
		return consigna;
	}

	/* (sin Javadoc)
	 * @see sibtra.ui.modulos.Modulo#getDescripcion()
	 */
	public String getDescripcion() {
		return DESCRIPCION;
	}

	/* (sin Javadoc)
	 * @see sibtra.ui.modulos.Modulo#getNombre()
	 */
	public String getNombre() {
		return NOMBRE;
	}

	/** quitamos el {@link #panelPredictivo} de la {@link #ventanaMonitoriza} */
	public void terminar() {
		if(ventanaMonitoriza==null)
			throw new IllegalStateException("Aun no inicializado");
		ventanaMonitoriza.quitaPanel(panelPredictivo);
	}
	
	public double getConsignaGrados() {
		return Math.toDegrees(consigna);
	}
	
	/** Cambiamos la trayectoria en {@link #controlPredictivo} y en {@link #panelPredictivo} */
	private void nuevaTrayectoria(Trayectoria tr) {
		if(tr==null)
			throw new IllegalArgumentException("La trayectoria no pude ser null");
		Tr=tr;
		controlPredictivo.setRuta(Tr);
		panelPredictivo.setTrayectoria(Tr);
		panelPredictivo.actualiza();
	}


	public void setMotor(Motor mtr) {
		motor=mtr;
	}

	public void setTrayectoriaModificada(Trayectoria tr) {
		nuevaTrayectoria(tr);
	}
	/** Apuntamos la trayectoria inicial */
	public void setTrayectoriaInicial(Trayectoria tra) {
		nuevaTrayectoria(tra);
	}

}
