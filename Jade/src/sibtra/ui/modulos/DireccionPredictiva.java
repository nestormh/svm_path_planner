/**
 * 
 */
package sibtra.ui.modulos;

import javax.swing.JOptionPane;

import sibtra.predictivo.Coche;
import sibtra.predictivo.ControlPredictivo;
import sibtra.predictivo.PanelMuestraPredictivo;
import sibtra.ui.VentanasMonitoriza;
import sibtra.util.LabelDatoFormato;
import sibtra.util.PanelFlow;

/**
 * @author alberto
 *
 */
public class DireccionPredictiva implements CalculoDireccion {
	
	String NOMBRE="Direccion Predictiva";
	String DESCRIPCION="Calcula sólo la dirección usando control predictivo";
	private VentanasMonitoriza ventanaMonitoriza;
	private double[][] Tr;
	private ControlPredictivo controlPredictivo;
	private Coche modCoche;
	private PanelMuestraPredictivo panelPredictivo;
	private int periodoMuestreoMili=200;
	// Variables ===========================================================
	private double consigna;
	private PanelFlow panelPropio;
	
	public DireccionPredictiva() {};


	/* (non-Javadoc)
	 * @see sibtra.ui.modulos.Modulo#setVentanaMonitoriza(sibtra.ui.VentanasMonitoriza)
	 */
	public boolean setVentanaMonitoriza(VentanasMonitoriza ventMonitoriza) {
		if(ventanaMonitoriza!=null)
			throw new IllegalStateException("Modulo ya inicializado, no se puede volver a inicializar");
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

		modCoche=ventanaMonitoriza.getMotor().getModeloCoche();
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

}
