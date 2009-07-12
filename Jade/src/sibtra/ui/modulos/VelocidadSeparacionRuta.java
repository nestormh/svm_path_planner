/**
 * 
 */
package sibtra.ui.modulos;

import javax.swing.JOptionPane;

import sibtra.gps.Trayectoria;
import sibtra.predictivo.Coche;
import sibtra.ui.VentanasMonitoriza;
import sibtra.util.LabelDatoFormato;
import sibtra.util.PanelFlow;
import sibtra.util.SpinnerDouble;

/**
 * @author alberto
 *
 */
public class VelocidadSeparacionRuta implements CalculoVelocidad {
	
	String NOMBRE="Velocidad Ruta";
	String DESCRIPCION="Velocidad según ruta, se minora con la distancia lateral y error de orientación";
	private VentanasMonitoriza ventanaMonitoriza;
	private Trayectoria Tr;
	private PanelFlow panelDatos;
	// Parametros ======================================================
	private double gananciaLateral=1;
	private double gananciaVelocidad=2;
	private double velocidadMaxima=2.5;
	private double factorReduccionV=0.7;
	private double velocidadMinima=1;
	// variables interesantes ===========================================
	private double errorLateral;
	private double errorOrientacion;
	private double velocidadReferencia;
	private double consigna;
	private Coche modCoche;
	
	public VelocidadSeparacionRuta() {};

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
		
		//obtenemos modelo del coche
		modCoche=ventanaMonitoriza.getMotor().getModeloCoche();
		
		//Definimos panel y ponemos ajuste para los parámetros y etiquetas con las variables
		panelDatos=new PanelFlow();
		panelDatos.añadeAPanel(new LabelDatoFormato(this.getClass(),"getConsigna","%5.2f m/s"), "Consigna");
		panelDatos.añadeAPanel(new LabelDatoFormato(this.getClass(),"getErrorLateral","%5.2f m"), "Err Lat");
		panelDatos.añadeAPanel(new LabelDatoFormato(this.getClass(),"getErrorOrientacionGrados","%5.2f º"), "Err Ori");
		panelDatos.añadeAPanel(new LabelDatoFormato(this.getClass(),"getVelocidadReferencia","%5.2f m/s"), "Vel Ref");
		
		panelDatos.añadeAPanel(new SpinnerDouble(this,"setFactorReduccionV",0.05,1,0.05), "Fact Reduc");
		panelDatos.añadeAPanel(new SpinnerDouble(this,"setGananciaLateral",0.1,10,0.1), "Gan Lat");
		panelDatos.añadeAPanel(new SpinnerDouble(this,"setGananciaVelocidad",0.1,10,0.1), "Gan Vel");
		panelDatos.añadeAPanel(new SpinnerDouble(this,"setVelocidadMaxima",1,7,0.25), "Vel Maxima");
		panelDatos.añadeAPanel(new SpinnerDouble(this,"setVelocidadMinima",1,7,0.25), "Vel Minima");
		
		ventanaMonitoriza.añadePanel(panelDatos, "Vel Ruta", false, false);

		return true;
	}

	/**
	 * Método para decidir la consigna de velocidad para cada instante.
	 * Se tiene en cuenta el error en la orientación y el error lateral para reducir la 
	 * consigna de velocidad. 
	 * @return
	 */
	public double getConsignaVelocidad() {
		if(ventanaMonitoriza==null)
			throw new IllegalStateException("Aun no inicializado");
		consigna = 0;
		//obtenemos posicion y orientación del modelo del coche.
        double angAct = modCoche.getTita();
		int indMin = Tr.indiceMasCercano();  //la posición del coche ya la ha puesto el motor
		errorLateral = Tr.distanciaAlMasCercano();
		errorOrientacion = Tr.rumbo[indMin] - angAct;
		velocidadReferencia=Tr.velocidad[indMin];
		//referencia minorada
		consigna=velocidadReferencia*factorReduccionV;
		//acotamos a velocidad máxima
		if (consigna>velocidadMaxima)
			consigna = velocidadMaxima;
		//minoramos la consigna con los errores
		consigna -=  Math.abs(errorOrientacion)*gananciaVelocidad + Math.abs(errorLateral)*gananciaLateral;        
		// Solo con esta condición el coche no se detiene nunca,aunque la referencia de la ruta sea cero
		if (consigna <= velocidadMinima)
			if( velocidadReferencia >= velocidadMinima )
				// Con esta condición se contempla el caso de que la consigna sea < 0
				consigna = velocidadMinima;
			else 
				// De esta manera si la velocidad de la ruta disminuye hasta cero el coche se 
				// detiene, en vez de seguir a velocidad mínima como ocurría antes. En este caso también
				// está contemplado el caso de que la consigna sea < 0
				consigna = velocidadReferencia;
		//actulizamos la presetación
		panelDatos.actualizaDatos(this);
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

	/* (non-Javadoc)
	 * @see sibtra.ui.modulos.Modulo#terminar()
	 */
	public void terminar() {
		if(ventanaMonitoriza==null)
			throw new IllegalStateException("Aun no inicializado");
		ventanaMonitoriza.quitaPanel(panelDatos);

	}

	public double getConsigna() {
		return consigna;
	}

	public double getErrorLateral() {
		return errorLateral;
	}

	public double getErrorOrientacionGrados() {
		return Math.toDegrees(errorOrientacion);
	}

	public double getFactorReduccionV() {
		return factorReduccionV;
	}

	public double getGananciaLateral() {
		return gananciaLateral;
	}

	public double getGananciaVelocidad() {
		return gananciaVelocidad;
	}

	public double getVelocidadMaxima() {
		return velocidadMaxima;
	}

	public double getVelocidadMinima() {
		return velocidadMinima;
	}

	public double getVelocidadReferencia() {
		return velocidadReferencia;
	}

	public void setFactorReduccionV(double factorReduccionV) {
		this.factorReduccionV = factorReduccionV;
	}

	public void setGananciaLateral(double gananciaLateral) {
		this.gananciaLateral = gananciaLateral;
	}

	public void setGananciaVelocidad(double gananciaVelocidad) {
		this.gananciaVelocidad = gananciaVelocidad;
	}

	public void setVelocidadMaxima(double velocidadMaxima) {
		this.velocidadMaxima = velocidadMaxima;
	}

	public void setVelocidadMinima(double velocidadMinima) {
		this.velocidadMinima = velocidadMinima;
	}

}
