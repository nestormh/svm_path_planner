package sibtra.ui.modulos;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.AbstractAction;
import javax.swing.JButton;

import sibtra.lms.BarridoAngular;
import sibtra.ui.VentanasMonitoriza;
import sibtra.ui.defs.CalculoDireccion;
import sibtra.ui.modulos.MotorSincrono.PanelSincrono;
import sibtra.util.LabelDatoFormato;
import sibtra.util.PanelFlow;
import sibtra.util.ThreadSupendible;

public class DireccionRF implements CalculoDireccion {
	String NOMBRE="Direccion RangeFinder";
	String DESCRIPCION="Calcula el angulo del volante para aproximarse al objetivo mas cercano";
	private VentanasMonitoriza ventanaMonitoriza;
	private BarridoAngular ba=null;
	private double[] angDistRF={0,80};
	private double consignaDir;
	private double distancia;
	private PanelDirRF panel;
	private int indMinAnt = -1;
	private ThreadSupendible thActulizacion;
	private int rangoInd = 10;

	public double getConsignaDireccion() {		
		return consignaDir;
	}
	
	/**
	 * Se encarga de encontrar el punto más cercano al coche (supuesto objetivo para 
	 * ser perseguido)
	 * @param ba barrido completo del rangeFinder
	 * @return distancia y ángulo en el que se encuentra el objetivo (obstáculo más cercano)
	 */
	public double[] getAnguloDistObjetivo(BarridoAngular ba){
		double distMin = Double.POSITIVE_INFINITY;
		double[] anguloDistRF = new double[2];
		int indMinDist = 0;
		if (indMinAnt < 0){ // Búsqueda exaustiva
			for (int i=0; i<ba.numDatos();i++){
				if (ba.getDistancia(i)< distMin){
					indMinDist = i;
					distMin = ba.getDistancia(i);
				}
			}
		}else { // Búsqueda en torno al ángulo donde se detectó el objetivo anteriormente
			int indInf = ((indMinAnt - rangoInd ) < 0)?0:(indMinAnt - rangoInd);
			int indSup = ((indMinAnt + rangoInd) > ba.numDatos())?ba.numDatos():(indMinAnt + rangoInd);
			for(int i=indInf; i < indSup;i++){
				if (ba.getDistancia(i)< distMin){
					indMinDist = i;
					distMin = ba.getDistancia(i);
				}
			}
		}		
		indMinAnt = indMinDist;
		anguloDistRF[0] = ba.getAngulo(indMinDist);
		anguloDistRF[1] = ba.getDistancia(indMinDist);
		return anguloDistRF;
	}
	

	public String getDescripcion() {
		return DESCRIPCION;
	}

	public String getNombre() {
		return NOMBRE;
	}

	public boolean setVentanaMonitoriza(VentanasMonitoriza ventMonitoriza) {
		if(ventanaMonitoriza!=null)
			throw new IllegalStateException("Modulo ya inicializado, no se puede volver a inicializar");
		ventanaMonitoriza=ventMonitoriza;
		panel=new PanelDirRF();
		ventanaMonitoriza.añadePanel(panel,NOMBRE);
		thActulizacion=new ThreadSupendible() {
			BarridoAngular ba=null;
			@Override
			protected void accion() {
				ba=ventanaMonitoriza.conexionRF.esperaNuevoBarrido(ba);
				angDistRF = getAnguloDistObjetivo(ba);
				consignaDir = angDistRF[0]-Math.PI/2;				
			}
		};
		thActulizacion.setName(NOMBRE);
		thActulizacion.activar();
		

		return true;
	}

	public void terminar() {
		// TODO Auto-generated method stub

	}
	
	protected class PanelDirRF extends PanelFlow {
		public PanelDirRF() {
			super();
//			setLayout(new GridLayout(0,4));
			//TODO Definir los tamaños adecuados o poner layout
//			añadeAPanel(new SpinnerDouble(DireccionRF.this,"getConsignaDir",0,6,0.1), "Min Vel");

			//TODO ponel labels que muestren la informacion recibida de los otros módulos y la que se aplica.
			añadeAPanel(new LabelDatoFormato(DireccionRF.class,"getConsignaDirGrados","%6.2f º"), "Ang RF");
			añadeAPanel(new LabelDatoFormato(DireccionRF.class,"getDistancia","%6.2f m"), "Dist RF");
			IniciaBusqueda iniBusqueda = new IniciaBusqueda();
			añadeAPanel(new JButton(iniBusqueda), "Reiniciar Búsqueda");			
			
		}
	}

	public double getConsignaDirGrados() {
		return Math.toDegrees(consignaDir);
	}

	public double getDistancia() {
		return distancia;
	}
	
	class IniciaBusqueda extends AbstractAction{
		
		public IniciaBusqueda(){
			super("Reinicia la búsqueda");
			setEnabled(true);
		}

		public void actionPerformed(ActionEvent e) {
			indMinAnt = -1;
			setEnabled(true);			
		}
		
	}

}
