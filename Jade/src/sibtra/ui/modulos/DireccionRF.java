package sibtra.ui.modulos;

import sibtra.lms.BarridoAngular;
import sibtra.ui.VentanasMonitoriza;
import sibtra.ui.defs.CalculoDireccion;
import sibtra.ui.modulos.MotorSincrono.PanelSincrono;
import sibtra.util.LabelDatoFormato;
import sibtra.util.PanelFlow;

public class DireccionRF implements CalculoDireccion {
	String NOMBRE="Direccion RangeFinder";
	String DESCRIPCION="Calcula el angulo del volante para aproximarse al objetivo mas cercano";
	private VentanasMonitoriza ventanaMonitoriza;
	private BarridoAngular ba=null;
	private double[] angDistRF={0,80};
	private double consignaDir;
	private double distancia;
	private PanelDirRF panel;

	public double getConsignaDireccion() {
		BarridoAngular nuevoBa=ventanaMonitoriza.conexionRF.ultimoBarrido();		
		if(nuevoBa!=ba && nuevoBa!=null) {
			ba=nuevoBa;
			angDistRF = getAnguloDistObjetivo(ba);
		}
		consignaDir = -(angDistRF[0]-Math.PI);
		distancia = angDistRF[1];
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
		for (int i=0; i<=ba.numDatos();i++){
			if (ba.getDistancia(i)< distMin){
				indMinDist = i;
				distMin = ba.getDistancia(i);
			}
		}
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
		ventanaMonitoriza.añadePanel(panel, getNombre(),false,false);

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
			
		}
	}

	public double getConsignaDirGrados() {
		return Math.toDegrees(consignaDir);
	}

	public double getDistancia() {
		return distancia;
	}

}
