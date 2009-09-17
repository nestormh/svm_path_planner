/**
 * 
 */
package sibtra.ui.modulos;

import sibtra.gps.GPSData;
import sibtra.gps.Trayectoria;
import sibtra.lms.BarridoAngular;
import sibtra.ui.VentanasMonitoriza;
import sibtra.ui.defs.Motor;
import sibtra.util.LabelDatoFormato;
import sibtra.util.SpinnerDouble;
import sibtra.util.ThreadSupendible;
import sibtra.util.UtilCalculos;

/**
 * @author alberto
 *
 */
public class MotorPerrito extends MotorSincrono implements Motor {

	protected String NOMBRE="Motor Perrito";
	protected String DESCRIPCION="Sigue al objeto más cercano";
	
	double distMax=0.3;
	double velAcercamiento=2.0;
	private BarridoAngular ba=null;
	private double[] angDistRF={0,80};
	
	public boolean setVentanaMonitoriza(VentanasMonitoriza ventMonito) {
		if(!super.setVentanaMonitoriza(ventMonito))
			return false;
		//Sustituimos panel Sincrono por panelPerrito
		ventanaMonitoriza.quitaPanel(panel);
		panel=new PanelPerrito();
		ventanaMonitoriza.añadePanel(panel, getNombre(),false,false);
		thCiclico=new ThreadSupendible() {
			private long tSig;

			@Override
			protected void accion() {
				//apuntamos cual debe ser el instante siguiente
		        tSig = System.currentTimeMillis() + periodoMuestreoMili;
				accionPeriodica();
		        //esparmos hasta que haya pasado el tiempo convenido
				while (System.currentTimeMillis() < tSig) {
		            try {
		                Thread.sleep(tSig - System.currentTimeMillis());
		            } catch (Exception e) {}
		        }
			}
		};
		thCiclico.setName(NOMBRE);
		return true;
	}
	
	protected void accionPeriodica() {
		consignaVolante=consignaVolanteRecibida=calculadorDireccion.getConsignaDireccion();
        consignaVolante=UtilCalculos.limita(consignaVolante, -cotaAngulo, cotaAngulo);
        ventanaMonitoriza.conexionCarro.setAnguloVolante(consignaVolante);
        double velocidadActual = ventanaMonitoriza.conexionCarro.getVelocidadMS();
        consignaVelocidad=consignaVelocidadRecibida=calculadorVelocidad.getConsignaVelocidad();
        ventanaMonitoriza.conexionCarro.setConsignaAvanceMS(consignaVelocidad);
        panel.actualizaDatos(MotorPerrito.this);
//		BarridoAngular nuevoBa=ventanaMonitoriza.conexionRF.ultimoBarrido();
//		actualizaModeloCoche();
//		if(nuevoBa!=ba && nuevoBa!=null) {
//			ba=nuevoBa;
//			angDistRF = getAnguloDistObjetivo(ba);
//		}
//		/* Partiendo de la posición y orientación del coche y de la distancia a la que se
//		 * encuentra el objetivo y a que ángulo, se calculan las coordenadas locales
//		 * del objetivo*/
//		double[] coorObjetivo = calculaCoorLocalObjetivo(modCoche.getX(),modCoche.getY()
//				,angDistRF[0]+modCoche.getTita(),angDistRF[1]+10);
//		double[] coorCoche = {modCoche.getX(),modCoche.getY()};
//		/* Calculamos la ruta hasta el objetivo*/
//		Trayectoria Tr = new Trayectoria(coorObjetivo,coorCoche,distMax,velAcercamiento);
//		ventanaMonitoriza.setNuevaTrayectoria(Tr);

		//super.accionPeriodica();

	}
    
    /**
	 * Calcula las coordenadas locales del objetivo que se desea perseguir 
	 * @param coorXCoche coordenada local x del vehículo
	 * @param coorYCoche coordenada local y del vehículo
	 * @param anguloRF ángulo con en el que se encuentra el objetivo con respecto al coche 
	 * (en radianes). Los 0º corresponden a la perpendicular de la dirección que lleva el
	 * coche hacia la izquierda
	 * @param distRF distancia en metros hasta el objetivo
	 * @return Coordenadas locales del objetivo
	 */
	public double[] calculaCoorLocalObjetivo(double coorXCoche,double coorYCoche,double anguloRF,double distRF){
		double [] coorObjetivo = new double[2];
		// Para ángulos del rf < de pi/2 el coseno es positivo, por lo tanto el objetivo 
		// está a la derecha según el sistema de ref del coche. Pero para el rf ángulos 
		// < de pi/2 es estar a la izquierda. No es problema, se le cambia el signo.
		coorObjetivo[0] = coorXCoche + Math.cos(anguloRF-Math.PI/2)*distRF;
		coorObjetivo[1] = coorYCoche + Math.sin(anguloRF-Math.PI/2)*distRF;
		return coorObjetivo;
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

	public double getDistMax() {
		return distMax;
	}

	public void setDistMax(double distMax) {
		this.distMax = distMax;
	}

	public double getVelAcercamiento() {
		return velAcercamiento;
	}

	public void setVelAcercamiento(double velAcercamiento) {
		this.velAcercamiento = velAcercamiento;
	}

	public double getDistanciaRF() {
		return angDistRF[1];
	}
	public double getAnguloRFGrados() {
		return Math.toDegrees(angDistRF[0]);
	}

	protected class PanelPerrito extends PanelSincrono {
		
		public PanelPerrito() {
			super();
			añadeAPanel(new SpinnerDouble(MotorPerrito.this,"setDistMax",0.1,3,0.1), "Dist Ptos");
			añadeAPanel(new SpinnerDouble(MotorPerrito.this,"setVelAcercamiento",1,6,0.2), "Vel acerca");
			añadeAPanel(new LabelDatoFormato(MotorPerrito.class,"getDistanciaRF","%6.2f m"), "Dist RF");
			añadeAPanel(new LabelDatoFormato(MotorPerrito.class,"getAnguloRFGrados","%5.2f º"), "Ang RF");

		}

	}

}
