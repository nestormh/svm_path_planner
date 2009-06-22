/**
 * 
 */
package sibtra.rfycarro;

import sibtra.lms.BarridoAngular;
import sibtra.util.Parametros;
import sibtra.util.UtilCalculos;

/**
 * Clase para gestionar la detección de obstáculos en el camino futuro del carro.
 * Sabiendo orientación de las ruedas, velocidad y barrido sabrá tiempo hasta próximo obstáculo
 * 
 * @author alberto
 *
 */
public class FuturoObstaculo {
	
	/** Águlo de la dirección en la última invocación de {@link #tiempoAObstaculo(double, double, BarridoAngular)} */
	double alfaAct;
	/** Velocidad en la última invocación de {@link #tiempoAObstaculo(double, double, BarridoAngular)} */
	double velMSAct;
	/** Radio de curvatura calculado en la última invocación de {@link #tiempoAObstaculo(double, double, BarridoAngular)} */
	double radioCur;
	
	/** Velocidad angular calculada en la última invocación de {@link #tiempoAObstaculo(double, double, BarridoAngular)} */
	double velAngular;
	/** Radio interior calculado en la última invocación de {@link #tiempoAObstaculo(double, double, BarridoAngular)} */
	double radioInterior;
	/** Signo de {@link #alfaAct} en la última invocación de {@link #tiempoAObstaculo(double, double, BarridoAngular)} */
	double signoAlfa;
	/** Radio de exterior calculado en la última invocación de {@link #tiempoAObstaculo(double, double, BarridoAngular)} */
	double radioExterior;
	/** Ángulo del punto interior calculado en la última invocación de {@link #tiempoAObstaculo(double, double, BarridoAngular)} */
	double anguloInterior;
	/** Índice del barrido del punto de tiempo mínimo calculado en la última invocación de {@link #tiempoAObstaculo(double, double, BarridoAngular)} */
	int indMin;
	/** Barrido que se usó en la última invocación de {@link #tiempoAObstaculo(double, double, BarridoAngular)} */
	BarridoAngular bAct;
	double distanciaObs;

	public FuturoObstaculo() {
		
	}
	
	/**
	 * Calcula tiempo a siguiente obstáculo. 
	 * @param alfa Águlo de la dirección
	 * @param velMS velocidad en metros por segundo
	 * @param barr barrido obtenido del RF
	 * @return los segundos hasta siguiente obstáculo. Será infinito si no se encuentra 
	 */
	public double distanciaAObstaculo(double alfa, BarridoAngular barr) {
		//copiamos los valores pasados
		bAct=barr;
		alfaAct=alfa;
		if(Math.abs(alfa)>Math.toRadians(0.1)) {
			//la orientación no es 0: Caso general
			signoAlfa=Math.signum(alfa);
			radioCur=Parametros.batalla*signoAlfa/Math.tan(alfa);
			
			radioInterior=radioCur-Parametros.medioAnchoCarro;
			double esquinaExterior[]={Parametros.distRFEje , -(radioCur+Parametros.medioAnchoCarro)*signoAlfa};
			radioExterior=UtilCalculos.largoVector(esquinaExterior);
			
			double esquinaInterior[]={Parametros.distRFEje , -(radioCur-Parametros.medioAnchoCarro)*signoAlfa};
			anguloInterior=Math.atan2(esquinaInterior[1],esquinaInterior[0])*signoAlfa;
			
			//recorremos todos los puntos del barrido para ver si están dentro del segmento angular
			double angMin=Double.POSITIVE_INFINITY;
			indMin=-1;
			for(int iPtoA=0; iPtoA<barr.numDatos();iPtoA++) {
				double posPtoA[]=barrAPos(iPtoA);
				double distPtoA=UtilCalculos.largoVector(posPtoA);
				if(distPtoA>radioExterior || distPtoA<radioInterior)
					//está fuera del segmento conflictivo
					continue;
				//el punto está dentro. Hay que saber con qué ángulo para calcular tiempo.
				double angPtoAct=Math.atan2(posPtoA[1]*signoAlfa, posPtoA[0]);
				if(angPtoAct<angMin) { //tenemos nuevo minimo
					angMin=angPtoAct;
					indMin=iPtoA;
				}
			}
			if(indMin<0) {
				//no hay ningún punto en zona peligrosa
				//damos la distancia de lo que vemos
				angMin=-anguloInterior;
//				return Double.POSITIVE_INFINITY;
			}
			//Tenemos un punto, calculamos tiempo al mismo
			double angAvance=angMin-anguloInterior;
			return distanciaObs=(angAvance*radioCur);
		} else { 
			//la orientación en practicamente 0
			radioCur=Double.POSITIVE_INFINITY;
			double distMin=Double.POSITIVE_INFINITY;
			indMin=-1;
			for(int iPtoA=0; iPtoA<barr.numDatos();iPtoA++) {
				double posPtoA[]={barr.getDistancia(iPtoA)*Math.sin(barr.getAngulo(iPtoA))
						, -barr.getDistancia(iPtoA)*Math.cos(barr.getAngulo(iPtoA))};
				if(Math.abs(posPtoA[1])>Parametros.medioAnchoCarro)
					continue;
				if(posPtoA[0]<distMin) {
					distMin=posPtoA[0];
					indMin=iPtoA;
				}
			}
			if(indMin<0) {
				System.err.println("ABSURDO: No hay minimo con alfa=0");
				return distanciaObs=Double.NaN;
			}
			return distanciaObs=distMin;
		}
	}

	/** Coordenadas del punto i-ésimo del barrido respecto a R */
	private double[] barrAPos(int i) {
		double angulo=bAct.getAngulo(i);
		double distancia=bAct.getDistancia(i);
		double pos[]={distancia*Math.sin(angulo)+Parametros.batalla
				, -distancia*Math.cos(angulo)-radioCur*signoAlfa};
		return pos;
	}
	
	public double tiempoAObstaculo(double velMS) {
		velAngular=velMS/radioCur;
		return distanciaObs/velMS;
	}


}
