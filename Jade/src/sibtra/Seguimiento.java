package sibtra;

public class Seguimiento {
	/** Distancia máxima entre los puntos de la ruta entre el objetivo y el coche*/
	static double dISTMAX = 0.1;
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
		coorObjetivo[0] = coorXCoche - Math.cos(anguloRF);
		coorObjetivo[1] = coorYCoche + Math.sin(anguloRF);
		return coorObjetivo;
	}
	
	public double[][] calculaRutaObjetivo(double[] coorObjetivo,double coorXCoche,double coorYCoche){
		double dx = coorObjetivo[0] - coorXCoche;
		double dy = coorObjetivo[1] - coorYCoche;
		double distAObjetivo = Math.sqrt(dx*dx +dy*dy);		
		int numPuntos = (int)Math.ceil(distAObjetivo/dISTMAX);
		double[][] rutaObjetivo = new double[2][numPuntos];
		return rutaObjetivo;
	}

}
