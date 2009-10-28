package sibtra;

import sibtra.controlcarro.ControlCarro;
import sibtra.gps.GPSConnectionTriumph;
import sibtra.imu.ConexionSerialIMU;
import sibtra.lms.LMSException;
import sibtra.lms.ManejaLMS;
import sibtra.rfyruta.MiraObstaculo;

public class Seguimiento {
	/** Distancia máxima entre los puntos de la ruta entre el objetivo y el coche*/
	static double dISTMAX = 0.1;
	/** Velocidad máxima a la que el coche se aproximará al objetivo*/
	static double VELACERCAMIENTO = 1.5;
	private ConexionSerialIMU csi;
    private GPSConnectionTriumph gpsCon;
    private ManejaLMS manLMS;
    private MiraObstaculo MiraObs;
	private ControlCarro contCarro;    	
    
    public Seguimiento(String[] args){
    	/* Se realizan las conexiones a los sensores y a la electrónica del coche*/
    	
    	if (args == null || args.length < 4) {
            System.err.println("Son necesarios 4 argumentos con los puertos seriales");
            System.exit(1);
        }

        //conexión de la IMU
        System.out.println("Abrimos conexión IMU");
        csi = new ConexionSerialIMU();
        if (!csi.ConectaPuerto(args[1], 5)) {
            System.err.println("Problema en conexión serial con la IMU");
            System.exit(1);
        }

        //comunicación con GPS
        System.out.println("Abrimos conexión GPS");
        try {
            gpsCon = new GPSConnectionTriumph(args[0]);
			if(gpsCon.esperaCentroBase())
				gpsCon.fijaCentro(gpsCon.posicionDeLaBase());
			gpsCon.comienzaEnvioPeriodico();
        } catch (Exception e) {
            System.err.println("Problema a crear GPSConnection:" + e.getMessage());
            System.exit(1);
        }
        if (gpsCon == null) {
            System.err.println("No se obtuvo GPSConnection");
            System.exit(1);
        }
        gpsCon.setCsIMU(csi);

/*TODO En las siguientes lineas se configura el RF. Configurarlo según el último trabajo
 *     de Alberto. Gracias a las nuevas mejoras Alberto se ha reducido brutalmente el tiempo de 
 *     cómputo y de transmisión*/ 
        
        //Conectamos a RF
        System.out.println("Abrimos conexión LMS");
        try {
            manLMS = new ManejaLMS(args[2]);
            manLMS.setDistanciaMaxima(80);
            manLMS.setResolucionAngular((short)100);
            manLMS.CambiaAModo25();
        } catch (LMSException e) {
            System.err.println("No fue posible conectar o configurar RF");
        }

        //Conectamos Carro
        System.out.println("Abrimos conexión al Carro");
        contCarro = new ControlCarro(args[3]);

        if (contCarro.isOpen() == false) {
            System.err.println("No se obtuvo Conexion al Carro");            
        }
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
		coorObjetivo[0] = coorXCoche - Math.cos(anguloRF);
		coorObjetivo[1] = coorYCoche + Math.sin(anguloRF);
		return coorObjetivo;
	}
	/**
	 * Traza una ruta recta entre el coche y el objetivo
	 * @param coorObjetivo Coordenadas locales del objetivo
	 * @param anguloRF ángulo con en el que se encuentra el objetivo con respecto al coche 
	 * (en radianes). Los 0º corresponden a la perpendicular de la dirección que lleva el
	 * coche hacia la izquierda
	 * @param coorXCoche coordenada local x del vehículo
	 * @param coorYCoche coordenada local y del vehículo
	 * @return Vector de doubles de 4 columnas (x,y,orientación,velocidad) y tantas filas como puntos
	 * tenga la ruta resultante. El número de puntos de la ruta dependerá de la distancia entre
	 * el objetivo y el coche, y de la distancia máxima permitida entre los puntos
	 */

	public double[][] calculaRutaObjetivo(double[] coorObjetivo,double anguloRF,double coorXCoche,double coorYCoche){
		double dx = coorObjetivo[0] - coorXCoche;
		double dy = coorObjetivo[1] - coorYCoche;
		double distAObjetivo = Math.sqrt(dx*dx +dy*dy);		
		int numPuntos = (int)Math.ceil(distAObjetivo/dISTMAX);
		double incX = dx/numPuntos;
		double incY = dy/numPuntos;
		double[][] rutaObjetivo = new double[4][numPuntos];
		for (int i=0;i<=numPuntos;i++){
			rutaObjetivo[0][i] = coorXCoche + incX*i;
			rutaObjetivo[1][i] = coorYCoche + incY*i;
			rutaObjetivo[2][i] = Math.PI - anguloRF;
			rutaObjetivo[3][i] = VELACERCAMIENTO; 
		}
		return rutaObjetivo;
	}
	public static void main(String[] args) {
		
	}

}
