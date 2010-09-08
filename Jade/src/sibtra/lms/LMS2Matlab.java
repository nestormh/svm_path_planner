/**
 * 
 */
package sibtra.lms;

import sibtra.lms.BarridoAngular221;
import sibtra.lms.LMSException;
import sibtra.lms.ManejaLMS221;
import sibtra.lms.BarridoAngular221.barridoAngularIterator;

/**
 * Clase para acceder a range finder LMS desde Matlab
 * 
 * @author alberto
 *
 */
public class LMS2Matlab {

	protected ManejaLMS221 manLMS;

	/**
	 * Conecta con el LMS y hace la configuración por defecto
	 * @param puerto
	 */
	public LMS2Matlab(String puerto){
		manLMS=new ManejaLMS221(puerto);
	}
	
	/**
	 * Devuleve un barrido de un sólo promedio en el rango máximo.
	 * El rango será de 0º a 180º si la resolución es de 1º o 0.5º.
	 * El rango será de 40º a 140º si la resolución es de 0.25º  
	 * @return el primer componete es el ángulo en radianes y el segundo es la distancia en metros. 
	 * {@code null} si hay algún problema.
	 */
	public double[][] getBarrido(){
		return getBarrido(0, Math.PI, 1);
	}
	
	/**
	 * Devuelve un barrido de un promedio desde 
	 * el angulo {@code angInicial} a {@code angFinal}.
	 * @param angInicial angulo inicial del barrido en radianes
	 * @param angFinal angulo final del barrido en radianes
	 * @return  el primer componete es el ángulo en radianes y el segundo es la distancia en metros
	 * {@code null} si hay algún problema.
	 */
	public double[][] getBarrido(double angInicial, double angFinal){
		return getBarrido(angInicial, angFinal, 1);
	}
	
	/**
	 * Devuelve un barrido de el número de promedios indicados desde 
	 * el angulo {@code angInicial} a {@code angFinal}.
	 * @param angInicial angulo inicial del barrido en radianes
	 * @param angFinal angulo final del barrido en radianes
	 * @param promedios numero de promedios del barrido 
	 * @return  el primer componete es el ángulo en radianes y el segundo es la distancia en metros
	 * {@code null} si hay algún problema.
	 */
	public double[][] getBarrido(double angInicial, double angFinal, double promedios){
		double[][] res;
		try {
			manLMS.pideBarrido((short)Math.toDegrees(angInicial)
					, (short)Math.toDegrees(angFinal)
					, (short)promedios);
			BarridoAngular221 barr=manLMS.recibeBarrido();
			res=new double[barr.numDatos()][2];
			barridoAngularIterator bit=barr.creaIterator();
			for(int i=0; bit.hasNext(); i++) {
				res[i][1]=Math.toRadians(bit.angulo());  //angulo en radianes
				res[i][2]=bit.distancia();  //distancia en metros
			} 
		} catch (LMSException e) {
			res=null;
		}
		return res;
	}
	
	/**
	 * Cambia configuracion de LMS.
	 * @param resolucionAngular puede ser 1º, 0.5º o 0.25º
	 * @param distanciaMaxima puede ser 8 metros, 16 metros, 32 metros, u 80 metros 
	 * @return 0 si todo fue bien, 1 si fallo comunicación, 2 si fallo prametros
	 */
	public double setConfiguracion(double resolucionAngular, double distanciaMaxima ) {
		if(resolucionAngular!=1 && resolucionAngular!=0.5 && resolucionAngular!=0.25)
			return 2.0;
		if (distanciaMaxima!=8 && distanciaMaxima!=16 && distanciaMaxima!=32 && distanciaMaxima!=80)
			return 2.0;
		try {
			if(resolucionAngular==0.25)
				manLMS.setVariante((short)100, (short)25);
			else
				manLMS.setVariante((short)180, (short)(resolucionAngular*100.0));
			manLMS.setDistanciaMaxima((int)distanciaMaxima);
			return 1.0;
		} 
		catch (IllegalArgumentException e) { return 2.0; }
		catch (LMSException e) { return 1.0; }
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		LMS2Matlab l2m=new LMS2Matlab("/dev/ttyUSB0");
		
		l2m.setConfiguracion(0.5, 32);
		
		double[][] barr=l2m.getBarrido();
		for(int i=0; i<barr.length; i++) {
			System.out.print("Fila "+i+": ");
			for(int j=0; j<barr[0].length; j++)
				System.out.print(barr[i][j]);
			System.out.println();
		}
	}

}
