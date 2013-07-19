/**
 * 
 */
package sibtra.lms;

import sibtra.lidar.BarridoAngular;

/**
 * @author alberto
 *
 */
public class BarridoAngular111 extends BarridoAngular {
	
	double factorDistancia = 1/1000.0;

	double anguloInicial=-45.0;
	
	double pasoAngulo=0.5;
	
	int[] datos=null;
	
	double distanciaMaxima=Double.NaN;

	/**
	 * 
	 */
	public BarridoAngular111() {
		// TODO Auto-generated constructor stub
	}
	
	public BarridoAngular111(int[] datosBarrido) {
		datos=datosBarrido;
	}
	

	public BarridoAngular111(int[] datosBarrido, int angIni, int pasAng, double factor) {
		datos=datosBarrido;
		anguloInicial=Math.toRadians(((double)angIni)/10000);
		pasoAngulo=Math.toRadians(((double)pasAng)/10000);
		factorDistancia=factor/1000.0;
	}

	public BarridoAngular111(int[] datosBarrido, double angIni, double pasAng, double factor) {
		datos=datosBarrido;
		anguloInicial=angIni;
		pasoAngulo=pasAng;
		factorDistancia=factor;
	}

	/* (non-Javadoc)
	 * @see sibtra.lidar.BarridoAngular#getAngulo(int)
	 */
	@Override
	public double getAngulo(int i) {
		return (i-1)*pasoAngulo+anguloInicial;
	}

	/* (non-Javadoc)
	 * @see sibtra.lidar.BarridoAngular#getDistancia(int)
	 */
	@Override
	public double getDistancia(int i) {
		return datos[i]*factorDistancia;
	}

	/* (non-Javadoc)
	 * @see sibtra.lidar.BarridoAngular#getDistanciaMaxima()
	 */
	@Override
	public double getDistanciaMaxima() {
		if(Double.isNaN(distanciaMaxima)) {
			distanciaMaxima=getDistancia(0);
			for(int i=1; i<datos.length; i++)
				if (distanciaMaxima<getDistancia(i))
					distanciaMaxima=getDistancia(i);
		}
		return distanciaMaxima;
	}

	/* (non-Javadoc)
	 * @see sibtra.lidar.BarridoAngular#numDatos()
	 */
	@Override
	public int numDatos() {
		// TODO Auto-generated method stub
		return datos.length;
	}

}
