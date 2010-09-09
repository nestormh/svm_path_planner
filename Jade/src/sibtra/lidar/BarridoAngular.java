package sibtra.lidar;

import java.awt.geom.Point2D;

public abstract class BarridoAngular {

	/**
	 * Hora del ordenador cuando se obtuvo el dato (en milisegundos).
	 * Como lo devuelve la llamada <code>System.currentTimeMillis()</code>.
	 */
	public long sysTime=System.currentTimeMillis();

	/**
	 * @return the sysTime
	 */
	public long getSysTime() {
		return sysTime;
	}

	/**
	 * @return Número de datos del barrido.
	 */
	public abstract int numDatos();

	/** @return Angulo del dato i-ésimo EN RADIANES */
	public abstract double getAngulo(int i);

	/** @return Distancia correspondiente al dato i-ésimo */
	public abstract double getDistancia(int i);

	/** @return {@link Point2D} correspondiente al dato i-ésimo */ 
	public Point2D.Double getPunto(int i){
		if (i<0 || i>=numDatos()) return null;
		double ang=getAngulo(i);
		double dis=getDistancia(i);
		return new Point2D.Double(dis*Math.cos(ang),dis*Math.sin(ang));
	}
	

	/** @return la distancia máxima posible */
	public abstract double getDistanciaMaxima();
	
	public String toString() {
		String ret="NumDat="+numDatos();
		for(int i=0; i<numDatos(); i++) {
			ret+=String.format("[%f,%f]", getAngulo(i),getDistancia(i));
		}
		return ret;
	}


}