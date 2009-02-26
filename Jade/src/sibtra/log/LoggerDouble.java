/**
 * 
 */
package sibtra.log;

import java.util.Locale;
import java.util.Vector;

/**
 * @author alberto
 *
 */
public class LoggerDouble extends Logger {
	
	private Vector<Double> datos=null;
	
	/** Nos vale el constructor de la superclase */
	LoggerDouble(Object objeto, String nombre, int numMuestras) {
		super(objeto, nombre, numMuestras);
	}

	/** Nos vale el constructor de la superclase */
	LoggerDouble(Object objeto, String nombre) {
		super(objeto, nombre);
	}
	
	/**
	 * Activa el logger y crea nuevo vector de tiempos usando duracion estimada y {@link #muestrasSg}
	 * @param duracionSg duracion estimada del experimento en segundos
	 */
	void activa(int duracionSg,long t0) {
		super.activa(duracionSg,t0);
		//TODO para borrar vector ya existente ??
		/*
		if(tiempos!=null)
			tiempos.setSize(0);
			*/
		datos=new Vector<Double>(duracionSg*muestrasSg+10);
	}
	
	public void add(double dato) {
		if(!activado)
			return;
		tiempos.add(System.currentTimeMillis());
		datos.add(dato);
	}

	
	public String toString() {
		String st="# name: "+nombre+"\n# type: matrix\n# rows: "+datos.size()+"\n# columns: 2";
		for(int i=0; i<datos.size(); i++) {
			st+=String.format((Locale)null,"\n%d\t%f", tiempos.get(i)-t0,datos.get(i));
		}
		return st;
	}
}
