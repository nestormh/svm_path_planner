/**
 * 
 */
package sibtra.log;

import java.io.IOException;
import java.util.Locale;
import java.util.Vector;

import sibtra.util.SalvaMATv4;

/**
 * @author alberto
 *
 */
public class LoggerDouble extends Logger {
	
	private Vector<Double> datos=null;
	
	/** Nos vale el constructor de la superclase */
	LoggerDouble(Object objeto, String nombre, long t0, int numMuestras) {
		super(objeto, nombre, t0, numMuestras);
	}

	/** Nos vale el constructor de la superclase */
	LoggerDouble(Object objeto, String nombre, long t0) {
		super(objeto, nombre, t0);
	}
	
	/**
	 * Activa el logger y crea nuevo vector de tiempos usando duracion estimada y {@link #muestrasSg}
	 * @param duracionSg duracion estimada del experimento en segundos
	 */
	void activa(int duracionSg) {
		super.activa(duracionSg);
		//TODO para borrar vector ya existente ??
		/*
		if(tiempos!=null)
			tiempos.setSize(0);
			*/
		if(datos==null)
			datos=new Vector<Double>(duracionSg*muestrasSg+10);
	}
	
	/** Añade nuevo dato apuntando instante de tiempo */
	public void add(double dato) {
		super.add();
		if(!activado)
			return;
		datos.add(dato);
	}

	/** Borra los datos almacenados */
	void clear() {
		super.clear();
		datos=null;
	}

	/**	Devuelve String en formato para fichero octave de texto	 */	
	public String toString() {
		if(datos==null) return null;
		String st=super.toString();  //volcado matriz de tiempos
		st+="# name: "+nombre+"\n# type: matrix\n# rows: "+datos.size()+"\n# columns: 1\n";
		for(int i=0; i<datos.size(); i++) {
			st+=String.format((Locale)null,"%f\n", datos.get(i));
		}
		return st;
	}
	
	/** Vuelca datos y tiempos a fichero MATv4 */
	void vuelcaMATv4(SalvaMATv4 smv4) throws IOException {
		super.vuelcaMATv4(smv4);
		smv4.vectorDoubles(datos, nombre);
	}

}
