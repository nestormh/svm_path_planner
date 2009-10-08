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
public class LoggerInt extends Logger {
	
	private Vector<Integer> datos=null;
	
	/** Nos vale el constructor de la superclase */
	LoggerInt(Object objeto, String nombre, long t0, int numMuestras) {
		super(objeto, nombre, t0, numMuestras);
	}

	/** Nos vale el constructor de la superclase */
	LoggerInt(Object objeto, String nombre, long t0) {
		super(objeto, nombre, t0);
	}
	
	/**
	 * Activa el logger y prepara espacio registro para duracion estimada y {@link #muestrasSg}
	 * Si no existe vector lo crea. 
	 * Si ya existe reserva espacio para datos que contiene más los esperados.
	 * @param duracionSg duracion estimada del experimento en segundos
	 */
	void activa(int duracionSg) {
		super.activa(duracionSg);
		int minCapacity=duracionSg*muestrasSg+10;
		if(datos==null)
			datos=new Vector<Integer>(minCapacity);
		else
			datos.ensureCapacity(datos.size()+minCapacity);
	}
	
	/** Este logger no puede tener el add()  sin datos */
	public void add() {
		throw new IllegalArgumentException("A este logger es necesario añadir algún dato");
	}

	/** Añade nuevo dato apuntando instante de tiempo */
	public void add(int dato) {
		super.add();
		if(!activado)
			return;
		datos.add(dato);
	}

	/** Borra los datos almacenados */
	void clear() {
		super.clear();
		if(datos!=null)
			datos.setSize(0);
	}

	/**	Devuelve String en formato para fichero octave de texto	 */	
	public String toString() {
		if(datos==null) return null;
		String st=super.toString();  //volcado matriz de tiempos
		st+="# name: "+nombre+"\n# type: matrix\n# rows: "+datos.size()+"\n# columns: 1\n";
		for(int i=0; i<datos.size(); i++) {
			st+=String.format((Locale)null,"%d\n", datos.get(i));
		}
		return st;
	}
	
	/** Vuelca datos y tiempos a fichero MATv4 */
	void vuelcaMATv4(SalvaMATv4 smv4) throws IOException {
		super.vuelcaMATv4(smv4);
		smv4.vectorIntegers(datos, nombre);
	}


}
