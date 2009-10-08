/**
 * 
 */
package sibtra.log;

import java.io.IOException;
import java.util.Locale;
import java.util.Vector;

import sibtra.util.SalvaMATv4;

/**
 * Logger para meter varios doubles en cada instante.
 * @author alberto
 */
public class LoggerArrayInts extends Logger {
	
	private Vector<int[]> datos=null;
	
	/** Nos vale el constructor de la superclase */
	LoggerArrayInts(Object objeto, String nombre, long t0, int numMuestras) {
		super(objeto, nombre, t0, numMuestras);
	}

	/** Nos vale el constructor de la superclase */
	LoggerArrayInts(Object objeto, String nombre, long t0) {
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
			datos=new Vector<int[]>(minCapacity);
		else
			datos.ensureCapacity(datos.size()+minCapacity);
	}
	
	/** Este logger no puede tener el add()  sin datos */
	public void add() {
		throw new IllegalArgumentException("A este logger es necesario añadir algún dato");
	}

	/** Añade nuevo array (haciendo copia) y apunta instante de tiempo */
	public void add(int[] dato) {
		super.add();
		if(!activado)
			return;
		//copiamos Datos
		int[] nD=new int[dato.length];
		System.arraycopy(dato, 0, nD, 0, dato.length);
		datos.add(nD);
	}

	/** Crea nuevo array con los datos pasados, lo añade y apunta instante de tiempo */
	public void add(int d1, int d2) {
		super.add();
		if(!activado)
			return;
		//copiamos Datos
		int[] nD={d1,d2};
		datos.add(nD);
	}

	/** Crea nuevo array con los datos pasados, lo añade y apunta instante de tiempo */
	public void add(int d1, int d2, int d3) {
		super.add();
		if(!activado)
			return;
		//copiamos Datos
		int[] nD={d1,d2,d3};
		datos.add(nD);
	}

	/** Crea nuevo array con los datos pasados, lo añade y apunta instante de tiempo */
	public void add(int d1, int d2, int d3, int d4) {
		super.add();
		if(!activado)
			return;
		//copiamos Datos
		int[] nD={d1,d2,d3,d4};
		datos.add(nD);
	}

	/** Crea nuevo array con los datos pasados, lo añade y apunta instante de tiempo */
	public void add(int d1, int d2, int d3,int d4, int d5) {
		super.add();
		if(!activado)
			return;
		//copiamos Datos
		int[] nD={d1,d2,d3,d4,d5};
		datos.add(nD);
	}

	/** Crea nuevo array con los datos pasados, lo añade y apunta instante de tiempo */
	public void add(int d1, int d2, int d3,int d4, int d5,int d6) {
		super.add();
		if(!activado)
			return;
		//copiamos Datos
		int[] nD={d1,d2,d3,d4,d5,d6};
		datos.add(nD);
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
		int nFil=datos.size();
		int nCol=(nFil>0?datos.get(0).length:0);
		st+="# name: "+nombre+"\n# type: matrix\n# rows: "+nFil
		+"\n# columns: "+nCol+"\n";
		if(nCol>0)
			for(int i=0; i<nFil; i++) {
				st+=String.format((Locale)null,"%f", datos.get(i)[0]);
				for(int j=0; j<nCol; j++)
					st+=(j<datos.get(i).length
							?String.format((Locale)null,"\t%d", datos.get(i)[j])
							:"\tNaN");
				st+="\n";
			}
		return st;
	}
	
	/** Vuelca datos y tiempos a fichero MATv4 */
	void vuelcaMATv4(SalvaMATv4 smv4) throws IOException {
		super.vuelcaMATv4(smv4);
		smv4.matrizIntegers(datos, nombre);
	}
	

}
