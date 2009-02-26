/**
 * 
 */
package sibtra.log;

import java.util.Vector;

/**
 * @author alberto
 *
 */
public class Logger {
	
	/** Vector que contendrá los instantes de tiempo en que se almacenan los datos */
	Vector<Long> tiempos=null;
	
	/** Nombre de la variable guardada en este looger */
	String nombre;
	
	/** Objeto donde está la variable*/
	Object objeto;
	
	/** Estimación del numero de muestras por segundo que se generarán en esta variable.
	 * por defecto 20 */
	int muestrasSg=20;
	
	/** Para indicar si esta activado o no*/
	boolean activado=false;
	
	/** tiempo en que se activó el logger */
	long t0;

	/**
	 * Asignamos los campos 
	 * @param nombre nombre de la varialbe
	 * @param clase clase a la que pertenece la variable
	 * @param muestrasSg Estimación del numero de muestras por segundo
	 */
	Logger(Object objeto, String nombre, int muestrasSg) {
		this.objeto=objeto;
		this.nombre=nombre;
		this.muestrasSg=muestrasSg;
	}
	
	/**
	 * Asignamos los campos 
	 * @param nombre nombre de la varialbe
	 * @param clase clase a la que pertenece la variable
	 */
	Logger(Object objeto, String nombre) {
		this.objeto=objeto;
		this.nombre=nombre;
	}
	
	/**
	 * Activa el logger y crea nuevo vector de tiempos usando duracion estimada y {@link #muestrasSg}
	 * @param duracionSg duracion estimada del experimento en segundos
	 */
	void activa(int duracionSg, long t0) {
		//TODO para borrar vector ya existente ??
		/*
		if(tiempos!=null)
			tiempos.setSize(0);
			*/
		tiempos=new Vector<Long>(duracionSg*muestrasSg+10);
		activado=true;
		this.t0=t0;
	}
	
	long tiempoMin() {
		if(tiempos!=null && tiempos.size()>=1)
			return tiempos.get(0);
		else
			return System.currentTimeMillis();
	}
}
