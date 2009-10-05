/**
 * 
 */
package sibtra.log;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Iterator;
import java.util.Vector;

import sibtra.gps.GPSConnectionTriumph;
import sibtra.util.SalvaMATv4;

/**
 * Clase para generar y gestionar los Loggers. Tendrá sólo métodos estáticos. 
 * Cada clase pedirá a esta los loggers que necesite.
 * Esta los activará y guardará los datos cuando se solicite. 
 * @author alberto
 *
 */
public class LoggerFactory {
	
	/** Vector que apuntará a todos los loggers solicitados*/
	static Vector<Logger> vecLoggers=null;
	
	/** Instante de tiempo que representa nuestro tiempo 0*/
	static long t0=0;
	
	/** Para iniciar los campos estáticos si aún no lo están*/
	private static void iniciaEstaticos() {
		if (vecLoggers==null)
			vecLoggers=new Vector<Logger>();
		if (t0==0)
			t0=System.currentTimeMillis();
	}
	
	/** Devuelve un {@link LoggerDouble} y lo apunta
	 * @param este objeto de referencia del logger
	 * @param nombreVariable nombre del logger
	 * @param muestrasSg numero de muestas esperadas
	 * @return logger creado
	 */
	public static LoggerDouble nuevoLoggerDouble(Object este,String nombreVariable, int muestrasSg) {
		iniciaEstaticos();
		LoggerDouble ld=new LoggerDouble(este,nombreVariable,t0,muestrasSg);
		vecLoggers.add(ld); //apuntamos el logger a la lista
		return ld;
	}
	
	/** Devuelve un {@link LoggerDouble} y lo apunta
	 * @param este objeto de referencia del logger
	 * @param nombreVariable nombre del logger
	 * @return logger creado
	 */
	public static LoggerDouble nuevoLoggerDouble(Object este,String nombreVariable) {
		iniciaEstaticos();
		LoggerDouble ld=new LoggerDouble(este,nombreVariable,t0);
		vecLoggers.add(ld); //apuntamos el logger a la lista
		return ld;
	}
	

	/** Devuelve un {@link Logger} y lo apunta
	 * @param este objeto de referencia del logger
	 * @param nombreVariable nombre del logger
	 * @return logger creado
	 */
	public static Logger nuevoLoggerTiempo(Object este,String nombreVariable) {
		iniciaEstaticos();
		Logger ld=new Logger(este,nombreVariable,t0);
		vecLoggers.add(ld); //apuntamos el logger a la lista
		return ld;
	}
	
	/** Devuelve un {@link Logger} y lo apunta
	 * @param este objeto de referencia del logger
	 * @param nombreVariable nombre del logger
	 * @param muestrasSg numero de muestas esperadas
	 * @return logger creado
	 */
	public static Logger nuevoLoggerTiempo(Object este,String nombreVariable, int muestrasSg) {
		iniciaEstaticos();
		Logger ld=new Logger(este,nombreVariable,t0,muestrasSg);
		vecLoggers.add(ld); //apuntamos el logger a la lista
		return ld;
	}
	
	/** Devuelve un {@link LoggerArrayDoubles} y lo apunta
	 * @param este objeto de referencia del logger
	 * @param nombreVariable nombre del logger
	 * @return logger creado
	 */
	public static LoggerArrayDoubles nuevoLoggerArrayDoubles(Object este,String nombreVariable) {
		iniciaEstaticos();
		LoggerArrayDoubles lad=new LoggerArrayDoubles(este,nombreVariable,t0);
		vecLoggers.add(lad); //apuntamos el logger a la lista
		return lad;
	}

	/** Devuelve un {@link LoggerArrayDoubles} y lo apunta
	 * @param este objeto de referencia del logger
	 * @param nombreVariable nombre del logger
	 * @param muestrasSg numero de muestas esperadas
	 * @return logger creado
	 */
	public static LoggerArrayDoubles nuevoLoggerArrayDoubles(Object este,String nombreVariable, int muestrasSg) {
		iniciaEstaticos();
		LoggerArrayDoubles lad=new LoggerArrayDoubles(este,nombreVariable,t0,muestrasSg);
		vecLoggers.add(lad); //apuntamos el logger a la lista
		return lad;
	}

	/** Devuelve un {@link LoggerInt} y lo apunta
	 * @param este objeto de referencia del logger
	 * @param nombreVariable nombre del logger
	 * @param muestrasSg numero de muestas esperadas
	 * @return logger creado
	 */
	public static LoggerInt nuevoLoggerInt(Object este,String nombreVariable, int muestrasSg) {
		iniciaEstaticos();
		LoggerInt la=new LoggerInt(este,nombreVariable,t0,muestrasSg);
		vecLoggers.add(la); //apuntamos el logger a la lista
		return la;
	}

	/** Devuelve un {@link LoggerInt} y lo apunta
	 * @param este objeto de referencia del logger
	 * @param nombreVariable nombre del logger
	 * @return logger creado
	 */
	public static LoggerInt nuevoLoggerInt(Object este,String nombreVariable) {
		iniciaEstaticos();
		LoggerInt la=new LoggerInt(este,nombreVariable,t0);
		vecLoggers.add(la); //apuntamos el logger a la lista
		return la;
	}

	/** Devuelve un {@link LoggerArrayInts} y lo apunta
	 * @param este objeto de referencia del logger
	 * @param nombreVariable nombre del logger
	 * @param muestrasSg numero de muestas esperadas
	 * @return logger creado
	 */
	public static LoggerArrayInts nuevoLoggerArrayInts(Object este,String nombreVariable, int muestrasSg) {
		iniciaEstaticos();
		LoggerArrayInts lad=new LoggerArrayInts(este,nombreVariable,t0,muestrasSg);
		vecLoggers.add(lad); //apuntamos el logger a la lista
		return lad;
	}
	
	/** Devuelve un {@link LoggerArrayInts} y lo apunta
	 * @param este objeto de referencia del logger
	 * @param nombreVariable nombre del logger
	 * @return logger creado
	 */
	public static LoggerArrayInts nuevoLoggerArrayInts(Object este,String nombreVariable) {
		iniciaEstaticos();
		LoggerArrayInts lad=new LoggerArrayInts(este,nombreVariable,t0);
		vecLoggers.add(lad); //apuntamos el logger a la lista
		return lad;
	}
	
	/** Devuelve un {@link LoggerLong} y lo apunta
	 * @param este objeto de referencia del logger
	 * @param nombreVariable nombre del logger
	 * @param muestrasSg numero de muestas esperadas
	 * @return logger creado
	 */
	public static LoggerLong nuevoLoggerLong(Object este,String nombreVariable, int muestrasSg) {
		iniciaEstaticos();
		LoggerLong la=new LoggerLong(este,nombreVariable,t0,muestrasSg);
		vecLoggers.add(la); //apuntamos el logger a la lista
		return la;
	}

	/** Devuelve un {@link LoggerLong} y lo apunta
	 * @param este objeto de referencia del logger
	 * @param nombreVariable nombre del logger
	 * @param muestrasSg numero de muestas esperadas
	 * @return logger creado
	 */
	public static LoggerLong nuevoLoggerLong(Object este,String nombreVariable) {
		iniciaEstaticos();
		LoggerLong la=new LoggerLong(este,nombreVariable,t0);
		vecLoggers.add(la); //apuntamos el logger a la lista
		return la;
	}
	
	/** Elimina de {@link #vecLoggers} el logger pasado */ 
	public static void borraLogger(Logger log) {
		vecLoggers.removeElement(log);
	}

	/** Activa todos los loggers para el tiempo de experimento
	 * @param duracionSg duración estimada del experimento
	 */
	public static void activaLoggers(int duracionSg) {
		for(Iterator<Logger> it=vecLoggers.iterator(); it.hasNext();) {
			it.next().activa(duracionSg);
		}
	}
	
	/** Activa todos los loggers para el tiempo de experimento de 5 minutos */	
	public static void activaLoggers() {
		for(Iterator<Logger> it=vecLoggers.iterator(); it.hasNext();) {
			it.next().activa(5*60);
		}
	}
	
	/** Vuelca los loggers por la salida estandar en formato de matriz de texto Octave */
	public static void volcarLoggers() {
		for(Iterator<Logger> it=vecLoggers.iterator(); it.hasNext();) {
			System.out.println(it.next());
		}		
	}
	
	/** Vuelve los logger a fichero Octave de texto construido con nombre base y momento 
	 * @param nombBase Comienzo del nombre
	 */
	public static void vuelcaLoggersOctave(String nombBase) {
		String nombCompleto=nombBase+new SimpleDateFormat("yyyyMMddHHmm").format(new Date())
			+".oct"
			;
		System.out.println("Escribiendo en Fichero "+nombCompleto);
        try {
			PrintWriter os = 
			    new PrintWriter(new FileWriter(nombCompleto));
			for(Iterator<Logger> it=vecLoggers.iterator(); it.hasNext();) {
				os.print(it.next());
			}		
			os.close();
		} catch (IOException e) {
			// TODO Bloque catch generado automáticamente
			e.printStackTrace();
		}

	}
	
	/** Vuelve los logger a fichero Matlaba v4 construido con nombre base y momento 
	 * @param nombBase Comienzo del nombre
	 */
	public static void vuelcaLoggersMATv4(String nombBase) {
		String nombCompleto=nombBase+new SimpleDateFormat("yyyyMMddHHmm").format(new Date())
			+".mat"
			;
		System.out.println("Escribiendo en Fichero "+nombCompleto);
        try {
        	SalvaMATv4 smv4=new SalvaMATv4(nombCompleto);
			for(Iterator<Logger> it=vecLoggers.iterator(); it.hasNext();) {
				it.next().vuelcaMATv4(smv4);
			}		
			smv4.close();
		} catch (IOException e) {
			// TODO Bloque catch generado automáticamente
			e.printStackTrace();
		}
	}

}
