/**
 * 
 */
package sibtra.log;

import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.util.Calendar;
import java.util.Iterator;
import java.util.Vector;

/**
 * Clase para generar y gestionar los Loggers. Tendrá sólo métodos estáticos. 
 * Cada clase pedirá a esta los loggers que necesite.
 * Esta los activará y guardará los datos cuando se solicite. 
 * @author alberto
 *
 */
public class LoggerFactory {
	
	/** Vector que apuntará a todos los loggers solicitados*/
	private static Vector<Logger> vecLoggers=null;
	
	private static void creaVL() {
		if (vecLoggers==null)
			vecLoggers=new Vector<Logger>();		
	}
	
	public static LoggerDouble nuevoLoggerDouble(Object este,String nombreVariable, int muestrasSg) {
		LoggerDouble ld=new LoggerDouble(este,nombreVariable,muestrasSg);
		creaVL();
		vecLoggers.add(ld); //apuntamos el logger a la lista
		return ld;
	}
	
	public static LoggerDouble nuevoLoggerDouble(Object este,String nombreVariable) {
		LoggerDouble ld=new LoggerDouble(este,nombreVariable);
		creaVL();
		vecLoggers.add(ld); //apuntamos el logger a la lista
		return ld;
	}
	
	
	public static void activaLoggers(int duracionSg) {
		long t0=System.currentTimeMillis();
		for(Iterator<Logger> it=vecLoggers.iterator(); it.hasNext();) {
			it.next().activa(duracionSg,t0);
		}
	}
	
	public static void activaLoggers() {
		long t0=System.currentTimeMillis();
		for(Iterator<Logger> it=vecLoggers.iterator(); it.hasNext();) {
			it.next().activa(5*60,t0);
		}
	}
	
	public static void volcarLoggers() {
		for(Iterator<Logger> it=vecLoggers.iterator(); it.hasNext();) {
			System.out.println(it.next());
		}		
	}
	
	public static void vuelcaLoggersOctave(String nombBase) {
		Calendar ahora=Calendar.getInstance();
		String nombCompleto=nombBase+ahora.get(Calendar.YEAR)
			+ahora.get(Calendar.MONTH)
			+ahora.get(Calendar.DAY_OF_MONTH)
			+ahora.get(Calendar.HOUR_OF_DAY)
			+ahora.get(Calendar.MINUTE)
			+".oct"
			;
		System.out.println("Escribiendo en Fichero "+nombCompleto);
        try {
			PrintWriter os = 
			    new PrintWriter(new FileWriter(nombCompleto));
			for(Iterator<Logger> it=vecLoggers.iterator(); it.hasNext();) {
				os.println(it.next());
			}		
			os.close();
		} catch (IOException e) {
			// TODO Bloque catch generado automáticamente
			e.printStackTrace();
		}

	}

}
