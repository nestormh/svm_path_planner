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

import sibtra.predictivo.ControlPredictivo;
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
	private static Vector<Logger> vecLoggers=null;
	
	/** Instante de tiempo que representa nuestro tiempo 0*/
	private static long t0=0;
	
	private static void iniciaEstaticos() {
		if (vecLoggers==null)
			vecLoggers=new Vector<Logger>();
		if (t0==0)
			t0=System.currentTimeMillis();
	}
	
	public static LoggerDouble nuevoLoggerDouble(Object este,String nombreVariable, int muestrasSg) {
		iniciaEstaticos();
		LoggerDouble ld=new LoggerDouble(este,nombreVariable,t0,muestrasSg);
		vecLoggers.add(ld); //apuntamos el logger a la lista
		return ld;
	}
	
	public static LoggerDouble nuevoLoggerDouble(Object este,String nombreVariable) {
		iniciaEstaticos();
		LoggerDouble ld=new LoggerDouble(este,nombreVariable,t0);
		vecLoggers.add(ld); //apuntamos el logger a la lista
		return ld;
	}
	

	public static Logger nuevoLoggerTiempo(Object este,String nombreVariable) {
		iniciaEstaticos();
		Logger ld=new Logger(este,nombreVariable,t0);
		vecLoggers.add(ld); //apuntamos el logger a la lista
		return ld;
	}
	
	
	public static void activaLoggers(int duracionSg) {
		for(Iterator<Logger> it=vecLoggers.iterator(); it.hasNext();) {
			it.next().activa(duracionSg);
		}
	}
	
	public static void activaLoggers() {
		for(Iterator<Logger> it=vecLoggers.iterator(); it.hasNext();) {
			it.next().activa(5*60);
		}
	}
	
	public static void volcarLoggers() {
		for(Iterator<Logger> it=vecLoggers.iterator(); it.hasNext();) {
			System.out.println(it.next());
		}		
	}
	
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
