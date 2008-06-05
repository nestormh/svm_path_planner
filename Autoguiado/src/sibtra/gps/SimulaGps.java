
package sibtra.gps;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.UTFDataFormatException;

import java.util.Vector;

/**
 * Clase para usar el gps directamente o simular su funcionamiento.
 * @author neztol
 */
public class SimulaGps implements Runnable {

    
    private GPSConnection gps = null;
	private Ruta rutaEspacial=null;
	private Ruta rutaTemporal=null;

    /**
     * Constructor para simulación que usa datos de los dos ficheros pasados. 
     * @param ruta nombre del fichero con la ruta
     * @param nmea nombre del fichero con los comandos NMEA
     */
    public SimulaGps(boolean Simular, String fichRuta) {
        gps = new GPSConnection();
        if (Simular && fichRuta != null)
    		try {
    			File file = new File(fichRuta);
    			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file));
    			rutaEspacial=(Ruta)ois.readObject();
    			rutaTemporal=(Ruta)ois.readObject();
    			ois.close();
                Thread hilo = new Thread(this);
                hilo.start(); 
    		} catch (IOException ioe) {
    			System.err.println("Error al abrir el fichero " + fichRuta);
    			System.err.println(ioe.getMessage());
    		} catch (ClassNotFoundException cnfe) {
    			System.err.println("Objeto leído inválido: " + cnfe.getMessage());            
    		}     
    }
    
    /**
     * Constructor para acceder al GPS real a través del puerto
     * @param puerto nombre del puerto serial
     */
    public SimulaGps(String puerto) {
    	try {
    		gps = new GPSConnection(puerto);
    	} catch (SerialConnectionException e) {
    		gps=null;
    	}
    }


    /**
     * Para simular los eventos de recepción de datos por la serial.
     */
    public void run() {
    	while (true) {
    		//empezamos con la ruta
    		long tAct=rutaTemporal.getPunto(0).getSysTime();
    		for(int i=0; i<rutaTemporal.getNumPuntos(); i++) {
    			long tNuevo=rutaTemporal.getPunto(i).getSysTime();

    			try {
    				Thread.sleep(tNuevo-tAct);
    			} catch (Exception e) {}
    			tAct=tNuevo;

    			String cadena = rutaTemporal.getPunto(i).getCadenaNMEA();

    			try {                
    				gps.actualizaNuevaCadena(cadena);                
    			} catch(Exception e) {
    				System.err.println("Error al procesar la cadena " + cadena);
    				System.err.println("\t" + e.getMessage());
    			}

    		}
    	}
    }

    /**
     * @return devuelve {@link #gps}
     */
    public GPSConnection getGps() {
        return gps;
    }
    
    /**
     * @return array con las coordenadas de cada uno de los puntos de la ruta espacial 
     * con  respecto a sistemas de coordenadas local
     */
    public double[][] getRuta() {
        Ruta rutaGPS = gps.getRutaEspacial();
        
        double ruta[][] = new double[rutaGPS.getNumPuntos()][];
        
        for (int i = 0; i < rutaGPS.getNumPuntos(); i++) {
            GPSData data = rutaGPS.getPunto(i);
            ruta[i] = new double[] { data.getXLocal(), data.getYLocal(), data.getZLocal() };
        }
        
        return ruta;
    }
    
    /**
     * @return array con los angulos de cada uno de los puntos de la ruta espacial
     */
    public double[] getAngulos() {
        Ruta rutaGPS = gps.getRutaEspacial();
        
        double angulos[] = new double[rutaGPS.getNumPuntos()];
        
        for (int i = 0; i < rutaGPS.getNumPuntos(); i++) {
            angulos[i] = rutaGPS.getPunto(i).getAngulo();
        }
        
        return angulos;
    }
    
    /**
     * @return array con las velocidades de cada uno de los puntos de la ruta espacial.
     */
    public double[] getVelocidades() {
        Ruta rutaGPS = gps.getRutaEspacial();
        
        double velocidades[] = new double[rutaGPS.getNumPuntos()];
        
        for (int i = 0; i < rutaGPS.getNumPuntos(); i++) {
            velocidades[i] = rutaGPS.getPunto(i).getVelocidad();
        }
        
        return velocidades;
    }      
    
    /** @return coordenadas locales de punto actual.  */
    public double[] getXYZ() {
        GPSData data = gps.getPuntoActualEspacial();
        
        return (new double[] { data.getXLocal(), data.getYLocal(), data.getZLocal() });
    }
    
    /** @return angulo del punto actual   */
    public double getAngulo() {
        return gps.getPuntoActualEspacial().getAngulo();
    }
    
    /** @return velocidad del punto actual */
    public double getVelocidad() {
        return gps.getPuntoActualEspacial().getVelocidad();
    }

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
//        SimulaGps sGps = new SimulaGps("nov19g.dat", "nov19.gps");
//        GPSConnection gps = sGps.getGps();
//        /*gps.startRuta();
//        try {
//            Thread.sleep(120000);
//        } catch (Exception e) {}
//        gps.stopRuta();
//        gps.saveRuta("nov19g.dat");
//        gps.loadRuta("nov19g.dat");        
//        for (int i = 0; i < gps.getRutaTemporal().size(); i++) {
//            System.out.println("**********************************************");
//            System.out.println((GPSData)gps.getRutaTemporal().elementAt(i));
//            System.out.println("==============================================");
//            System.out.println((GPSData)gps.getBufferRutaTemporal().elementAt(i));
//            System.out.println("**********************************************");
//        }*/
//        /*try {
//            Thread.sleep(10000);
//        } catch (Exception e) {}
//        System.out.println(gps.getPuntoActualEspacial());
//        Vector retorno = gps.getLastPuntosEspacial(10000);
//        System.out.println(retorno.size());
//        for (int i = 0; i < retorno.size(); i++) {
//            System.out.println((GPSData)retorno.elementAt(i));
//        }*/
    }
}

