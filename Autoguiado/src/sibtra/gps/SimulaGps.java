/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */



package sibtra.gps;

//~--- JDK imports ------------------------------------------------------------

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.UTFDataFormatException;

import java.util.Vector;

/**
 *
 * @author neztol
 */
public class SimulaGps implements Runnable {

    // Vectores del GPS
    private Vector vCadena           = new Vector();
    private Vector vTipoPaquete      = new Vector();
    private Vector vTiempo           = new Vector();
    private Vector vPdop             = new Vector();
    private Vector vHdop             = new Vector();
    private Vector vVdop             = new Vector();
    private Vector vSpeed            = new Vector();
    private Vector vRms              = new Vector();
    private Vector vOrientacionMayor = new Vector();
    private Vector vLongitud         = new Vector();
    private Vector vLatitud          = new Vector();
    private Vector vHora             = new Vector();
    private Vector vHdgPoloN         = new Vector();
    private Vector vHdgPoloM         = new Vector();
    private Vector vDesvLongitud     = new Vector();
    private Vector vDesvLatitud      = new Vector();
    private Vector vDesvEjeMenor     = new Vector();
    private Vector vDesvEjeMayor     = new Vector();
    private Vector vDesvAltura       = new Vector();
    private Vector vAltura           = new Vector();
    private Vector vX                = new Vector();
    private Vector vY                = new Vector();
    private Vector vZ                = new Vector();
    private Vector vSatelites        = new Vector();
    private Vector vMsl              = new Vector();
    private Vector vLongitudG        = new Vector();
    private Vector vLatitudG         = new Vector();
    private Vector vHGeoide          = new Vector();
    private Vector vAngulo           = new Vector();
    private Vector vAge              = new Vector();
    
    private GPSConnection gps = null;

    public SimulaGps(String ruta, String nmea) {
        gps = new GPSConnection();
        if (ruta != null)
            gps.loadRuta(ruta);
        loadDatos(nmea);
        Thread hilo = new Thread(this);
        hilo.start(); 
    }
    
    public SimulaGps(String puerto) {
        gps = new GPSConnection(puerto);        
    }

    public void loadDatos(String fichero) {
        try {
            File              fich = new File(fichero);
            ObjectInputStream is   = new ObjectInputStream(new FileInputStream(fich));

            while (is.available() != 0) {
                vCadena.add(is.readUTF());                
                vTipoPaquete.add(is.readUTF());                
                vTiempo.add(is.readLong());
                vPdop.add(is.readDouble());
                vHdop.add(is.readDouble());
                vVdop.add(is.readDouble());
                vRms.add(is.readDouble());
                vDesvEjeMayor.add(is.readDouble());
                vDesvEjeMenor.add(is.readDouble());
                vOrientacionMayor.add(is.readDouble());
                vDesvLatitud.add(is.readDouble());
                vDesvLongitud.add(is.readDouble());
                vDesvAltura.add(is.readDouble());
                vHdgPoloN.add(is.readDouble());
                vHdgPoloM.add(is.readDouble());
                vSpeed.add(is.readDouble());
                vHora.add(is.readUTF());
                vLatitud.add(is.readDouble());
                vLongitud.add(is.readDouble());
                vAltura.add(is.readDouble());
                vX.add(is.readDouble());
                vY.add(is.readDouble());
                vZ.add(is.readDouble());
                vAngulo.add(is.readDouble());
                vLatitudG.add(is.readUTF());
                vLongitudG.add(is.readUTF());
                vSatelites.add(is.readInt());
                vMsl.add(is.readDouble());
                vHGeoide.add(is.readDouble());
                vAge.add(is.readDouble());
            }

            is.close();
        } catch (UTFDataFormatException udfe) {
            System.err.println("Error con el formato de la cadena: " + udfe.getMessage());
            System.exit(1);
        } catch (FileNotFoundException fnfe) {
            System.err.println("Fichero inexistente: " + fnfe.getMessage());
            System.exit(1);
        } catch (IOException ioe) {
            System.err.println("Error de E/S: " + ioe.getMessage());            
        }
    }

    public void run() {
        int i = 0;

        while (true) {
            try {
                Thread.sleep(((Long) vTiempo.elementAt(i)).longValue());
            } catch (Exception e) {}

            String cadena = (String) vCadena.elementAt(i);
            
            try {                
                gps.procesaCadena(cadena);                
            } catch(Exception e) {
                System.err.println("Error al procesar la cadena " + cadena);
                System.err.println("\t" + e.getMessage());
            }
            
            i = (i + 1) % vCadena.size();
        }
    }
    
    public GPSConnection getGps() {
        return gps;
    }
    
    public double[][] getRuta() {
        Vector rutaGPS = gps.getRutaEspacial();
        
        double ruta[][] = new double[rutaGPS.size()][];
        
        for (int i = 0; i < rutaGPS.size(); i++) {
            GPSData data = (GPSData)rutaGPS.elementAt(i);
            ruta[i] = new double[] { data.getXLocal(), data.getYLocal(), data.getZLocal() };
        }
        
        return ruta;
    }
    
    public double[] getAngulos() {
        Vector rutaGPS = gps.getRutaEspacial();
        
        double angulos[] = new double[rutaGPS.size()];
        
        for (int i = 0; i < rutaGPS.size(); i++) {
            GPSData data = (GPSData)rutaGPS.elementAt(i);
            angulos[i] = ((GPSData)rutaGPS.elementAt(i)).getAngulo();
        }
        
        return angulos;
    }
    
    public double[] getVelocidades() {
        Vector rutaGPS = gps.getRutaEspacial();
        
        double velocidades[] = new double[rutaGPS.size()];
        
        for (int i = 0; i < rutaGPS.size(); i++) {
            GPSData data = (GPSData)rutaGPS.elementAt(i);
            velocidades[i] = ((GPSData)rutaGPS.elementAt(i)).getVelocidad();
        }
        
        return velocidades;
    }      
    
    public double[] getXY() {
        GPSData data = gps.getPuntoActualEspacial();
        
        return (new double[] { data.getXLocal(), data.getYLocal(), data.getZLocal() });
    }
    
    public double getAngulo() {
        return gps.getPuntoActualEspacial().getAngulo();
    }
    
    public double getVelocidad() {
        return gps.getPuntoActualEspacial().getVelocidad();
    }

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        SimulaGps sGps = new SimulaGps("nov19g.dat", "nov19.gps");
        GPSConnection gps = sGps.getGps();
        /*gps.startRuta();
        try {
            Thread.sleep(120000);
        } catch (Exception e) {}
        gps.stopRuta();
        gps.saveRuta("nov19g.dat");
        gps.loadRuta("nov19g.dat");        
        for (int i = 0; i < gps.getRutaTemporal().size(); i++) {
            System.out.println("**********************************************");
            System.out.println((GPSData)gps.getRutaTemporal().elementAt(i));
            System.out.println("==============================================");
            System.out.println((GPSData)gps.getBufferRutaTemporal().elementAt(i));
            System.out.println("**********************************************");
        }*/
        /*try {
            Thread.sleep(10000);
        } catch (Exception e) {}
        System.out.println(gps.getPuntoActualEspacial());
        Vector retorno = gps.getLastPuntosEspacial(10000);
        System.out.println(retorno.size());
        for (int i = 0; i < retorno.size(); i++) {
            System.out.println((GPSData)retorno.elementAt(i));
        }*/
    }
}


//~ Formatted by Jindent --- http://www.jindent.com
