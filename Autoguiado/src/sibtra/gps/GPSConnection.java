package sibtra.gps;

import Jama.Matrix;
import java.io.*;
import java.util.*;
import java.util.regex.*;
//import javax.comm.*;
import gnu.io.*;

/**
 A class that handles the details of a serial connection. Reads from one
 TextArea and writes to a second TextArea.
 Holds the state of the connection.
 */
public class GPSConnection implements SerialPortEventListener {

    private static final long a = 6378137;
    private static final double b = 6356752.31424518d;
    private static final double e = 0.0821;//0.08181919084262032d;
    private static final double e1 = 1.4166d;
    private static final int MAXBUFFER = 1000;

    private static final double NULLANG = -5 * Math.PI;

    // Vector del polo N
    double u[] = new double[] { 0, b };

    private boolean open;

    private SerialParameters parameters;    
    private InputStream is;    
    
    private CommPortIdentifier portId;
    private SerialPort sPort;

    private String cadena = "";
    GPSData data = new GPSData();    

    public static double minDistOperativa = 0.7;
    
    // Centro de la ruta
    private GPSData centro = null;
    
    // Matriz de cambio de coordenadas
    Matrix T = null;
    
    private Vector rutaTemporal = new Vector();
    private Vector rutaEspacial = new Vector();
    
    private Vector bufferEspacial = new Vector();
    private Vector bufferTemporal = new Vector();  
     
    private boolean enRuta = false;
    private Vector bufferRutaTemporal = new Vector();
    private Vector bufferRutaEspacial = new Vector();
    

  /**
         Creates a SerialConnection object and initilizes variables passed in
         as params.

         @param parameters A SerialParameters object.
     */
    public GPSConnection() {
      //lastPaquete = System.currentTimeMillis();
    }

    public GPSConnection(String portName) {
        parameters = new SerialParameters(portName, 9600, 0, 0, 8, 1, 0);
        try {
            openConnection();
        } catch (SerialConnectionException e2) {

            System.out.println("Error al abrir el puerto " + portName);
            System.out.flush();
        }
        if (isOpen()) {
            System.out.println("Puerto Abierto BaudRate " + portName);
        }
        //lastPaquete = System.currentTimeMillis();
    }

    /**
        Attempts to open a serial connection and streams using the parameters
        in the SerialParameters object. If it is unsuccesfull at any step it
        returns the port to a closed state, throws a
        <code>SerialConnectionException</code>, and returns.

     Gives a timeout of 30 seconds on the portOpen to allow other applications
        to reliquish the port if have it open and no longer need it.
     */
    public void openConnection() throws SerialConnectionException {
        // Obtain a CommPortIdentifier object for the port you want to open.
        try {
            portId =
                    CommPortIdentifier.getPortIdentifier(parameters.getPortName());
        } catch (NoSuchPortException e) {
            throw new SerialConnectionException(e.getMessage());
        }

        // Open the port represented by the CommPortIdentifier object. Give
        // the open call a relatively long timeout of 30 seconds to allow
        // a different application to reliquish the port if the user
        // wants to.
        try {
            sPort = (SerialPort) portId.open("SerialDemo", 30000);
        } catch (PortInUseException e) {
            throw new SerialConnectionException(e.getMessage());
        }

        // Set the parameters of the connection. If they won't set, close the
        // port before throwing an exception.
        try {
            setConnectionParameters();
        } catch (SerialConnectionException e) {
            sPort.close();
            throw e;
        }

        // Open the input and output streams for the connection. If they won't
        // open, close the port before throwing an exception.
        try {            
            is = sPort.getInputStream();
        } catch (IOException e) {
            sPort.close();
            throw new SerialConnectionException("Error opening i/o streams");
        }

        // Add this object as an event listener for the serial port.
        try {
            sPort.addEventListener(this);
        } catch (TooManyListenersException e) {
            sPort.close();
            throw new SerialConnectionException("too many listeners added");
        }

        // Set notifyOnDataAvailable to true to allow event driven input.
        sPort.notifyOnDataAvailable(true);

        // Set notifyOnBreakInterrup to allow event driven break handling.
        sPort.notifyOnBreakInterrupt(true);

        // Set receive timeout to allow breaking out of polling loop during
        // input handling.
        //	try {
        //	    sPort.enableReceiveTimeout(30);
        //	} catch (UnsupportedCommOperationException e) {
        //	}                

        open = true;

        sPort.disableReceiveTimeout();
    }

    /**
     Sets the connection parameters to the setting in the parameters object.
         If set fails return the parameters object to origional settings and
         throw exception.
     */
    public void setConnectionParameters() throws SerialConnectionException {

        // Save state of parameters before trying a set.
        int oldBaudRate = sPort.getBaudRate();
        int oldDatabits = sPort.getDataBits();
        int oldStopbits = sPort.getStopBits();
        int oldParity = sPort.getParity();
        int oldFlowControl = sPort.getFlowControlMode();

        // Set connection parameters, if set fails return parameters object
        // to original state.
        try {
            sPort.setSerialPortParams(parameters.getBaudRate(),
                                      parameters.getDatabits(),
                                      parameters.getStopbits(),
                                      parameters.getParity());
        } catch (UnsupportedCommOperationException e) {
            parameters.setBaudRate(oldBaudRate);
            parameters.setDatabits(oldDatabits);
            parameters.setStopbits(oldStopbits);
            parameters.setParity(oldParity);
            throw new SerialConnectionException("Unsupported parameter");
        }

        // Set flow control.
        try {
            sPort.setFlowControlMode(parameters.getFlowControlIn()
                                     | parameters.getFlowControlOut());
        } catch (UnsupportedCommOperationException e) {
            throw new SerialConnectionException("Unsupported flow control");
        }
    }

    /**
         Close the port and clean up associated elements.
     */
    public void closeConnection() {
        // If port is alread closed just return.
        if (!open) {
            return;
        }

        // Remove the key listener.
        //	messageAreaOut.removeKeyListener(keyHandler);

        // Check to make sure sPort has reference to avoid a NPE.
        if (sPort != null) {
            try {
                // close the i/o streams.                
                is.close();                
            } catch (IOException e) {
                System.err.println(e);
            }

            // Close the port.
            sPort.close();            
        }

        open = false;
    }

    /**
         Send a one second break signal.
     */
    public void sendBreak() {
        sPort.sendBreak(1000);
    }

    /**
         Reports the open status of the port.
         @return true if port is open, false if port is closed.
     */
    public boolean isOpen() {
        return open;
    }

    /**
         Handles SerialPortEvents. The two types of SerialPortEvents that this
         program is registered to listen for are DATA_AVAILABLE and BI. During
     DATA_AVAILABLE the port buffer is read until it is drained, when no more
         data is availble and 30ms has passed the method returns. When a BI
     event occurs the words BREAK RECEIVED are written to the messageAreaIn.
     */

    public synchronized void serialEvent(SerialPortEvent e) {
        if (e.getEventType() == e.DATA_AVAILABLE) {
            try {
                while (is.available() != 0) {
                    int val = is.read();
                    if (val != 10) {
                        cadena += (char) val;
                    } else {
                      procesaCadena(cadena);
                      cadena = "";
                    }
                }
            } catch (IOException ioe) {
                System.err.println("\nError al recibir los datos");
            } catch (Exception ex) {
              System.err.println("\nGPSConnection Error: Cadena fragmentada " + ex.getMessage());
              cadena = "";
            }
        }
    }

    public void procesaCadena(String cadena) {//throws IOException, Exception {
        String[] msj = cadena.split(",");

      if (Pattern.matches("\\$..GSA", msj[0])) {
        //System.out.println(System.currentTimeMillis() + "***" + cadena + "***");
        if (msj[15].equals("")) {
          data.setPDOP(0);
        } else {
          data.setPDOP(Double.parseDouble(msj[15]));
        }

        if (msj[16].equals("")) {
          data.setHDOP(0);
        } else {
          data.setHDOP(Double.parseDouble(msj[16]));
        }

        msj[17] = (msj[17].split("\\*"))[0];
        if (msj[17].equals("")) {
          data.setVDOP(0);
        } else {
          data.setVDOP(Double.parseDouble(msj[17]));
        }
      }

      if (Pattern.matches("\\$..GST", msj[0])) {
        //System.out.println(System.currentTimeMillis() + "***" + cadena + "***");

        if (msj[2].equals("")) {
          data.setRms(0);
        } else {
          data.setRms(Double.parseDouble(msj[2]));
        }

        if (msj[3].equals("")) {
          data.setDesvEjeMayor(0);
        } else {
          data.setDesvEjeMayor(Double.parseDouble(msj[3]));
        }

        if (msj[4].equals("")) {
          data.setDesvEjeMenor(0);
        } else {
          data.setDesvEjeMenor(Double.parseDouble(msj[4]));
        }

        if (msj[5].equals("")) {
          data.setOrientacionMayor(0);
        } else {
          data.setOrientacionMayor(Double.parseDouble(msj[5]));
        }

        if (msj[6].equals("")) {
          data.setDesvLatitud(0);          
        } else {
          data.setDesvLatitud(Double.parseDouble(msj[6]));
        }

        if (msj[7].equals("")) {
          data.setDesvLongitud(0);
        } else {
          data.setDesvLongitud(Double.parseDouble(msj[7]));
        }

        msj[8] = (msj[8].split("\\*"))[0];
        if (msj[8].equals("")) {
          data.setDesvAltura(0);
        } else {
          data.setDesvAltura(Double.parseDouble(msj[8]));
        }

      }

      if (Pattern.matches("\\$..VTG", msj[0])) {
        //System.out.println(System.currentTimeMillis() + "***" + cadena + "***");
        if ((msj[2].equals("T") && (! msj[1].equals("")))) {
          data.setHdgPoloN(Math.toRadians(Double.parseDouble(msj[1])));
        } else {
          data.setHdgPoloN(0);
        }

        if ((msj[4].equals("M")) && (! msj[3].equals(""))) {
          data.setHdgPoloM(Math.toRadians(Double.parseDouble(msj[3])));
        } else {
          data.setHdgPoloM(0);
        }

        msj[8] = (msj[8].split("\\*"))[0];
        if ((msj[8].equals("K")) && (msj[7].equals(""))) {
            data.setVelocidadGPS(0);
        } else {
            data.setVelocidadGPS(Double.parseDouble(msj[7]));
        }        
      }

      if (Pattern.matches("\\$..GGA", msj[0])) {
        //System.out.println(System.currentTimeMillis() + "***" + cadena + "***");

        if (msj[1].equals("")) {
          data.setHora("");
        } else {
          data.setHora(cadena2Time(msj[1]));
        }
        if (msj[2].equals("")) {
          data.setLatitud(0);
        } else {
          data.setLatitud(sexagesimal2double(msj[2], 2));          
        }
        if (! msj[3].equals("")) {
          if (msj[3].equals("S"))
            data.setLatitud(data.getLatitud() * -1);            
        }
        if (msj[2].equals("")) {
          data.setLongitud(0);
        } else {
          data.setLongitud(sexagesimal2double(msj[4], 3));          
        }
        if (! msj[5].equals(""))  {
          if (msj[5].equals("W"))
            data.setLongitud(data.getLongitud() * -1);
        }

        if (msj[7].equals("")) {            
          data.setSatelites(0);
        } else {
          data.setSatelites(Integer.parseInt(msj[7]));
        }

        if ((!msj[9].equals("")) || (!msj[10].equals("M"))) {
          data.setMSL(Double.parseDouble(msj[9]));
        } else {
          data.setMSL(0);
        }

        if ((!msj[11].equals("")) || (!msj[12].equals("M"))) {
          data.setHGeoide(Double.parseDouble(msj[11]));
        } else {
          data.setHGeoide(0);
        }
        
        //altura = msl + hgeoide;
        //data.setAltura(data.getHGeoide() + data.getMSL());
        data.setAltura(data.getHGeoide());


        if (msj[13].equals("")) {
          data.setAge(-1);
        } else {
          data.setAge(Double.parseDouble(msj[13]));
        }

        //calculaLLA(latitud, longitud, altura);        
      }
      
      calculaValores(cadena);
    }
    
        /**
     * Calcula los valores de �ngulo y velocidad
     */
    private void calculaValores(String cadena) {
      data.setSysTime(System.currentTimeMillis());
      setECEF();
      setCoordenadasLocales();
      anadeBufferTemporal();
      anadeBufferEspacial();
      data = new GPSData();
    }
        
    private void updateBuffers() {
        
        if (rutaEspacial != null && rutaEspacial.size() != 0) {
            for (int i = 0; i < rutaEspacial.size(); i++) {
                GPSData elem = (GPSData)(rutaEspacial.elementAt(i));
                Matrix res = new Matrix(new double[][] { {elem.getX() - centro.getX()}, 
                                                     {elem.getY() - centro.getY()}, 
                                                     {elem.getZ() - centro.getZ()} });                
                res = T.times(res).transpose();
                elem.setXLocal(res.getArray()[0][0]);
                elem.setYLocal(res.getArray()[0][1]);
                elem.setZLocal(res.getArray()[0][2]);                                
            }
        }
        
        if (rutaTemporal != null && rutaTemporal.size() != 0) {
            for (int i = 0; i < rutaTemporal.size(); i++) {
                GPSData elem = (GPSData)(rutaTemporal.elementAt(i));                
                Matrix res = new Matrix(new double[][] { {elem.getX() - centro.getX()}, 
                                                     {elem.getY() - centro.getY()}, 
                                                     {elem.getZ() - centro.getZ()} });            
                res = T.times(res).transpose();
                elem.setXLocal(res.getArray()[0][0]);
                elem.setYLocal(res.getArray()[0][1]);
                elem.setZLocal(res.getArray()[0][2]);
            }
        }
        
        if (bufferRutaEspacial != null && bufferRutaEspacial.size() != 0) {
            for (int i = 0; i < bufferRutaEspacial.size(); i++) {
                GPSData elem = (GPSData)(bufferRutaEspacial.elementAt(i));
                Matrix res = new Matrix(new double[][] { {elem.getX() - centro.getX()}, 
                                                     {elem.getY() - centro.getY()}, 
                                                     {elem.getZ() - centro.getZ()} });                
                res = T.times(res).transpose();
                elem.setXLocal(res.getArray()[0][0]);
                elem.setYLocal(res.getArray()[0][1]);
                elem.setZLocal(res.getArray()[0][2]);                                
            }
        }
        
        if (bufferRutaTemporal != null && bufferRutaTemporal.size() != 0) {
            for (int i = 0; i < bufferRutaTemporal.size(); i++) {
                GPSData elem = (GPSData)(bufferRutaTemporal.elementAt(i));                
                Matrix res = new Matrix(new double[][] { {elem.getX() - centro.getX()}, 
                                                     {elem.getY() - centro.getY()}, 
                                                     {elem.getZ() - centro.getZ()} });            
                res = T.times(res).transpose();
                elem.setXLocal(res.getArray()[0][0]);
                elem.setYLocal(res.getArray()[0][1]);
                elem.setZLocal(res.getArray()[0][2]);
            }
        }
        
        if (bufferEspacial != null && bufferEspacial.size() != 0) {
            for (int i = 0; i < bufferEspacial.size(); i++) {
                GPSData elem = (GPSData)(bufferEspacial.elementAt(i));
                Matrix res = new Matrix(new double[][] { {elem.getX() - centro.getX()}, 
                                                     {elem.getY() - centro.getY()}, 
                                                     {elem.getZ() - centro.getZ()} });                
                res = T.times(res).transpose();
                elem.setXLocal(res.getArray()[0][0]);
                elem.setYLocal(res.getArray()[0][1]);
                elem.setZLocal(res.getArray()[0][2]);                                
            }
        }
        
        if (bufferTemporal != null && bufferTemporal.size() != 0) {
            for (int i = 0; i < bufferTemporal.size(); i++) {
                GPSData elem = (GPSData)(bufferTemporal.elementAt(i));                
                Matrix res = new Matrix(new double[][] { {elem.getX() - centro.getX()}, 
                                                     {elem.getY() - centro.getY()}, 
                                                     {elem.getZ() - centro.getZ()} });            
                res = T.times(res).transpose();
                elem.setXLocal(res.getArray()[0][0]);
                elem.setYLocal(res.getArray()[0][1]);
                elem.setZLocal(res.getArray()[0][2]);
            }
        }
        
        
    }
    
    private void setParams(Vector ruta) {
        if (ruta == null || ruta.size() == 0)
            return;                
        
        // Primero buscamos el punto central exacto
        double xCentral = 0, yCentral = 0, zCentral = 0;
        for (int i = 0; i < ruta.size(); i++) {
            xCentral += ((GPSData)ruta.elementAt(i)).getX();
            yCentral += ((GPSData)ruta.elementAt(i)).getY();
            zCentral += ((GPSData)ruta.elementAt(i)).getZ();            
        }
        xCentral /= ruta.size();
        yCentral /= ruta.size();
        zCentral /= ruta.size();
        
        // Ahora buscamos el punto que más cerca esté de ese centro
        double dist = Math.sqrt(Math.pow(((GPSData)ruta.elementAt(0)).getX() - xCentral, 2.0f) + 
                    Math.pow(((GPSData)ruta.elementAt(0)).getY() - yCentral, 2.0f) + 
                    Math.pow(((GPSData)ruta.elementAt(0)).getZ() - zCentral, 2.0f));
        centro = (GPSData)ruta.elementAt(0);
        for (int i = 0; i < ruta.size(); i++) {
            double myDist = Math.sqrt(Math.pow(((GPSData)ruta.elementAt(i)).getX() - xCentral, 2.0f) + 
                    Math.pow(((GPSData)ruta.elementAt(i)).getY() - yCentral, 2.0f) + 
                    Math.pow(((GPSData)ruta.elementAt(i)).getZ() - zCentral, 2.0f));
            if (myDist < dist) {
                dist = myDist;
                centro = (GPSData)ruta.elementAt(i);
            }            
        }
        
        // Matriz de rotaci�n en torno a un punto
        double v[][] = new double[3][];
        v[0] = new double[] { -Math.sin(centro.getLongitud()), Math.cos(centro.getLongitud()), 0 };
        v[1] = new double[] { -Math.cos(centro.getLongitud()) * Math.sin(centro.getLatitud()), -Math.sin(centro.getLatitud()) * Math.sin(centro.getLongitud()), Math.cos(centro.getLatitud()) };
        v[2] = new double[] { Math.cos(centro.getLatitud()) * Math.cos(centro.getLongitud()), Math.cos(centro.getLatitud()) * Math.sin(centro.getLongitud()), Math.sin(centro.getLatitud())};

        Matrix M1 = new Matrix(v);

        // Matriz de inversi�n del eje z en torno al eje x (Norte)
        double w[][] = new double[3][];
        w[0] = new double[] { -1, 0, 0 };
        w[1] = new double[] { 0, 1, 0 };
        w[2] = new double[] { 0, 0, -1 };
        Matrix M2 = new Matrix(w);

        T = M2.times(M1);  
        
        updateBuffers();
    }

    public void setECEF() {
      double altura = data.getAltura();
      double latitud = Math.toRadians(data.getLatitud());
      double longitud = Math.toRadians(data.getLongitud());
      double N = a / Math.sqrt(1 - (Math.pow(e, 2.0f) * Math.pow(Math.sin(latitud), 2.0f)));
      double x = (N + altura) * Math.cos(latitud) * Math.cos(longitud);
      double y = (N + altura) * Math.cos(latitud) * Math.sin(longitud);
      double z = ( ( (Math.pow(b, 2.0f) / Math.pow(a, 2.0f)) * N) + altura) * Math.sin(latitud);
      data.setX(x);
      data.setY(y);
      data.setZ(z);      
    }
    
    public void setCoordenadasLocales() {
        if (T == null) {       
            if (bufferEspacial.size() > 100) {
                setParams(bufferEspacial);
            } else {
                data.setXLocal(-1);
                data.setYLocal(-1);
                data.setZLocal(-1);
                return;
            }
        }
        
        Matrix res = new Matrix(new double[][] { {data.getX() - centro.getX()}, 
                                                     {data.getY() - centro.getY()}, 
                                                     {data.getZ() - centro.getZ()} });            
        res = T.times(res).transpose();
        data.setXLocal(res.getArray()[0][0]);
        data.setYLocal(res.getArray()[0][1]);
        data.setZLocal(res.getArray()[0][2]);                    
    }
    
    private void anadeBufferTemporal() {
        calculaAngSpeed(bufferTemporal, data);
        bufferTemporal.add(data);
        
        if (enRuta) {
            bufferRutaTemporal.add(data);
        }
        
        while (bufferTemporal.size() > this.MAXBUFFER) {
            bufferTemporal.remove(0);
        }
    }
    
    private void anadeBufferEspacial() {       
        if (bufferEspacial.size() == 0) {
            calculaAngSpeed(bufferEspacial, data);
            bufferEspacial.add(data);
            return;
        }
            
        double dist = Math.sqrt(Math.pow(data.getX() - ((GPSData)bufferEspacial.lastElement()).getX(), 2.0f) + 
                Math.pow(data.getY() - ((GPSData)bufferEspacial.lastElement()).getY(), 2.0f) + 
                Math.pow(data.getZ() - ((GPSData)bufferEspacial.lastElement()).getZ(), 2.0f));
        if (dist > minDistOperativa) {
            calculaAngSpeed(bufferEspacial, data);
            bufferEspacial.add(data);
            
            if (enRuta) {
                bufferRutaEspacial.add(data);
            }
        
            while (bufferEspacial.size() > this.MAXBUFFER) {
                bufferEspacial.remove(0);
            }
        }
    }

    
    private void calculaAngSpeed(Vector buffer, GPSData val) {
        if (buffer.size() == 0) {
            val.setAngulo(0);
            val.setVelocidad(0);
            return;
        }
        double x = val.getXLocal() - ((GPSData)buffer.lastElement()).getXLocal();
        double y = val.getYLocal() - ((GPSData)buffer.lastElement()).getYLocal();       
      
        double ang = Math.atan2(x, y);
        if (ang < 0) ang += 2 * Math.PI;
        
        // En principio no diferencio entre angulo y angulo local
        val.setAngulo(ang);        
        
        double vel = Math.sqrt(Math.pow(x , 2.0f) + Math.pow(y , 2.0f));
        vel /= (val.getSysTime() - ((GPSData)buffer.lastElement()).getSysTime()) / 1000.0;
        val.setVelocidad(vel);                
    }

    public static double sexagesimal2double(String valor, int enteros) {
        int grados = 0;
        float minutos = 0;
        if (valor.length() > enteros) {
            grados = Integer.parseInt(valor.substring(0, enteros));
            minutos = Float.parseFloat(valor.substring(enteros, valor.length()));
            return grados + (minutos / 60.0f);
        } else {
            return Double.parseDouble(valor);
        }
    }

    private static String cadena2Time(String cadena) {
        if (cadena.length() < 6)
            return "";
        int hora = 0;
        int minutos = 0;
        int segundos = 0;
        hora = Integer.parseInt(cadena.substring(0, 2));
        minutos = Integer.parseInt(cadena.substring(2, 4));
        segundos = Integer.parseInt(cadena.substring(4, 6));
        return hora + ":" + minutos + ":" + segundos;
    }

    public void setMinDistOperativa(double minDistOperativa) {
        this.minDistOperativa = minDistOperativa;
    }
    
    public double getMinDistOperativa() {
        return this.minDistOperativa;
    }

    public GPSData getPuntoActualEspacial() {
        if (bufferEspacial == null || bufferEspacial.size() == 0) return null;
        return (GPSData)(bufferEspacial.lastElement());
    }
    
    public GPSData getPuntoActualTemporal() {
        if (bufferTemporal == null || bufferTemporal.size() == 0) return null;
        return (GPSData)(bufferTemporal.lastElement());
    }
    
    public Vector getLastPuntosEspacial(int n) {
        if (bufferEspacial == null || bufferEspacial.size() == 0) return null;
        
        Vector retorno = new Vector();
        Vector buffer = (Vector)bufferEspacial.clone();
        
        for (int i = n; i > 0; i--) {          
            if (buffer.size() - i < 0) continue;
            retorno.add(buffer.elementAt(buffer.size() - i));
        }        
        
        return retorno;
    }
    
    public Vector getLastPuntosTemporal(int n) {
        if (bufferTemporal == null || bufferTemporal.size() == 0) return null;
        
        Vector retorno = new Vector();
        Vector buffer = (Vector)bufferTemporal.clone();
        
        for (int i = n; i > 0; i--) {          
            if (buffer.size() - i < 0) continue;
            retorno.add(buffer.elementAt(buffer.size() - i));
        }   
        
        return retorno;
    }
    
    public Vector getBufferEspacial() {
        return bufferEspacial;
    }
    
    public Vector getBufferTemporal() {
        return bufferTemporal;
    }
    
    public Vector getBufferRutaEspacial() {
        return bufferRutaEspacial;
    }
    
    public Vector getBufferRutaTemporal() {        
        return bufferRutaTemporal;
    }
    
    public Vector getRutaEspacial() {
        return rutaEspacial;
    }
    
    public Vector getRutaTemporal() {        
        return rutaTemporal;
    }
    
    public void startRuta() {
        bufferRutaEspacial = new Vector();
        bufferRutaTemporal = new Vector();
        
        enRuta = true;
    }    
    
    public void stopRuta() {
        enRuta = false;
    }
    
    public void stopRutaAndSave(String fichero) {
        enRuta = false;
        
        saveRuta(fichero);
    }
    
    public void saveRuta(String fichero) {        
         try {
            File file = new File(fichero);
 	    ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(file));
            oos.writeInt(bufferRutaEspacial.size());
            oos.writeInt(bufferRutaTemporal.size());
            for (int i = 0; i < bufferRutaEspacial.size(); i++) {
                oos.writeObject(bufferRutaEspacial.elementAt(i));
            }
            for (int i = 0; i < bufferRutaTemporal.size(); i++) {
                oos.writeObject(bufferRutaTemporal.elementAt(i));
            }
 	    oos.close();
        } catch (IOException ioe) {
            System.err.println("Error al escribir en el fichero " + fichero);
            System.err.println(ioe.getMessage());
        }
    }
    
    public void loadRuta(String fichero) {
        rutaEspacial = new Vector();
        rutaTemporal = new Vector();
 	//    Vector valores = new Vector();
        try {
            File file = new File(fichero);
 	    ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file));
 	    int tamEspacial = -1;
            int tamTemporal = -1;
            tamEspacial = ois.readInt();
            tamTemporal = ois.readInt();
            for (int i = 0; i < tamEspacial; i++) {
                rutaEspacial.add(ois.readObject());
            }
            for (int i = 0; i < tamTemporal; i++) {
                rutaTemporal.add(ois.readObject());
            }
 	    ois.close();
        } catch (IOException ioe) {
            System.err.println("Error al abrir el fichero " + fichero);
            System.err.println(ioe.getMessage());
        } catch (ClassNotFoundException cnfe) {
            System.err.println("Objeto leído inválido: " + cnfe.getMessage());            
        }     
        
        setParams(rutaEspacial);
    }
    
    public void loadOldRuta(String fichero) {
        rutaTemporal = new Vector();
 	//    Vector valores = new Vector();
        try {
            File file = new File(fichero);
 	    ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file));
 	    double x = 0, y = 0, z = 0, latitud = 0, longitud = 0, altura = 0;
 	    while (ois.available() != 0) {
                GPSData valor = new GPSData();
                valor.setX(ois.readDouble());
                valor.setY(ois.readDouble());
                valor.setZ(ois.readDouble());
                valor.setLatitud(ois.readDouble());
                valor.setLongitud(ois.readDouble());
                valor.setAltura(ois.readDouble());
                valor.setAngulo(ois.readDouble());
                valor.setVelocidad(ois.readDouble());
                
                rutaTemporal.add(valor);
            }
 	    ois.close();
        } catch (IOException ioe) {
            System.err.println("Error al abrir el fichero " + fichero);
            System.err.println(ioe.getMessage());
        }

        setParams(rutaTemporal);
    }

}

