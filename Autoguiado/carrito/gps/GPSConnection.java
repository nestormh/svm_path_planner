package carrito.gps;

import Jama.Matrix;
import java.io.*;
import java.util.*;
import java.util.regex.*;
import javax.comm.*;

import carrito.media.*;
import carrito.server.serial.*;

/**
 A class that handles the details of a serial connection. Reads from one
 TextArea and writes to a second TextArea.
 Holds the state of the connection.
 */
public class GPSConnection implements SerialPortEventListener,
        CommPortOwnershipListener, Runnable {

    private static final long a = 6378137;
    private static final double b = 6356752.31424518d;
    private static final double e = 0.0821;//0.08181919084262032d;
    private static final double e1 = 1.4166d;

    private static final double NULLANG = -5 * Math.PI;

    // Vector del polo N
    double u[] = new double[] { 0, b };

    private boolean open;

    private SerialParameters parameters;
    private OutputStream os;
    private InputStream is;
    private ObjectOutputStream osECEF;
    private BufferedWriter bw;

    private long lastPaquete = 0;
    private String tipoPaquete = "";
    private ObjectOutputStream oosFull = null;

    private CommPortIdentifier portId;
    private SerialPort sPort;

    private String cadena = "";
    private double latitud = 0;
    private String latitudg = "";
    private double longitud = 0;
    private String longitudg = "";
    private double altura = 0;
    private double x = 0;
    private double y = 0;
    private double z = 0;
    private int satelites = 0;
    private double pdop = 0;
    private double hdop = 0;
    private double vdop = 0;
    private double speed = 0;
    private double msl = 0;
    private double hgeoide = 0;
    private double hdgPoloN = 0;
    private double hdgPoloM = 0;
    private String hora = "";
    private double age = 0;

    // Paquete GST
    private double rms = 0;
    private double desvEjeMayor = 0;
    private double desvEjeMenor = 0;
    private double orientacionMayor = 0;
    private double desvLatitud = 0;
    private double desvLongitud = 0;
    private double desvAltura  = 0;

    // Mi propios valores
    private double angulo = 0;

    CapturaImagen ci1 = null;
    CapturaImagen ci2 = null;

    private String ruta = "";
    private int index = 0;

    private ConexionBD bd = null;
    private Media media = null;
    private String dispositivo1 = null, dispositivo2 = null;

    CambioCoordenadas cc = null;
    private Vector posiciones = new Vector();

    private Vector vectorMedia = new Vector();
    private int maxMedidas = 5;

    private boolean write = false;
    private Vector vCaptura = new Vector();

    private boolean independiente = false;
    private boolean filtrarPuntos = true;
    private boolean filtrarAngulos = true;

    private long lastInstruccion = System.currentTimeMillis() * 2;

    private SerialConnection sc = null;

    // Posición anterior para calcular el ángulo
    double posAnt[] = { 0, 0, 0 };
    // Posición anterior para calcular la posición actual si hay filtrado
    double oldLLA[] = new double[]{ 0, 0 ,0 };
    double[] oldAng = new double[] { NULLANG, NULLANG };
    double[] lastAng = new double[] { NULLANG, NULLANG };

    public static double minDistOperativa = 0.4;

  /**
         Creates a SerialConnection object and initilizes variables passed in
         as params.

         @param parameters A SerialParameters object.
     */
    public GPSConnection() {
      lastPaquete = System.currentTimeMillis();
    }

    public GPSConnection(SerialParameters parameters) {
        this.parameters = parameters;
        open = false;
        try {
            openConnection();
        } catch (SerialConnectionException e2) {

            System.out.println("Error al abrir el puerto GPSConnection");
            System.out.flush();
        }
        if (isOpen()) {
            System.out.println("Puerto Abierto");
        }
        lastPaquete = System.currentTimeMillis();
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
        lastPaquete = System.currentTimeMillis();
    }

    public GPSConnection(String portName,
                            int baudRate,
                            int flowControlIn,
                            int flowControlOut,
                            int databits,
                            int stopbits,
                            int parity) {

        parameters = new SerialParameters(portName, baudRate, flowControlIn,
                                          flowControlOut, databits, stopbits,
                                          parity);
        try {
            openConnection();
        } catch (SerialConnectionException e2) {

            System.out.println("Error al abrir el puerto " + portName);
            System.out.flush();

        }
        if (isOpen()) {
            System.out.println("Puerto Abierto BaudRate " + baudRate);
        }
        lastPaquete = System.currentTimeMillis();
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
            os = sPort.getOutputStream();
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

        // Add ownership listener to allow ownership event handling.
        portId.addPortOwnershipListener(this);

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
                os.close();
                is.close();
                if (write) {
                  osECEF.close();
                  bw.close();
                }
            } catch (IOException e) {
                System.err.println(e);
            }

            // Close the port.
            sPort.close();

            // Remove the ownership listener.
            portId.removePortOwnershipListener(this);
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

    public void procesaCadena(String cadena) throws IOException, Exception {
      String[] msj = cadena.split(",");

      if (Pattern.matches("\\$..GSA", msj[0])) {
        //System.out.println(System.currentTimeMillis() + "***" + cadena + "***");
        if (msj[15].equals("")) {
          pdop = 0;
        } else {
          pdop = Double.parseDouble(msj[15]);
        }

        if (msj[16].equals("")) {
          hdop = 0;
        } else {
          hdop = Double.parseDouble(msj[16]);
        }

        msj[17] = (msj[17].split("\\*"))[0];
        if (msj[17].equals("")) {
          vdop = 0;
        } else {
          vdop = Double.parseDouble(msj[17]);
        }
      }

      if (Pattern.matches("\\$..GST", msj[0])) {
        //System.out.println(System.currentTimeMillis() + "***" + cadena + "***");

        if (msj[2].equals("")) {
          rms = 0;
        } else {
          rms = Double.parseDouble(msj[2]);
        }

        if (msj[3].equals("")) {
          desvEjeMayor = 0;
        } else {
          desvEjeMayor = Double.parseDouble(msj[3]);
        }

        if (msj[4].equals("")) {
          desvEjeMenor = 0;
        } else {
          desvEjeMenor = Double.parseDouble(msj[4]);
        }

        if (msj[5].equals("")) {
          orientacionMayor = 0;
        } else {
          orientacionMayor = Double.parseDouble(msj[5]);
        }

        if (msj[6].equals("")) {
          desvLatitud = 0;
        } else {
          desvLatitud = Double.parseDouble(msj[6]);
        }

        if (msj[7].equals("")) {
          desvLongitud = 0;
        } else {
          desvLongitud = Double.parseDouble(msj[7]);
        }

        msj[8] = (msj[8].split("\\*"))[0];
        if (msj[8].equals("")) {
          desvAltura = 0;
        } else {
          desvAltura = Double.parseDouble(msj[8]);
        }

      }

      if (Pattern.matches("\\$..VTG", msj[0])) {
        //System.out.println(System.currentTimeMillis() + "***" + cadena + "***");
        if ((msj[2].equals("T") && (! msj[1].equals("")))) {
          hdgPoloN = double2radians(Double.parseDouble(msj[1]));
        } else {
          hdgPoloN = 0;
        }

        if ((msj[4].equals("M")) && (! msj[3].equals(""))) {
          hdgPoloM = double2radians(Double.parseDouble(msj[3]));
        } else {
          hdgPoloM = 0;
        }

        if (sc != null) {
          speed = sc.getVelocidad();
        } else {
          msj[8] = (msj[8].split("\\*"))[0];
          if ((msj[8].equals("K")) && (msj[7].equals(""))) {
            speed = 0;
          } else {
            speed = Double.parseDouble(msj[7]);
          }
        }

      }
      if (Pattern.matches("\\$..GGA", msj[0])) {
        //System.out.println(System.currentTimeMillis() + "***" + cadena + "***");

        if (msj[1].equals("")) {
          hora = "";
        } else {
          hora = cadena2Time(msj[1]);
        }
        if (msj[2].equals("")) {
          latitud = 0;
        } else {
          latitud = sexagesimal2double(msj[2], 2);
          latitudg = sexagesimal2string(msj[2], 2);
        }
        if (! msj[3].equals("")) {
          if (msj[3].equals("S"))
            latitud *= -1;
          latitudg += " " + msj[3];
        }
        if (msj[2].equals("")) {
          longitud = 0;
        } else {
          longitud = sexagesimal2double(msj[4], 3);
          longitudg = sexagesimal2string(msj[4], 3);
        }
        if (! msj[5].equals(""))  {
          if (msj[5].equals("W"))
            longitud *= -1;
          longitudg += " " + msj[5];
        }

        if (msj[7].equals("")) {
          satelites = 0;
        } else {
          satelites = Integer.parseInt(msj[7]);
        }

        if ((!msj[9].equals("")) || (!msj[10].equals("M"))) {
          msl = Double.parseDouble(msj[9]);
        } else {
          msl = 0;
        }

        if ((!msj[11].equals("")) || (!msj[12].equals("M"))) {
          hgeoide = Double.parseDouble(msj[11]);
        } else {
          hgeoide = 0;
        }
        //altura = msl + hgeoide;
        altura = hgeoide;


        if (msj[13].equals("")) {
          age = -1;
        } else {
          age = Double.parseDouble(msj[13]);
        }

        //calculaLLA(latitud, longitud, altura);
        setValores();

        if (write) {
          vCaptura.add(new double[] { x, y, z });
          osECEF.writeDouble(x);
          osECEF.writeDouble(y);
          osECEF.writeDouble(z);
          osECEF.writeDouble(latitud);
          osECEF.writeDouble(longitud);
          osECEF.writeDouble(altura);
          osECEF.writeDouble(angulo);
          osECEF.writeDouble(speed);
          bw.write("(" + x + ", " + y + ", " + z + ")\n");
          //System.out.println("Escribiendo: (" + x + ", " + y + ", " + z + ")");

          oosFull.writeUTF(cadena);
          oosFull.writeUTF(msj[0].substring(3, 5));
          oosFull.writeLong(System.currentTimeMillis() - lastPaquete);
          // Variables del paquete GSA:
          oosFull.writeDouble(pdop);
          oosFull.writeDouble(hdop);
          oosFull.writeDouble(vdop);
          // Variables del paquete GST:
          oosFull.writeDouble(rms);
          oosFull.writeDouble(desvEjeMayor);
          oosFull.writeDouble(desvEjeMenor);
          oosFull.writeDouble(orientacionMayor);
          oosFull.writeDouble(desvLatitud);
          oosFull.writeDouble(desvLongitud);
          oosFull.writeDouble(desvAltura);
          // Variables del paquete VTG
          oosFull.writeDouble(hdgPoloN);
          oosFull.writeDouble(hdgPoloM);
          oosFull.writeDouble(speed);
          // Variables del paquete GGA
          oosFull.writeUTF(hora);
          oosFull.writeDouble(latitud);
          oosFull.writeDouble(longitud);
          oosFull.writeDouble(altura);
          oosFull.writeDouble(x);
          oosFull.writeDouble(y);
          oosFull.writeDouble(z);
          oosFull.writeDouble(angulo);
          oosFull.writeUTF(latitudg);
          oosFull.writeUTF(longitudg);
          oosFull.writeInt(satelites);
          oosFull.writeDouble(msl);
          oosFull.writeDouble(hgeoide);
          oosFull.writeDouble(age);

          if (ci1 != null) {
            String nombre = ruta + "\\Imagen" + index + "a.jpg";
            System.out.println(nombre);
            ci1.saveImagen(nombre);
            //ImagenId ii = ci1.getImagen(x, y, z);
          }
          if (ci2 != null) {
            String nombre = ruta + "\\Imagen" + index + "b.jpg";
            System.out.println(nombre);
            ci2.saveImagen(nombre);
          }
          if ((ci1 != null) || (ci2 != null))
            index++;
        }

        if (bd != null) {
          ObjetoRuta or = new ObjetoRuta(media, dispositivo1, dispositivo2,
                                         x, y, z, angulo, speed);
          for (int i = 0; i < 10; i++) {
            System.out.print("[" + or.getImg1()[i] + "]");
          }
          System.out.println();
          //bd.writeObject(or, ruta);
        }
        lastInstruccion = System.currentTimeMillis();
      }
      lastPaquete = System.currentTimeMillis();
    }

    public void ownershipChange(int type) {
    }

    /**
         Devuelve el Objeto de escritura del puerto serie

     */
    public OutputStream getOutputStream() {
        return os;

    }

    public GPSData getLLA() {
        GPSData data = new GPSData();
        data.setLLA(latitud, longitud, altura);
        return data;
    }

    public void setECEF() {
      double altura = this.altura;
      double latitud = double2radians(this.latitud);
      double longitud = double2radians(this.longitud);
      double N = a / Math.sqrt(1 - (Math.pow(e, 2.0f) * Math.pow(Math.sin(latitud), 2.0f)));
      x = (N + altura) * Math.cos(latitud) * Math.cos(longitud);
      y = (N + altura) * Math.cos(latitud) * Math.sin(longitud);
      z = ( ( (Math.pow(b, 2.0f) / Math.pow(a, 2.0f)) * N) + altura) * Math.sin(latitud);
    }

    public GPSData getECEF() {
        setECEF();
        GPSData data = new GPSData();
        data.setECEF(x, y, z);
        return data;
    }

    public double getSpeed() {
        return speed;
    }

    public int getSatelites() {
        return satelites;
    }

    public double getAltura() {
        return altura;
    }

    public double getHdop() {
        return hdop;
    }

    public double getLatitud() {
        return latitud;
    }

    public double getLongitud() {
        return longitud;
    }

    public double getPdop() {
        return pdop;
    }

    public double getVdop() {
        return vdop;
    }

    public double getX() {
        return x;
    }

    public double getY() {
        return y;
    }

    public double getZ() {
        return z;
    }

    public double[] getXY() {
      return cc.cambioCoordenadas(x, y, z);
    }

    public double[] getXYZ() {
      return new double[] { x, y, z };
    }

    public double getDesvAltura() {
        return desvAltura;
    }

    public double getDesvEjeMayor() {
        return desvEjeMayor;
    }

    public double getDesvEjeMenor() {
        return desvEjeMenor;
    }

    public double getDesvLatitud() {
        return desvLatitud;
    }

    public double getDesvLongitud() {
        return desvLongitud;
    }

    public double getHdgPoloM() {
        return hdgPoloM;
    }

    public double getHdgPoloN() {
        return hdgPoloN;
    }

    public double getHgeoide() {
        return hgeoide;
    }

    public String getHora() {
        return hora;
    }

    public String getLatitudg() {
        return latitudg;
    }

    public String getLongitudg() {
        return longitudg;
    }

    public double getMsl() {
        return msl;
    }

    public double getOrientacionMayor() {
        return orientacionMayor;
    }

    public double getRms() {
        return rms;
    }

    public double getAge() {
        return age;
    }

  public double getAngulo() {
    return angulo;
  }

  public CambioCoordenadas getCc() {
    return cc;
  }

  public boolean isIndependiente() {
    return independiente;
  }

  public long getLastInstruccion() {
    return lastInstruccion;
  }

  public int getMaxMedidas() {
    return maxMedidas;
  }

  public double getMinDistOperativa() {
    return minDistOperativa;
  }

  public void setFiltrarPuntos(boolean valor) {
    this.filtrarPuntos = valor;
  }

  public void setFiltrarAngulos(boolean valor) {
    this.filtrarAngulos = valor;
  }

  public GPSData getError() {
        GPSData data = new GPSData();
        data.setError(pdop, hdop, vdop);
        return data;
    }

    public GPSData getGPSData() {
        setECEF();
        GPSData data = new GPSData(latitud, longitud, altura, x, y, z,
                                   pdop, hdop, vdop, speed, satelites,
                                   latitudg, longitudg, msl, hgeoide, hdgPoloN,
                                   hdgPoloM, hora, age, rms, desvEjeMayor,
                                   desvEjeMenor, orientacionMayor, desvLatitud,
                                   desvLongitud, desvAltura, angulo);
        return data;
    }


    private void Envia(byte a[]) {
        for (int i = 0; i < a.length; i++) {
            try {
                os.write(a[i]);

            } catch (Exception e) {
                System.out.println("Error al enviar");
            }
        }
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

    public static String sexagesimal2string(String valor, int enteros) {
        int grados = 0;
        float minutos = 0;
        if (valor.length() > enteros) {
            grados = Integer.parseInt(valor.substring(0, enteros));
            minutos = Float.parseFloat(valor.substring(enteros, valor.length()));
            return grados + "º " + minutos + "\"";
        } else {
            return valor;
        }
    }

    public static double double2radians(double angle) {
      return angle * Math.PI / 180.0f;
    }

    public static String cadena2Time(String cadena) {
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

    private int recibe() {
        String retorno = "";
        try {
            while (is.available() > 0) {
                retorno += (char) is.read();
            }
        } catch (Exception e) {
            System.out.println("Error al recibir");
        }
        return Integer.parseInt(retorno);
    }

    public void startReading(String nombre) {
      try {
        File ecef = new File(nombre);
        osECEF = new ObjectOutputStream(new FileOutputStream(ecef, false));
        File full = new File(nombre.substring(0, nombre.length() - 4) + ".gps");
        oosFull = new ObjectOutputStream(new FileOutputStream(full, false));
        bw = new BufferedWriter(new FileWriter("ruta.txt", false));
        vCaptura.clear();
        write = true;
      } catch (IOException ioe) {
        System.err.println("No se pudo abrir el flujo de datos: " + ioe.getMessage());
      }
      System.out.println("Comienza el proceso de escritura");
    }

    public void stopReading() {
      if (write) {
        try {
          osECEF.close();
          oosFull.close();
          bw.close();
          write = false;
        } catch (IOException ioe) {}
      }
    }

    public void pauseReading() {
        if (write) {
            write = false;
        } else {
            write = true;
        }
    }

    public void startReadingImages(String nombre, CapturaImagen ci1, CapturaImagen ci2) {
      ruta = nombre.substring(0, nombre.length() - 4);
      File f = new File(ruta);
      if (!f.exists()) {
        f.mkdir();
      }
      index = 0;
      startReading(nombre);
      this.ci1 = ci1;
      this.ci2 = ci2;
    }

    public void stopReadingImages() {
      stopReading();
      ci1 = null;
      ci2 = null;
    }

    public synchronized void startBD(String nombre, Media media, String disp1, String disp2) {
      this.ruta = nombre;
      this.media = media;
      this.dispositivo1 = disp1;
      this.dispositivo2 = disp2;
      bd = new ConexionBD();
      bd.getConnection();
    }

    public synchronized void stopBD() {
      bd.stopConnection();
      bd = null;
      media = null;
    }

    private double regresionLineal(Vector v) {
       int n = v.size();
       double x[] = new double[n];
       double y[] = new double[n];

       // Obtenemos los datos que nos interesan del vector
       for (int i = 0; i < n; i++) {
           double xyz[] = (double[])v.elementAt(i);
           x[i] = xyz[1];
           y[i] = xyz[2];
           // En caso de usar Mercator:
           //double coord[] = ECEF2LLA(xyz[0], xyz[1], xyz[2]);
           //x[i] = coord[1];
           //y[i] = Math.log(Math.tan(Math.PI / 4 + coord[0] / 2));
       }

       // Hacemos la regresión lineal
       double pxy, sx, sy, sx2, sy2, dirX, dirY;
       pxy=sx=sy=sx2=sy2=dirX=dirY=0.0;

       for (int i = 0; i < n; i++) {
           sx += x[i];
           sy += y[i];
           sx2 += x[i] * x[i];
           sy2 += y[i] * y[i];
           pxy += x[i] * y[i];
       }

       double mediaX = sx / n;
       double mediaY = sy / n;

       for (int i = 0; i < n; i++) {
           /*if (x[i] - mediaX >= 0)
               dirX++;
           else dirX--;
           if (y[i] - mediaY >= 0)
               dirY++;
           else dirY--;*/

           dirX += x[i];
           dirY += y[i];
       }

       // Pendiente
       double valY = n * pxy - sx * sy;
       double valX = n * sx2 - sx * sx;

       int cuadrante = -1;

       if (valX == 0) {
           if (valY < 0)
               return Math.toRadians(90);
           else
               return Math.toRadians(270);
       }

       double ang = Math.atan(valY / valX);
       double ang2 = ang;

       // Primer y 3º cuadrante
       if (ang < 0) {
           if (dirX < 0)
               ang = Math.toRadians(180) - Math.abs(ang);
           else
               ang = Math.abs(ang);
       } else { // 3º y 4º cuadrante
           if (dirX > 0)
               ang = Math.toRadians(360) - Math.abs(ang);
           else
               ang = Math.toRadians(180) + Math.abs(ang);
       }

       System.out.println(Math.toDegrees(ang2) + " / " + dirX + " = " + Math.toDegrees(ang));

       return ang;
    }

    /**
     * Establece los valores de ángulo y velocidad
     */
    public void setValores() {
      if (filtrarPuntos) {
        if (oldLLA[0] != 0 || oldLLA[1] != 0 || oldLLA[2] != 0) {
          latitud = (latitud + oldLLA[0]) / 2.0d;
          longitud = (longitud + oldLLA[1]) / 2.0d;
          altura = (altura + oldLLA[2]) / 2.0d;
        }
      }

      setECEF();

      if (Math.sqrt(Math.pow(x - posAnt[0], 2.0f) +
                    Math.pow(y - posAnt[1], 2.0f) +
                    Math.pow(z - posAnt[2], 2.0f)) < minDistOperativa)
        return;

      double valores[] = calculaAnguloVel(new double[] {x, y, z }, posAnt, latitud, longitud);

      angulo = valores[0];
      speed = valores[1] * 5 * 3.60;

      if (filtrarAngulos) {
        if ((oldAng[0] != NULLANG) && (oldAng[1] != NULLANG)){
          double difAng[] = new double[2];
          difAng[0] = Math.min(oldAng[1] - oldAng[0] + ((oldAng[1] - oldAng[0] < 0)? (Math.PI * 2):0),
                               oldAng[0] - oldAng[1] + ((oldAng[0] - oldAng[1] < 0)? (Math.PI * 2):0));
          difAng[1] = Math.min(oldAng[1] - valores[0] + ((oldAng[1] - valores[0] < 0)? (Math.PI * 2):0),
                               valores[0] - oldAng[1] + ((valores[0] - oldAng[1] < 0)? (Math.PI * 2):0));
          System.out.println(difAng[0] + " " + difAng[1] + " " + Math.abs(difAng[1] - difAng[0]) + ";");
          if (Math.abs(difAng[1] - difAng[0]) > 0.0001) {
            angulo = oldAng[1];
          }
        }
      }
      posAnt = new double[] { x, y, z };
      oldLLA = new double[] { latitud, longitud, altura };
      oldAng[0] = oldAng[1];
      oldAng[1] = valores[0];
      lastAng[0] = lastAng[1];
      lastAng[1] = angulo;
    }

    public static double[] calculaAnguloVel(double punto1[], double punto2[], double latitud, double longitud) {
        Matrix m = CambioCoordenadas.getPTP(latitud, longitud);

        double v[] = CambioCoordenadas.cambioCoordenadas(punto1[0], punto1[1], punto1[2], m, new double[] { punto2[0], punto2[1], punto2[2] });

        double angulo = Math.atan2(v[1], v[0]);
        if (angulo < 0)
            angulo += 2 * Math.PI;

        double vel = Math.sqrt(Math.pow(v[0], 2.0f) + Math.pow(v[1], 2.0f));

        return new double[] { angulo, vel };
    }

    // Añade una nueva posicion a la lista y elimina las últimas de distancia superior a 1 m.
    public void getDifAngulo() {
      //if (sc != null && sc.getVelocidad() < 10)
      //    return;

      //double xy[] = cc.cambioCoordenadas(x, y, z);
      double xy[] = new double[] { x, y, z };
      double oldXY[] = null;

      double distancia = -1;

      int indice = -2;

      for (int i = posiciones.size() - 1; i >= 0; i --) {
        oldXY = ((double[])posiciones.elementAt(i));
        distancia = Math.sqrt(Math.pow((xy[0] - oldXY[0]), 2) + Math.pow((xy[1] - oldXY[1]), 2) + Math.pow((xy[2] - oldXY[2]), 2));

        //System.out.println(distancia);
        if (distancia > 1.5) {
          indice = i - 1;
          break;
        }
      }
      posiciones.add(xy);
      if (indice != -2) {            // Se encontro una medida mayor a 1 metro
        for (int i = 0; i < indice; i++) {
          posiciones.remove(0);
        }

        double v[] = new double[] { xy[1] - oldXY[1], xy[2] - oldXY[2] };
        //double v[] = regresionLineal(posiciones);

        double num = u[0] * v[0] + u[1] * v[1];
        double modU = (double)a;
        double modV = Math.sqrt(Math.pow(v[0], 2.0f) + Math.pow(v[1], 2.0f));

        double ang = 0;

        if (modU * modV != 0) {
          ang = Math.acos(num / (modU * modV));
          if (v[0] > 0) {
            ang = 2 * Math.PI - ang;
          }
        }

        angulo = ang;
        //angulo = regresionLineal(posiciones);

        //angulo = hdgPoloN;
      }
    }

    /*public void getDifAngulo(double x, double y, double z) {
      this.x = x;
      this.y = y;
      this.z = z;
      this.getDifAngulo();
    }*/

  public void setCc(CambioCoordenadas cc) {
    this.cc = cc;
    this.sc = cc.getControl().getPuerto();
  }

  public void setMaxMedidas(int maxMedidas) {
    this.maxMedidas = maxMedidas;
  }

  public void setMinDistOperativa(double minDistOperativa) {
    this.minDistOperativa = minDistOperativa;
  }

  private void calculaLLA(double latitud, double longitud, double altura) {
    vectorMedia.add(new double[] { latitud, longitud, altura });

    while(vectorMedia.size() > maxMedidas)
      vectorMedia.remove(0);

    double sumaLat = 0, sumaLong = 0, sumaAlt = 0;
    for (int i = 0; i < vectorMedia.size(); i++) {
      double valor[] = (double[])vectorMedia.elementAt(i);
      sumaLat += valor[0];
      sumaLong += valor[1];
      sumaAlt += valor[2];
    }
    this.latitud = sumaLat / vectorMedia.size();
    this.longitud = sumaLong / vectorMedia.size();
    this.altura = sumaAlt / vectorMedia.size();
  }

  public void run() {
      while(true) {
        try {
            if (write) {
                osECEF.writeDouble(x);
                osECEF.writeDouble(y);
                osECEF.writeDouble(z);
                osECEF.writeDouble(angulo);
                osECEF.writeDouble(speed);
                bw.write("(" + x + ", " + y + ", " + z + ")\n");
                System.out.println("Escribiendo: (" + x + ", " + y + ", " + z + ")");
                if (ci1 != null) {
                    String nombre = ruta + "\\Imagen" + index + "a.jpg";
                    System.out.println(nombre);
                    ci1.saveImagen(nombre);
                    //ImagenId ii = ci1.getImagen(x, y, z);
                }
                if (ci2 != null) {
                    String nombre = ruta + "\\Imagen" + index + "b.jpg";
                    System.out.println(nombre);
                    ci2.saveImagen(nombre);
                }
                if ((ci1 != null) || (ci2 != null))
                    index++;
            }
            Thread.sleep(1000);
        } catch(Exception e) {}
      }
  }

  public double[][] getVCaptura() {
      int n = vCaptura.size();
      double retorno[][] = new double[vCaptura.size()][];
      for (int i = 0; i < n; i++) {
          double xyz[] = (double [])vCaptura.elementAt(i);
          retorno[i] = xyz;
      }

      return retorno;
  }
}

