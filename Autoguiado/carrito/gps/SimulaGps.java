package carrito.gps;

import java.io.*;
import java.util.*;
import javax.comm.*;

import carrito.server.serial.*;

public class SimulaGps implements Runnable {
  private SerialParameters parameters;
  private OutputStream os;
  private CommPortIdentifier portId;
  private SerialPort sPort;
  private boolean open;
  
  private static boolean simulando = false;

  // Vectores del GPS
  private Vector vCadena = new Vector();
  private Vector vTipoPaquete = new Vector();
  private Vector vTiempo = new Vector();
  private Vector vPdop = new Vector();
  private Vector vHdop = new Vector();
  private Vector vVdop = new Vector();
  private Vector vRms = new Vector();
  private Vector vDesvEjeMayor = new Vector();
  private Vector vDesvEjeMenor = new Vector();
  private Vector vOrientacionMayor = new Vector();
  private Vector vDesvLatitud = new Vector();
  private Vector vDesvLongitud = new Vector();
  private Vector vDesvAltura = new Vector();
  private Vector vHdgPoloN = new Vector();
  private Vector vHdgPoloM = new Vector();
  private Vector vSpeed = new Vector();
  private Vector vHora = new Vector();
  private Vector vLatitud = new Vector();
  private Vector vLongitud = new Vector();
  private Vector vAltura = new Vector();
  private Vector vX = new Vector();
  private Vector vY = new Vector();
  private Vector vZ = new Vector();
  private Vector vAngulo = new Vector();
  private Vector vLatitudG = new Vector();
  private Vector vLongitudG = new Vector();
  private Vector vSatelites = new Vector();
  private Vector vMsl = new Vector();
  private Vector vHGeoide = new Vector();
  private Vector vAge = new Vector();

  public SimulaGps() {}

  public SimulaGps(String portName) {
    parameters = new SerialParameters(portName, 9600, 0, 0, 8, 1, 0);
    try {
      openConnection();
    }
    catch (SerialConnectionException e2) {

      System.out.println("Error al abrir el puerto " + portName);
      System.out.flush();
      System.exit(1);
    }
    if (isOpen()) {
      System.out.println("Puerto Abierto BaudRate " + portName);
    }
  }

  public void openConnection() throws SerialConnectionException {

    // Obtain a CommPortIdentifier object for the port you want to open.
    try {
      portId =
          CommPortIdentifier.getPortIdentifier(parameters.getPortName());
    }
    catch (NoSuchPortException e) {
      throw new SerialConnectionException(e.getMessage());
    }

    // Open the port represented by the CommPortIdentifier object. Give
    // the open call a relatively long timeout of 30 seconds to allow
    // a different application to reliquish the port if the user
    // wants to.
    try {
      sPort = (SerialPort) portId.open("SerialDemo", 30000);
    }
    catch (PortInUseException e) {
      throw new SerialConnectionException(e.getMessage());
    }

    // Set the parameters of the connection. If they won't set, close the
    // port before throwing an exception.
    try {
      setConnectionParameters();
    }
    catch (SerialConnectionException e) {
      sPort.close();
      throw e;
    }

    // Open the input and output streams for the connection. If they won't
    // open, close the port before throwing an exception.
    try {
      os = sPort.getOutputStream();
    }
    catch (IOException e) {
      sPort.close();
      throw new SerialConnectionException("Error opening i/o streams");
    }

    open = true;
  }

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
    }
    catch (UnsupportedCommOperationException e) {
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
    }
    catch (UnsupportedCommOperationException e) {
      throw new SerialConnectionException("Unsupported flow control");
    }
  }
  
  public boolean isOpen() {
    return open;
  }

  public double[] getAge() {
    double retorno[] = new double[vAge.size()];
    for (int i = 0; i < retorno.length; i++) {
      retorno[i] = ((Double)vAge.elementAt(i)).doubleValue();
    }
    return retorno;
  }

  public double[] getAltura() {
    double retorno[] = new double[vAltura.size()];
    for (int i = 0; i < retorno.length; i++) {
      retorno[i] = ((Double)vAltura.elementAt(i)).doubleValue();
    }
    return retorno;
  }

  public double[] getAngulo() {
    double retorno[] = new double[vAngulo.size()];
    for (int i = 0; i < retorno.length; i++) {
      retorno[i] = ((Double)vAngulo.elementAt(i)).doubleValue();
    }
    return retorno;

  }

  public String[] getCadena() {
    String retorno[] = new String[vCadena.size()];
    for (int i = 0; i < retorno.length; i++) {
      retorno[i] = (String)vCadena.elementAt(i);
    }
    return retorno;  
  }

  public double[] getDesvAltura() {
    double retorno[] = new double[vDesvAltura.size()];
    for (int i = 0; i < retorno.length; i++) {
      retorno[i] = ((Double)vDesvAltura.elementAt(i)).doubleValue();
    }
    return retorno;

  }

  public double[] getDesvEjeMayor() {
    double retorno[] = new double[vDesvEjeMayor.size()];
    for (int i = 0; i < retorno.length; i++) {
      retorno[i] = ((Double)vDesvEjeMayor.elementAt(i)).doubleValue();
    }
    return retorno;

  }

  public double[] getDesvEjeMenor() {
    double retorno[] = new double[vDesvEjeMenor.size()];
    for (int i = 0; i < retorno.length; i++) {
      retorno[i] = ((Double)vDesvEjeMenor.elementAt(i)).doubleValue();
    }
    return retorno;

  }

  public double[] getDesvLatitud() {
    double retorno[] = new double[vDesvLatitud.size()];
    for (int i = 0; i < retorno.length; i++) {
      retorno[i] = ((Double)vDesvLatitud.elementAt(i)).doubleValue();
    }
    return retorno;

  }

  public double[] getDesvLongitud() {
    double retorno[] = new double[vDesvLongitud.size()];
    for (int i = 0; i < retorno.length; i++) {
      retorno[i] = ((Double)vDesvLongitud.elementAt(i)).doubleValue();
    }
    return retorno;

  }

  public double[] getHdgPoloM() {
    double retorno[] = new double[vHdgPoloM.size()];
    for (int i = 0; i < retorno.length; i++) {
      retorno[i] = ((Double)vHdgPoloM.elementAt(i)).doubleValue();
    }
    return retorno;

  }

  public double[] getHdgPoloN() {
    double retorno[] = new double[vHdgPoloN.size()];
    for (int i = 0; i < retorno.length; i++) {
      retorno[i] = ((Double)vHdgPoloN.elementAt(i)).doubleValue();
    }
    return retorno;

  }

  public double[] getHdop() {
    double retorno[] = new double[vHdop.size()];
    for (int i = 0; i < retorno.length; i++) {
      retorno[i] = ((Double)vHdop.elementAt(i)).doubleValue();
    }
    return retorno;

  }

  public double[] getHGeoide() {
    double retorno[] = new double[vHGeoide.size()];
    for (int i = 0; i < retorno.length; i++) {
      retorno[i] = ((Double)vHGeoide.elementAt(i)).doubleValue();
    }
    return retorno;

  }

  public String[] getHora() {
    String retorno[] = new String[vHora.size()];
    for (int i = 0; i < retorno.length; i++) {
      retorno[i] = (String)vHora.elementAt(i);
    }
    return retorno;  
  }

  public double[] getLatitud() {
    double retorno[] = new double[vLatitud.size()];
    for (int i = 0; i < retorno.length; i++) {
      retorno[i] = ((Double)vLatitud.elementAt(i)).doubleValue();
    }
    return retorno;

  }

  public String[] getLatitudG() {
    String retorno[] = new String[vLatitudG.size()];
    for (int i = 0; i < retorno.length; i++) {
      retorno[i] = (String)vLatitudG.elementAt(i);
    }
    return retorno;  
  }

  public double[] getLongitud() {
    double retorno[] = new double[vLongitud.size()];
    for (int i = 0; i < retorno.length; i++) {
      retorno[i] = ((Double)vLongitud.elementAt(i)).doubleValue();
    }
    return retorno;
  }

  public String[] getLongitudG() {
    String retorno[] = new String[vLongitudG.size()];
    for (int i = 0; i < retorno.length; i++) {
      retorno[i] = (String)vLongitudG.elementAt(i);
    }
    return retorno;  
  }

  public double[] getMsl() {
    double retorno[] = new double[vMsl.size()];
    for (int i = 0; i < retorno.length; i++) {
      retorno[i] = ((Double)vMsl.elementAt(i)).doubleValue();
    }
    return retorno;

  }

  public double[] getOrientacionMayor() {
    double retorno[] = new double[vOrientacionMayor.size()];
    for (int i = 0; i < retorno.length; i++) {
      retorno[i] = ((Double)vOrientacionMayor.elementAt(i)).doubleValue();
    }
    return retorno;

  }

  public double[] getPdop() {
    double retorno[] = new double[vPdop.size()];
    for (int i = 0; i < retorno.length; i++) {
      retorno[i] = ((Double)vPdop.elementAt(i)).doubleValue();
    }
    return retorno;

  }

  public double[] getRms() {
    double retorno[] = new double[vRms.size()];
    for (int i = 0; i < retorno.length; i++) {
      retorno[i] = ((Double)vRms.elementAt(i)).doubleValue();
    }
    return retorno;

  }

  public int[] getSatelites() {
    int retorno[] = new int[vSatelites.size()];
    for (int i = 0; i < retorno.length; i++) {
      retorno[i] = ((Integer)vSatelites.elementAt(i)).intValue();
    }
    return retorno;

  }

  public double[] getSpeed() {
    double retorno[] = new double[vSpeed.size()];
    for (int i = 0; i < retorno.length; i++) {
      retorno[i] = ((Double)vSpeed.elementAt(i)).doubleValue();
    }
    return retorno;

  }

  public long[] getTiempo() {
    long retorno[] = new long[vTiempo.size()];
    for (int i = 0; i < retorno.length; i++) {
      retorno[i] = ((Long)vTiempo.elementAt(i)).longValue();
    }
    return retorno;

  }

  public String[] getTipoPaquete() {
    String retorno[] = new String[vTipoPaquete.size()];
    for (int i = 0; i < retorno.length; i++) {
      retorno[i] = (String)vTipoPaquete.elementAt(i);
    }
    return retorno;  

  }

  public double[] getVdop() {
    double retorno[] = new double[vVdop.size()];
    for (int i = 0; i < retorno.length; i++) {
      retorno[i] = ((Double)vVdop.elementAt(i)).doubleValue();
    }
    return retorno;

  }

  public double[] getX() {
    double retorno[] = new double[vX.size()];
    for (int i = 0; i < retorno.length; i++) {
      retorno[i] = ((Double)vX.elementAt(i)).doubleValue();
    }
    return retorno;

  }

  public double[] getY() {
    double retorno[] = new double[vY.size()];
    for (int i = 0; i < retorno.length; i++) {
      retorno[i] = ((Double)vY.elementAt(i)).doubleValue();
    }
    return retorno;

  }

  public double[] getZ() {
    double retorno[] = new double[vZ.size()];
    for (int i = 0; i < retorno.length; i++) {
      retorno[i] = ((Double)vZ.elementAt(i)).doubleValue();
    }
    return retorno;

  }

  public void loadDatos(String fichero) {
    try {
      File fich = new File(fichero);
      ObjectInputStream is = new ObjectInputStream(new FileInputStream(fich));
      while (is.available() != 0) {                
        vCadena.add(is.readUTF());        
        vTipoPaquete.add(is.readUTF());
        System.out.println(vTipoPaquete.lastElement());
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
    } catch(FileNotFoundException fnfe) {
      System.err.println("Fichero inexistente: " + fnfe.getMessage());
    } catch(IOException ioe) {
      System.err.println("Error de E/S: " + ioe.getMessage());
    }
  }
  
  public void run() {
      int i = 0;
      try {
        os.flush();
      } catch(IOException ioe) {}
      while(simulando) {
          try {
              Thread.sleep(((Long)vTiempo.elementAt(i)).longValue());              
          } catch(Exception e) {}
          String cadena = (String)vCadena.elementAt(i);
          try {
            for (int j = 0; j < cadena.length(); j++) {                
               os.write(cadena.charAt(j));
               //System.out.print(cadena.charAt(j));
            }
            os.write((char)10);
            //System.out.println((char)10);
          } catch(IOException ioe) {
              System.err.println("ERROR: " + ioe.getMessage());
          }
                  
          i = (i + 1) % vCadena.size();
      }
  }
  
  public void startSimulacion() {
      if (os == null)
          return;
      simulando = true;
      Thread hilo = new Thread(this);
      hilo.start();
  }
  
  public void stopSimulacion() {
      simulando = false;
  }

  public static void main(String[] args) {
    SimulaGps simulagps = new SimulaGps();
    simulagps.loadDatos("C:\\GPS\\Integracion\\build\\classes\\nov19.gps");
    double[] x = simulagps.getX();
    for (int i = 0; i < x.length; i++) {
        System.out.println(x[i]);        
    }
    simulagps.startSimulacion();
  }
}
