package carrito.gps;

import java.io.*;
import java.util.*;
import javax.comm.*;

import carrito.server.serial.*;

public class SimulaGps {
  private SerialParameters parameters;
  private OutputStream os;
  private CommPortIdentifier portId;
  private SerialPort sPort;
  private boolean open;

  // Vectores del GPS
  Vector vCadena = new Vector();
  Vector vTipoPaquete = new Vector();
  Vector vTiempo = new Vector();
  Vector vPdop = new Vector();
  Vector vHdop = new Vector();
  Vector vVdop = new Vector();
  Vector vRms = new Vector();
  Vector vDesvEjeMayor = new Vector();
  Vector vDesvEjeMenor = new Vector();
  Vector vOrientacionMayor = new Vector();
  Vector vDesvLatitud = new Vector();
  Vector vDesvLongitud = new Vector();
  Vector vDesvAltura = new Vector();
  Vector vHdgPoloN = new Vector();
  Vector vHdgPoloM = new Vector();
  Vector vSpeed = new Vector();
  Vector vHora = new Vector();
  Vector vLatitud = new Vector();
  Vector vLongitud = new Vector();
  Vector vAltura = new Vector();
  Vector vX = new Vector();
  Vector vY = new Vector();
  Vector vZ = new Vector();
  Vector vAngulo = new Vector();
  Vector vLatitudG = new Vector();
  Vector vLongitudG = new Vector();
  Vector vSatelites = new Vector();
  Vector vMsl = new Vector();
  Vector vHGeoide = new Vector();
  Vector vAge = new Vector();
  private Vector VAge;
  private Vector VAltura;
  private Vector VAngulo;
  private Vector VCadena;
  private Vector VDesvAltura;
  private Vector VDesvEjeMayor;
  private Vector VDesvEjeMenor;
  private Vector VDesvLatitud;
  private Vector VDesvLongitud;
  private Vector VHdgPoloM;
  private Vector VHdgPoloN;
  private Vector VHdop;
  private Vector VHGeoide;
  private Vector VHora;
  private Vector VLatitud;
  private Vector VLatitudG;
  private Vector VLongitud;
  private Vector VLongitudG;
  private Vector VMsl;
  private Vector VOrientacionMayor;
  private Vector VPdop;
  private Vector VRms;
  private Vector VSatelites;
  private Vector VSpeed;
  private Vector VTiempo;
  private Vector VTipoPaquete;
  private Vector VVdop;
  private Vector VX;
  private Vector VY;
  private Vector VZ;

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
    return (String[])vCadena.toArray();
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
    return (String[])vHora.toArray();
  }

  public double[] getLatitud() {
    double retorno[] = new double[vLatitud.size()];
    for (int i = 0; i < retorno.length; i++) {
      retorno[i] = ((Double)vLatitud.elementAt(i)).doubleValue();
    }
    return retorno;

  }

  public String[] getLatitudG() {
    return (String[])vLatitudG.toArray();
  }

  public double[] getLongitud() {
    double retorno[] = new double[vLongitud.size()];
    for (int i = 0; i < retorno.length; i++) {
      retorno[i] = ((Double)vLongitud.elementAt(i)).doubleValue();
    }
    return retorno;
  }

  public String[] getLongitudG() {
    return (String[])vLongitudG.toArray();
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
    return (String[])vTipoPaquete.toArray();
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
      ObjectInputStream is = new ObjectInputStream(new FileInputStream(fichero));
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
        vHora.add(is.readDouble());
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
    } catch(FileNotFoundException fnfe) {
      System.err.println("Fichero inexistente: " + fnfe.getMessage());
    } catch(IOException ioe) {
      System.err.println("Error de E/S: " + ioe.getMessage());
    }
  }

  public static void main(String[] args) {
    SimulaGps simulagps = new SimulaGps("COM1");
  }
}
