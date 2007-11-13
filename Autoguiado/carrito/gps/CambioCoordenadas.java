package carrito.gps;


import java.io.*;
import java.util.*;
import javax.comm.*;

import javax.swing.*;

import Jama.*;
import carrito.media.*;
import carrito.server.serial.*;

public class CambioCoordenadas implements Runnable {
  // Nuevo eje
  //private double v[][] = { { 1.0f, 0.0f, 0.0f}, { 0.0f, 1.0f, 0.0f }, { 1.0f, 0.0f, 0.0f } };
  private double v[][] = { { 0.5265236863206882f, 0.4953194619879353f, 0.6909641367822614f },
      { 0.28748583110525694f, 0.6611338861623393f, -0.6930035219835485f },
      { 0.8000779366142983f, -0.563525168154773f, -0.2057053237015069f } };

  // Origen del eje original desde el nuevo eje
  //private double origen[] = { 0, 0, 0 };
  private double origen[] = { 5105873.521765788f, -3589966.553928558f, -1311074.7639646954f };

  // Puntos que definen el plano
  private double p[] = { 1, 0, 0 };
  private double q[] = { 0, 1, 0 };
  private double r[] = { 0, 0, 1 };

  // Matriz de transformación
  Matrix T = null;

  // Límites del mapa
  private double limites[] = { 10, 10, 320, 240 };

  // Objetos de la interfaz
  private CanvasRuta canvas = null;
  private JFrame frmRuta = new JFrame("Ruta del vehículo");

  // Conexión con el GPS
  private GPSConnection gps = null;

  // Control del Coche
  ControlCarro control = null;
  boolean cocheActivo = false;

  // Array que contiene las coordenadas y valores en el plano de la ruta
  private double ruta[] = null;
  private double rutaECEF[][] = null;
  private double rutaLLA[][] = null;
  private double rms[] = null;
  private double velocidades[] = null;
  private double angulos[] = null;
  private ImagenId[] imagenes1a = null;
  private ImagenId[] imagenes1b = null;
  private ImagenId[] imagenes2a = null;
  private ImagenId[] imagenes2b = null;

  // Array que contiene los obstáculos
  Hashtable obstaculos = null;

  // Número de polígonos existentes
  private int poligonos = 0;

  // Último cercano
  private int lastCerca = 0;

  private int MAX_CHECK = 50;

  private int nextPos = 8;  // 2 por defecto

  // Variables del hilo
  private static boolean independiente = false;
  private static long refresco = 0;
  private static CambioCoordenadas cc = null;

  // Punto base de control de la distancia
  private double px = 0, py = 0, pz = 0;

  // Constantes necesarias para afinar el control del vehículo
  double k1 = 1;           // Distancia
  double k2 = 1;           // Diferencia angular
  double k3 = 1;
  double k4 = 1;

  double kAngulo = 1;
  double kAlfa = 1;
  double kConsigna = 1;

  private double kDer = 0;
  private double kIntegral = 0;

  private double kVel = ControlCarro.CARRO_DIST / 255;
  private double kAlfa2 = 1;
  private double kDer2 = 0;
  private double kIntegral2 = 0;

  String dllpath = "";

  String imagenes = null;

  Media media = null;

  double vDirector[] = null;

  public CambioCoordenadas(String puerto, String params, String puertoCoche, boolean cocheActivo) {
      loadParams(params);
    if (puerto.equals("")) {
      gps = new GPSConnection();
    } else {
      gps = new GPSConnection(puerto);
    }
    cc = this;

    if ((puertoCoche != null) && (! puertoCoche.equals("")))
      control = new ControlCarro(puertoCoche);
    else
      control = new ControlCarro();
    this.cocheActivo = cocheActivo;
    gps.setCc(this);
  }

  public CambioCoordenadas(String puerto, String params, ControlCarro control, boolean cocheActivo) {
    loadParams(params);
    if (puerto.equals("")) {
      gps = new GPSConnection();
    } else {
      gps = new GPSConnection(puerto);
      gps.setCc(this);
    }
    cc = this;

    if (control != null)
      this.control = control;
    this.cocheActivo = cocheActivo;
  }

  public double[] cambioCoordenadas(double x, double y, double z) {
    double resultado[] = null;

    Matrix res = new Matrix(new double[][] { {x - origen[0]}, {y - origen[1]}, {z - origen[2]} });
    resultado = (T.times(res).transpose().getArray())[0];

    return resultado;
  }

  public static double[] cambioCoordenadas(double x, double y, double z, Matrix m, double coord[]) {
    double resultado[] = null;

    Matrix res = new Matrix(new double[][] { {x - coord[0]}, {y - coord[1]}, {z - coord[2]} });
    resultado = (m.times(res).transpose().getArray())[0];

    return resultado;
  }


  /*public double[] cambioCoordenadas(double x, double y, double z) {
      // Pasamos de coordenadas ECEF a LLA
      double coord[] = GPSConnection.ECEF2LLA(x, y, z);
      double latitud = coord[0];
      double longitud = coord[1];
      x = longitud + 1.0725693203475835d;
      y = Math.log(Math.tan(Math.PI / 4 + latitud / 2));

      return new double[] { x, y, 0 };
  }*/

  public void setOrigen(double[] origen) {
    this.origen = origen;
  }

  public void setOrigen() {
    GPSData data = gps.getECEF();
    origen = new double[] { data.getX(), data.getY(), data.getZ() };
  }

  public void setV(double[][] v) {
    this.v = v;
  }

  public void setK1(double k1) {
    this.k1 = k1;
  }

  public void setK2(double k2) {
    this.k2 = k2;
  }

  public void setK3(double k3) {
    this.k3 = k3;
  }

  public void setK4(double k4) {
    this.k4 = k4;
  }

  public void setKVel(double kVel) {
    this.kVel = kVel;
  }

  public void setk9Vel(double kVel) {
    this.kVel = kVel;
  }

  public void setKAlfa2(double kAlfa2) {
    this.kAlfa2 = kAlfa2;
  }

  public void setKIntegral2(double kIntegral2) {
    this.kIntegral2 = kIntegral2;
  }

  public void setKDer2(double kDer2) {
    this.kDer2 = kDer2;
  }

  public void setLastCerca(int lastCerca) {
    this.lastCerca = lastCerca;
  }

  public void setKAngulo(double KAngulo) {
    this.kAngulo = KAngulo;
  }

  public void setP() {
    GPSData data = gps.getECEF();
    this.p = new double[] { data.getX(), data.getY(), data.getZ() };
  }

  public void setQ() {
    GPSData data = gps.getGPSData();
    this.q = new double[] { data.getX(), data.getY(), data.getZ() };
  }

  public void setR() {
    GPSData data = gps.getECEF();
    this.r = new double[] { data.getX(), data.getY(), data.getZ() };
  }

  public void saveParams() {
    try {
      File file = new File("params.dat");
      ObjectOutputStream os = new ObjectOutputStream(new FileOutputStream(file, false));
      os.writeObject(v);
      os.writeObject(origen);
      if (limites[0] > limites[2]) {
        double tmp = limites[0];
        limites[0] = limites[2];
        limites[2] = tmp;
      }
      if (limites[1] > limites[3]) {
        double tmp = limites[1];
        limites[1] = limites[3];
        limites[3] = tmp;
      }
      os.writeObject(limites);
      os.close();
    } catch (Exception e) {
      System.out.println("Error al cargar los parámetros desde el fichero");
    }

  }

  public void loadParams(String fichero) {
    try {
      File file = new File(fichero);
      ObjectInputStream is = new ObjectInputStream(new FileInputStream(file));
      v = (double[][])is.readObject();
      origen = (double[])is.readObject();
      limites = (double[])is.readObject();

      if (limites[0] > limites[2]) {
        double tmp = limites[2];
        limites[2] = limites[0];
        limites[0] = tmp;
      } else if (limites[1] > limites[3]) {
        double tmp = limites[3];
        limites[3] = limites[1];
        limites[1] = tmp;
      }

      T = new Matrix(v);
      T = T.transpose().inverse();
      is.close();
    } catch (Exception e) {
      System.out.println("Error al cargar los parámetros desde el fichero");
    }
  }

  public void loadRuta2(String fichero) {
    Vector v = new Vector();
    Vector valores = new Vector();
    try {
      File file = new File(fichero);
      ObjectInputStream is = new ObjectInputStream(new FileInputStream(file));
      double x = 0, y = 0, z = 0;
      double xy[] = null;
      while (is.available() != 0) {
        x = is.readDouble();
        y = is.readDouble();
        z = is.readDouble();

        xy = cambioCoordenadas(x, y, z);

        v.add(xy);

        // Lee el ángulo
        valores.add(is.readDouble());
        // Lee la velocidad
        valores.add(is.readDouble());
      }
      is.close();
    } catch (Exception e) {
      System.out.println("Error al cargar la ruta desde el fichero " + e.getMessage());
    }

    double ruta[] = new double[v.size() * 2];
    for (int i = 0; i < v.size(); i++) {
      ruta[i * 2] = ((double[])v.elementAt(i))[0];
      ruta[(i * 2) + 1] = ((double[])v.elementAt(i))[1];
    }


    double velocidades[] = new double[valores.size() / 2];
    double angulos[] = new double[valores.size() / 2];
    for (int i = 0; i < valores.size(); i += 2) {
      angulos[i / 2] = ((Double)valores.elementAt(i)).doubleValue();
      velocidades[i / 2] = ((Double)valores.elementAt(i + 1)).doubleValue();
    }

    canvas.setRuta2(ruta, angulos, velocidades);

  }

  public void loadRuta(String fichero, boolean conImag) {
    if (conImag) {
      if (media == null)
        media = new Media(dllpath);
    }
    Vector v = new Vector();
    Vector valores = new Vector();
    Vector vRms = new Vector();
    try {
      File file = new File(fichero);
      ObjectInputStream is = new ObjectInputStream(new FileInputStream(file));
      double x = 0, y = 0, z = 0;
      while (is.available() != 0) {
        x = is.readDouble();
        y = is.readDouble();
        z = is.readDouble();

        v.add(new double[]{x, y, z});

        // Lee el ángulo
        valores.add(is.readDouble());
        // Lee la velocidad
        valores.add(is.readDouble());

        //vRms.add(is.readDouble());
      }
      is.close();

      if (conImag)
        imagenes = fichero.substring(0, fichero.length() - 4);
      else
        imagenes = null;
    } catch (Exception e) {
      System.out.println("Error al cargar la ruta desde el fichero " + e.getMessage());
    }

    rutaECEF = new double[v.size()][3];
    rutaLLA = new double[v.size()][3];
    for (int i = 0; i < v.size(); i++) {
      double elem[] = (double[])v.elementAt(i);
      double LLA[] = gps.ECEF2LLA(elem[0], elem[1], elem[2]);

      for (int j = 0; j < 3; j++) {
        rutaECEF[i][j] = elem[j];
        rutaLLA[i][j] = LLA[j];
      }
    }

    setParams();

    ruta = new double[v.size() * 2];
    for (int i = 0; i < v.size(); i++) {
      double elem[] = (double[])v.elementAt(i);
      double xy[] = this.cambioCoordenadas(elem[0], elem[1], elem[2]);

      ruta[i * 2] = xy[0];
      ruta[(i * 2) + 1] = xy[1];
    }

    System.out.println("Puntos: " + ruta.length / 2);

    velocidades = new double[valores.size() / 2];
    angulos = new double[valores.size() / 2];
    rms = new double[vRms.size()];
    /*for (int i = 0; i < valores.size(); i += 2) {
      velocidades[i / 2] = ((Double)valores.elementAt(i + 1)).doubleValue();
    }*/

    for (int i = 0; i < vRms.size(); i++) {
        rms[i] = ((Double)vRms.elementAt(i)).doubleValue();
    }

    angulos[0] = 0;
    for (int i = 1; i < angulos.length; i++) {
      if (Math.sqrt(Math.pow(rutaECEF[i - 1][0] - rutaECEF[i][0], 2.0f) +
                    Math.pow(rutaECEF[i - 1][1] - rutaECEF[i][1], 2.0f) +
                    Math.pow(rutaECEF[i - 1][2] - rutaECEF[i][2], 2.0f)) < GPSConnection.minDistOperativa) {
        angulos[i] = angulos[i - 1];
        continue;
      }

      double val[] = calculaAnguloVel(rutaECEF[i-1], rutaECEF[i], rutaLLA[i][0], rutaLLA[i][1]);
      angulos[i] = val[0];
      velocidades[i] = val[1];
    }
    angulos[0] = angulos[1];

    if (canvas == null) {
        canvas = new CanvasRuta(ruta, angulos, velocidades, this, 800, 600);
        control.setCanvas(canvas);
    } else {
        //canvas = new CanvasRuta(ruta, angulos, velocidades, this, 800, 600);
        canvas.setRuta(ruta, angulos, velocidades, 800, 600);
    }

  }

  public double[] testRuta(String fichero) {
    Vector r = new Vector();
    Vector a = new Vector();
    Vector v = new Vector();

    try {
      File file = new File(fichero);
      ObjectInputStream is = new ObjectInputStream(new FileInputStream(file));
      double x = 0, y = 0, z = 0;
      while (is.available() != 0) {
        x = is.readDouble();
        y = is.readDouble();
        z = is.readDouble();

        r.add(new double[]{x, y, z});

        // Lee el ángulo
        a.add(is.readDouble());
        // Lee la velocidad
        v.add(is.readDouble());
      }
      is.close();
    } catch (Exception e) {
      System.out.println("Error al cargar la ruta desde el fichero " + e.getMessage());
    }

    double ang[] = new double[r.size()];
    for (int i = 0; i < r.size(); i++) {
        double xyz[] = (double[])r.elementAt(i);
        ang[i] = gps.testAngulo(xyz[0], xyz[1], xyz[2]);

        /*try {
            Thread.sleep(200);
        } catch (Exception e) {}*/
    }

    return ang;
  }

  public void loadRutaConImagenes2(String fichero) {
    Vector v = new Vector();
    Vector valores = new Vector();
    Vector img2a = new Vector();
    Vector img2b = new Vector();

    try {
      File file = new File(fichero);
      ObjectInputStream is = new ObjectInputStream(new FileInputStream(file));
      double x = 0, y = 0, z = 0;
      double xy[] = null;
      while (is.available() != 0) {
        x = is.readDouble();
        y = is.readDouble();
        z = is.readDouble();

        xy = cambioCoordenadas(x, y, z);

        v.add(xy);

        // Lee el ángulo
        valores.add(is.readDouble());
        // Lee la velocidad
        valores.add(is.readDouble());

        img2a.add(is.readObject());
        img2b.add(is.readObject());
      }
      is.close();
    } catch (Exception e) {
      System.out.println("Error al cargar la ruta desde el fichero " + e.getMessage());
    }

    double ruta[] = new double[v.size() * 2];
    for (int i = 0; i < v.size(); i++) {
      ruta[i * 2] = ((double[])v.elementAt(i))[0];
      ruta[(i * 2) + 1] = ((double[])v.elementAt(i))[1];
    }


    double velocidades[] = new double[valores.size() / 2];
    double angulos[] = new double[valores.size() / 2];
    for (int i = 0; i < valores.size(); i += 2) {
      angulos[i / 2] = ((Double)valores.elementAt(i)).doubleValue();
      velocidades[i / 2] = ((Double)valores.elementAt(i + 1)).doubleValue();
    }

    imagenes2a = new ImagenId[img2a.size()];
    for (int i = 0; i < img2a.size(); i++) {
      imagenes2a[i] = (ImagenId)img2a.elementAt(i);
    }
    imagenes2b = new ImagenId[img2b.size()];
    for (int i = 0; i < img2b.size(); i++) {
      imagenes2b[i] = (ImagenId)img2b.elementAt(i);
    }

    canvas.setRuta2(ruta, angulos, velocidades);

  }

  public void loadRutaConImagenes(String fichero) {
    Vector v = new Vector();
    Vector valores = new Vector();
    Vector img1a = new Vector();
    Vector img1b = new Vector();

    try {
      File file = new File(fichero);
      ObjectInputStream is = new ObjectInputStream(new FileInputStream(file));
      double x = 0, y = 0, z = 0;
      double xy[] = null;
      while (is.available() != 0) {
        x = is.readDouble();
        y = is.readDouble();
        z = is.readDouble();

        xy = cambioCoordenadas(x, y, z);

        v.add(xy);

        // Lee el ángulo
        valores.add(is.readDouble());
        // Lee la velocidad
        valores.add(is.readDouble());

        img1a.add(is.readObject());
        img1b.add(is.readObject());
      }
      is.close();
    } catch (Exception e) {
      System.out.println("Error al cargar la ruta desde el fichero " + e.getMessage());
    }

    ruta = new double[v.size() * 2];
    for (int i = 0; i < v.size(); i++) {
      ruta[i * 2] = ((double[])v.elementAt(i))[0];
      ruta[(i * 2) + 1] = ((double[])v.elementAt(i))[1];
    }

    velocidades = new double[valores.size() / 2];
    angulos = new double[valores.size() / 2];
    for (int i = 0; i < valores.size(); i += 2) {
      angulos[i / 2] = ((Double)valores.elementAt(i)).doubleValue();
      velocidades[i / 2] = ((Double)valores.elementAt(i + 1)).doubleValue();
    }

    imagenes1a = new ImagenId[img1a.size()];
    for (int i = 0; i < img1a.size(); i++) {
      imagenes1a[i] = (ImagenId)img1a.elementAt(i);
    }
    imagenes1b = new ImagenId[img1b.size()];
    for (int i = 0; i < img1b.size(); i++) {
      imagenes1b[i] = (ImagenId)img1b.elementAt(i);
    }

    canvas = new CanvasRuta(ruta, angulos, velocidades, this, 800, 600);

  }

  public double[] getOrigen() {
    return origen;
  }

  public double[][] getV() {
    for (int i = 0; i < v.length; i++) {
      for (int j = 0; j < v[i].length; j++) {
        System.out.print("[" + v[i][j] + "]");
      }
      System.out.println();
    }
    return v;
  }

  public double[] getP() {
    return p;
  }

  public double[] getQ() {
    return q;
  }

  public double[] getR() {
    return r;
  }

  public double getK1() {
    return k1;
  }

  public double getK2() {
    return k2;
  }

  public double getK3() {
    return k3;
  }

  public double getK4() {
    return k4;
  }

  public int getLastCerca() {
    return lastCerca;
  }

  public double getKAngulo() {
    return kAngulo;
  }

  public double getKAlfa() {
    return kAlfa;
  }

  public GPSConnection getGps() {
    return gps;
  }

  public String getDllpath() {
    return dllpath;
  }

  public ControlCarro getControl() {
    return control;
  }

  public SerialConnection getCoche() {
    return control.getPuerto();
  }

  public double getKConsigna() {
    return kConsigna;
  }

  public double getKDer() {
    return kDer;
  }

  public double getKIntegral() {
    return kIntegral;
  }

  public int getNextPos() {
    return nextPos;
  }

  public CanvasRuta getCanvas() {
    return canvas;
  }

  public double[] getVelocidades() {
    return velocidades;
  }

  public double[] getAngulos() {
    return angulos;
  }

  public boolean isAcelera() {
    if (control != null)
      return control.isAcelera();
    else
      return false;
  }

  public boolean isGira() {
        if (control != null)
          return control.isGira();
        else return false;
  }

  public void savePolygon(boolean nuevo) {
    if (nuevo)
      poligonos++;
    try {
      File file = new File("polygons.dat");
      FileOutputStream fos = new FileOutputStream(file, true);
      ObjectOutputStream pos = new ObjectOutputStream(fos);
      pos.writeInt(poligonos);
      GPSData data = gps.getECEF();
      pos.writeDouble(data.getX());
      pos.writeDouble(data.getY());
      pos.writeDouble(data.getZ());
      System.out.println("Guardando polígono " + poligonos + ": (" + data.getX() + ", " + data.getY() + ", " + data.getZ() + ")");
      pos.close();
    } catch (Exception e) {
      System.out.println("Error al escribir en el fichero " + e.getMessage());
    }

  }

  public void loadPolygon() {
    try {
      FileInputStream fis = new FileInputStream(new File("polygons.dat"));
      ObjectInputStream is = new ObjectInputStream(fis);
      int pos = 0;
      double x = 0, y = 0, z = 0;
      double xy[] = null;
      Vector v = new Vector();
      while (is.available() != 0) {
        pos = is.readInt();
        x = is.readDouble();
        y = is.readDouble();
        z = is.readDouble();
        System.out.println(pos);
        System.out.println(x);
        System.out.println(y);
        System.out.println(z);
        System.out.println(is.available());
        /*xy = cambioCoordenadas(x, y, z);
        Vector obs = (Vector)v.elementAt(pos);
        if (obs == null)
          obs = new Vector();
        obs.add(xy);
        v.insertElementAt(obs, pos);*/
      }

      /*obstaculos = new Vector();
      for (int i = 0; i < v.size(); i++) {
        Vector obs = (Vector)v.elementAt(pos);
        if (obs == null)
          continue;
        double obstaculo[] = new double[obs.size() * 2];
        for (int j = 0; j < obs.size(); j++) {
          obstaculo[j * 2] = ((double[])obs.elementAt(j))[0];
          obstaculo[(j * 2) + 1] = ((double[])obs.elementAt(j))[1];
        }
        obstaculos.add(obstaculo);
      }

      for (int i = 0; i < obstaculos.size(); i++) {
        System.out.print(new Matrix(new double[][] { ((double[])obstaculos.elementAt(i)) }));
      }*/
      is.close();
    } catch (Exception e) {
      System.out.println("Error al cargar los parámetros desde el fichero");
    }
  }

  public void showLLA() {
    GPSData data = gps.getLLA();
    System.out.println(data.getLatitud());
    System.out.println(data.getLongitud());
    System.out.println(data.getAltura());
  }

  public void showECEF() {
    GPSData data = gps.getECEF();
    System.out.println(data.getX());
    System.out.println(data.getY());
    System.out.println(data.getZ());
  }

  // Obtiene la matriz del plano tangente a un punto
  public static Matrix getPTP(double latitud, double longitud) {
    // Matriz de rotación en torno a un punto
    double v[][] = new double[3][];
    v[0] = new double[] { -Math.sin(longitud), Math.cos(longitud), 0 };
    v[1] = new double[] { -Math.cos(longitud) * Math.sin(latitud), -Math.sin(latitud) * Math.sin(longitud), Math.cos(latitud) };
    v[2] = new double[] { Math.cos(latitud) * Math.cos(longitud), Math.cos(latitud) * Math.sin(longitud), Math.sin(latitud)};

    Matrix M1 = new Matrix(v);

    // Matriz de inversión del eje z en torno al eje x (Norte)
    double w[][] = new double[3][];
    w[0] = new double[] { -1, 0, 0 };
    w[1] = new double[] { 0, 1, 0 };
    w[2] = new double[] { 0, 0, -1 };
    Matrix M2 = new Matrix(w);

    return M2.times(M1);
  }

  public void setParams() {
    // Buscamos el centro de la ruta
    origen = new double[]{ 0, 0, 0 };
    for (int i = 0; i < rutaECEF.length; i++) {
      for (int j = 0; j < 3; j++) {
        origen[j] += rutaECEF[i][j];
      }
    }
    for (int j = 0; j < 3; j++) {
      origen[j] /= rutaECEF.length;
    }

    double coord[] = gps.ECEF2LLA(origen[0], origen[1], origen[2]);

    T = getPTP(coord[0], coord[1]);
  }

  public void setParams(double[] p, double[] q, double[] r) {
    double a[] = { q[0] - p[0], q[1] - p[1], q[2] - p[2] };
    double b[] = { r[0] - p[0], r[1] - p[1], r[2] - p[2] };

    System.out.println("A = [" + a[0] + ", " + a[1] + ", " + a[2] + "]");
    System.out.println("B = [" + b[0] + ", " + b[1] + ", " + b[2] + "]");

    double x[] = a;
    double z[] = {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
    double y[] = {
        x[1] * z[2] - x[2] * z[1],
        x[2] * z[0] - x[0] * z[2],
        x[0] * z[1] - x[1] * z[0]
    };

    double kx = Math.sqrt(Math.pow(x[0], 2) + Math.pow(x[1], 2) + Math.pow(x[2], 2));
    double ky = Math.sqrt(Math.pow(y[0], 2) + Math.pow(y[1], 2) + Math.pow(y[2], 2));
    double kz = Math.sqrt(Math.pow(z[0], 2) + Math.pow(z[1], 2) + Math.pow(z[2], 2));

    for (int i = 0; i < 3; i++) {
      x[i] /= kx;
      y[i] /= ky;
      z[i] /= kz;
    }

    System.out.println("[0, " + x[0] + "], [0, " + x[1] + "], [0, " + x[2] + "]");
    System.out.println("[0, " + y[0] + "], [0, " + y[1] + "], [0, " + y[2] + "]");
    System.out.println("[0, " + z[0] + "], [0, " + z[1] + "], [0, " + z[2] + "]");

    v = new double[][] { x, y, z };

    T = new Matrix(v);
    T = T.transpose().inverse();

    origen = new double[] { p[0], p[1], p[2] };

    if (cambioCoordenadas(0, 0, 0)[2] < 0)
        setParams(q, p, r);
  }

  public static double[] calculaAnguloVel(double punto1[], double punto2[], double latitud, double longitud) {
    Matrix m = getPTP(latitud, longitud);

    double v[] = CambioCoordenadas.cambioCoordenadas(punto1[0], punto1[1], punto1[2], m, new double[] { punto2[0], punto2[1], punto2[2] });

    double angulo = Math.atan2(v[1], v[0]);
    if (angulo < 0)
      angulo += 2 * Math.PI;

    double vel = Math.sqrt(Math.pow(v[0], 2.0f) + Math.pow(v[1], 2.0f));

    return new double[] { angulo, vel };
  }

  public void savePunto(String id) {
    GPSData data = gps.getGPSData();
    try {
      System.out.println("Guardando punto clave " + id + ": (" + data.getLatitud() + ", " + data.getLongitud() + ", " + data.getAltura() + ")");
      System.out.println("\t\t" + id + ": (" + data.getX() + ", " + data.getY() + ", " + data.getZ() + ")");
      BufferedWriter bw = new BufferedWriter(new FileWriter("puntosClave.txt", true));
      bw.write("Punto \"" + id + "\":\n");
      bw.write("\t- Latitud: " + data.getLatitud() + "\n");
      bw.write("\t- Longitud: " + data.getLongitud() + "\n");
      bw.write("\t- Altura: " + data.getAltura() + "\n");
      bw.write("\t- X: " + data.getX() + "\n");
      bw.write("\t- Y: " + data.getY() + "\n");
      bw.write("\t- Z: " + data.getZ() + "\n");
      bw.write("\t- PDOP: " + data.getPdop() + "\n");
      bw.write("\t- HDOP: " + data.getHdop() + "\n");
      bw.write("\t- VDOP: " + data.getVdop() + "\n");
      bw.write("\t- Velocidad: " + data.getSpeed() + "\n");
      bw.write("\t- Satelites: " + data.getSatelites() + "\n");
      bw.write("\t- LatitudG: " + data.getLatitudg() + "\n");
      bw.write("\t- LongitudG: " + data.getLongitudg() + "\n");
      bw.write("\t- MSL: " + data.getMsl() + "\n");
      bw.write("\t- Altura respecto geoide: " + data.getHgeoide() + "\n");
      bw.write("\t- Inclinacion polo Norte: " + data.getHdgPoloN() + "\n");
      bw.write("\t- Inclinacion polo Magnetico: " + data.getHdgPoloM() + "\n");
      bw.write("\t- Hora:  " + data.getHora() + "\n");
      bw.write("\t- Antigüedad de la señal: " + data.getAge() + "\n");
      bw.write("\t- RMS: " + data.getRms() + "\n");
      bw.write("\t- Desviacion respecto al eje mayor: " + data.getDesvEjeMayor() + "\n");
      bw.write("\t- Desviacion respecto al eje menor: " + data.getDesvEjeMenor() + "\n");
      bw.write("\t- Orientacion respecto al eje mayor: " + data.getOrientacionMayor() + "\n");
      bw.write("\t- Desviacion de la latitud: " + data.getDesvLatitud() + "\n");
      bw.write("\t- Desviacion de la longitud: " + data.getDesvLongitud() + "\n");
      bw.write("\t- Desviacion de la altura: " + data.getDesvAltura() + "\n\n");
      bw.close();
    } catch (Exception e) {
      System.out.println("Error al cargar los parámetros desde el fichero");
    }
  }

  public void verTransformacion() {
    GPSData data = gps.getGPSData();
    System.out.println("Latitud: " + data.getLatitud());
    System.out.println("Longitud: " + data.getLongitud());
    System.out.println("Altura: " + data.getAltura());
    System.out.println("******************************");
    System.out.println("X: " + data.getX());
    System.out.println("Y: " + data.getY());
    System.out.println("Z: " + data.getZ());
    System.out.println("******************************");
    double coordenadas[] = cambioCoordenadas(data.getX(), data.getY(), data.getZ());
    System.out.println("Nueva X: " + coordenadas[0]);
    System.out.println("Nueva Y: " + coordenadas[1]);
    System.out.println("Nueva Z: " + coordenadas[2]);
  }

  public void startRuta(String nombre) {
    gps.startReading(nombre);
  }

  public void stopRuta() {
    gps.stopReading();
  }

  public void pauseReading() {
      gps.pauseReading();
  }

  public void startRutaConImagenes(String nombre, int id1, int id2) {
    CapturaImagen ci1 = null, ci2 = null;
    if (id1 >= 0) {
        String disp1 = Media.listaDispositivos()[id1] + ":" + id1;
        ci1 = new CapturaImagen(dllpath, disp1, false);
    }
    if (id2 >= 0) {
        String disp2 = Media.listaDispositivos()[id2] + ":" + id2;
        ci2 = new CapturaImagen(dllpath, disp2, false);
    }
    gps.startReadingImages(nombre, ci1, ci2);
  }

  public void testRutaConImagenes(String libreria, String ruta, int id1, int id2, long tiempo) {
    startCamaras(id1, id2, libreria);

    startRutaConImagenes(ruta, id1, id2);

    Thread hilo = new Thread(gps);
    hilo.start();

    long actual = System.currentTimeMillis();
    while (System.currentTimeMillis() - actual < tiempo) {
    try {
        Thread.sleep(10);
    } catch(Exception e) {}
    }
    stopRutaConImagenes();

  }

  public void stopRutaConImagenes() {
    gps.stopReadingImages();
  }

  public void startRutaBD(String nombre, int id1, int id2) {
    String disp1 = Media.listaDispositivos()[id1] + ":" + id1;
    String disp2 = Media.listaDispositivos()[id2] + ":" + id2;
    gps.startBD(nombre, media, disp1, disp2);
  }

  public void stopRutaBD() {
    gps.stopBD();
  }

  public void setLimite(int i) {
    i *= 2;
    GPSData data = gps.getECEF();
    double xy[] = cambioCoordenadas(data.getX(), data.getY(), data.getZ());
    limites[i] = xy[0];
    limites[i + 1] = xy[1];
  }

  public void showCanvas() {
    if (canvas == null)
      return;
    if (control != null)
      control.setCanvas(canvas);
    frmRuta.getContentPane().add(canvas);
    frmRuta.setSize(canvas.getWidth() + 10, canvas.getHeight() + 30);
    frmRuta.setVisible(true);
    frmRuta.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    frmRuta.setResizable(true);
    frmRuta.addKeyListener(canvas);
    canvas.repaint();
  }

  public void hideCanvas(boolean valor) {
      frmRuta.setVisible(! valor);
  }

  public void conduce(double angulo, double velocidad) {
    control.conduceSolo(angulo, velocidad, kAlfa, kDer, kIntegral, kVel, kAlfa2, kDer2, kIntegral2);
  }

  public double[] setPuntoActual() {
    GPSData data = gps.getGPSData();
    return setPuntoActual(data.getX(), data.getY(), data.getZ(), data.getAngulo(), data.getSpeed());
  }

  public double[] setPuntoActual(double x, double y, double z, double angulo, double speed) {
      double xy[] = cambioCoordenadas(x, y, z);

      double u[] = getU(xy[0], xy[1], angulo, speed);

      if (cocheActivo) {
        conduce(u[0], u[1]);
      } else {
        simulaConduceSolo(u[0], u[1]);
      }

      if (frmRuta.isVisible()) {
        canvas.setPoint(xy[0], xy[1], u, angulo, speed);
      }

      control.getPuerto().setConsignaAvance(u[1]);

      return xy;
  }

  public void simulaPuntoActual(double x, double y, double angulo, double speed) {
    double xy[] = { x , y };

    double u[] = getU(x, y, angulo, speed);

    if (cocheActivo) {
      conduce(u[0], u[1]);
    } else {
        simulaConduceSolo(u[0], u[1]);
    }

    canvas.setPoint(xy[0], xy[1], u, angulo, speed);

  }


  public int setCercano(double xy[]) {
    lastCerca = (lastCerca - nextPos + ruta.length) % ruta.length;
    int minPos = lastCerca;
    double min = Math.sqrt(Math.pow(ruta[lastCerca] - xy[0], 2) + Math.pow(ruta[lastCerca + 1] - xy[1], 2));

    for (int i = 0; i < MAX_CHECK; i++) {
      int j = (i * 2 + lastCerca) % ruta.length;
      double valor = Math.sqrt(Math.pow(ruta[j] - xy[0], 2) + Math.pow(ruta[j + 1] - xy[1], 2));
      if (valor < min) {
        min = valor;
        minPos = j;
      }
    }

    minPos = (minPos + nextPos) % ruta.length;

    System.out.println("Punto: " + (minPos / 2));

    canvas.setCercano(minPos);

    return minPos;
  }

  public void muestraImagenCercana(double x, double y) {
    if (imagenes == null)
      return;

    double min = Math.sqrt(Math.pow(ruta[0] - x, 2) + Math.pow(ruta[1] - y, 2));
    int pos = 0;

    for (int i = 0; i < ruta.length; i += 2) {
      double val = Math.sqrt(Math.pow(ruta[i] - x, 2) + Math.pow(ruta[i + 1] - y, 2));
      if (val < min) {
        min = val;
        pos = i;
      }
    }

    String miImagen1 = imagenes + "\\Imagen" + (pos / 2) + "a.jpg";
    String miImagen2 = imagenes + "\\Imagen" + (pos / 2) + "a.jpg";

    media.loadImagen(miImagen1);
    media.loadImagen(miImagen2);
  }

  public void guardarTodo(String fichero) {
    try {
      File file = new File(fichero);
      ObjectOutputStream os = new ObjectOutputStream(new FileOutputStream(file, false));
      os.writeObject(v);
      os.writeObject(origen);
      os.writeObject(limites);
      os.close();
      BufferedWriter bw = new BufferedWriter(new FileWriter(fichero + ".txt"));
      bw.write("Matriz V:\n");
      for (int i = 0; i < v.length; i++) {
        for (int j = 0; j < v[i].length; j++) {
          bw.write("[" + v[i][j] + "]");
        }
        bw.write("\n");
      }
      bw.write("Origen: ");
      for (int i = 0; i < origen.length; i++) {
        bw.write("[" + origen[i] + "]");
      }
      bw.write("\nLimites: ");
      for (int i = 0; i < limites.length; i++) {
        bw.write("[" + limites[i] + "]");
      }
      bw.write("\n");
      bw.close();
    } catch (Exception e) {
      System.out.println("Error al cargar los parámetros desde el fichero");
    }
  }

  public static void showSerial() {
    Enumeration ports = CommPortIdentifier.getPortIdentifiers();
    CommPortIdentifier port = null;
    while ((port = (CommPortIdentifier) ports.nextElement()) != null) {
      if (port.getPortType() == CommPortIdentifier.PORT_SERIAL) {
        System.out.println(port.getName());
      }
    }
  }

  public void setPuntoBase() {
    GPSData data = gps.getECEF();
    px = data.getX();
    py = data.getY();
    pz = data.getZ();
  }

  public void muestraDistancia() {
    GPSData data = gps.getGPSData();
    double base[] = cambioCoordenadas(px, py, pz);
    double coordenadas[] = cambioCoordenadas(data.getX(), data.getY(), data.getZ());
    System.out.println("PX: " + px + " --> " + base[0]);
    System.out.println("PY: " + py + " --> " + base[1]);
    System.out.println("PZ: " + pz + " --> " + base[2]);
    System.out.println("******************************");
    System.out.println("X: " + data.getX() + " --> " + coordenadas[0]);
    System.out.println("Y: " + data.getY() + " --> " + coordenadas[1]);
    System.out.println("Z: " + data.getZ() + " --> " + coordenadas[2]);
    System.out.println("******************************");
    double distancia = Math.sqrt(Math.pow((coordenadas[0] - base[0]), 2) +
                                 Math.pow((coordenadas[1] - base[1]), 2));
    double distanciaz = Math.sqrt(Math.pow((coordenadas[0] - base[0]), 2) +
                                 Math.pow((coordenadas[1] - base[1]), 2) +
                                 Math.pow((coordenadas[2] - base[2]), 2));
    System.out.println("Distancia a la base: " + distancia + ", con la z: " + distanciaz);
    distancia = Math.sqrt(Math.pow((data.getX() - px), 2) +
                                 Math.pow((data.getY() - py), 2));
    distanciaz = Math.sqrt(Math.pow((data.getX() - px), 2) +
                                 Math.pow((data.getY() - py), 2) +
                                 Math.pow((data.getZ() - pz), 2));
    System.out.println("Distancia a la base en ECEF: " + distancia + ", con la z: " + distanciaz);
    distancia = Math.sqrt(Math.pow((data.getX() - px), 2) +
                                 Math.pow((data.getY() - py), 2));
    distanciaz = Math.sqrt(Math.pow((data.getX() - px), 2) +
                                 Math.pow((data.getY() - py), 2) +
                                 Math.pow((data.getZ() - pz), 2));
    System.out.println("Distancia a la base en ECEF con factor: " + distancia + ", con la z: " + distanciaz);
  }

  public void setMatriz() {
    T = new Matrix(v);
    T = T.transpose().inverse();
  }

  public void addObstaculo(String nombre) {
      if (obstaculos == null)
        obstaculos = new Hashtable();
      double obs[] = null;
      if ((obs = (double [])obstaculos.get(nombre)) == null)
        return;
      canvas.addObstaculo(obs);
  }

  public void addObstaculo(String nombre, double puntos[]) {
    if (obstaculos == null)
      obstaculos = new Hashtable();
    obstaculos.put(nombre, puntos);
    canvas.addObstaculo(puntos);
  }

  public void setObstaculo(String id) {
    if (obstaculos == null)
      obstaculos = new Hashtable();
    GPSData data = gps.getECEF();
    double xy[] = cambioCoordenadas(data.getX(), data.getY(), data.getZ());
    double obs[] = null;
    if ((obs = (double[])obstaculos.get(id)) == null) {
      obs = new double[] { xy[0], xy[1] };
    } else {
      double tmp[] = obs;
      obs = new double[tmp.length + 2];
      for (int i = 0; i < tmp.length; i++) {
        obs[i] = tmp[i];
      }
      obs[tmp.length] = xy[0];
      obs[tmp.length + 1] = xy[1];
    }
    obstaculos.put(id, obs);
  }

  public void setObstaculo(String id, double xy[]) {
   double obs[] = null;
   if (obstaculos == null)
      obstaculos = new Hashtable();
   if ((obs = (double[])obstaculos.get(id)) == null) {
     obs = new double[2];
     obs[0] = xy[0];
     obs[1] = xy[1];
   } else {
     double tmp[] = obs;
     obs = new double[tmp.length + 2];
     for (int i = 0; i < tmp.length; i++) {
       obs[i] = tmp[i];
     }
     obs[tmp.length] = xy[0];
     obs[tmp.length + 1] = xy[1];
   }

   obstaculos.put(id, obs);
 }


  public void saveObstaculos(String fichero) {
    try {
      File file = new File(fichero);
      ObjectOutputStream os = new ObjectOutputStream(new FileOutputStream(file, false));
      os.writeObject(obstaculos);
      os.close();
    } catch (Exception e) {
      System.out.println("Error al cargar los parámetros desde el fichero");
    }
  }

  public void loadObstaculos(String fichero) {
    try {
      File file = new File(fichero);
      ObjectInputStream is = new ObjectInputStream(new FileInputStream(file));
      obstaculos = (Hashtable)is.readObject();
      is.close();
    } catch (Exception e) {
      System.out.println("Error al cargar los parámetros desde el fichero");
    }
    double obs[] = null;
    Enumeration keys = obstaculos.keys();
    while (keys.hasMoreElements()) {
      obs = (double[])obstaculos.get((String)keys.nextElement());
      canvas.addObstaculo(obs);
    }
  }

  public double getOrientacionM() {
    GPSData data = gps.getGPSData();
    return data.getHdgPoloM();
  }

  public double getOrientacionN() {
    GPSData data = gps.getGPSData();
    return data.getHdgPoloN();
  }

  public double getVelocidad() {
    GPSData data = gps.getGPSData();
    return data.getSpeed();
  }

  public double getAngulo() {
    GPSData data = gps.getGPSData();
    System.out.println(data.getAngulo());
    return data.getAngulo();
  }

  public double[] getU(double x, double y, double miAngulo, double velocidad) {
    double xy[] = { x, y };
    lastCerca = setCercano(xy);
    double vConsigna[] = { kConsigna * Math.cos(angulos[lastCerca / 2]), kConsigna * Math.sin(angulos[lastCerca / 2]) };
    double vDistancia[] = { kAngulo * (ruta[lastCerca] - x), kAngulo * (ruta[lastCerca + 1] - y) };
    double vDeseado[] = { vConsigna[0] + vDistancia[0] + x,
        vConsigna[1] + vDistancia[1] + y};
    double u[] = new double[2];
    double anguloDeseado = getAnguloEntrePuntos(xy, vDeseado);
    double anguloConsigna = Math.min(Math.abs(miAngulo - anguloDeseado),
                             Math.toRadians(360) - Math.abs(miAngulo - anguloDeseado));
    if (Math.abs(miAngulo - anguloDeseado) <
        (Math.toRadians(360) - Math.abs(miAngulo - anguloDeseado))) {
      if ((miAngulo - anguloDeseado) > 0) {
        anguloConsigna *= -1;
      }
    } else {
      if ((miAngulo - anguloDeseado) < 0) {
        anguloConsigna *= -1;
      }
    }

    u[0] = anguloConsigna;

    /*System.out.println("*************************");
    System.out.println("Posicion: [" + x + ", " + y + "]");
    System.out.println("vConsigna: [" + vConsigna[0] + ", " + vConsigna[1] + "]");
    System.out.println("vDistancia: [" + vDistancia[0] + ", " + vDistancia[1] + "]");
    System.out.println("vDeseado: [" + vDeseado[0] + ", " + vDeseado[1] + "]");
    System.out.println("Angulo Actual: " + Math.toDegrees(miAngulo));
    System.out.println("Angulo Deseado: " + Math.toDegrees(anguloDeseado));
    System.out.println("Necesario girar: " + Math.toDegrees(anguloConsigna));
    System.out.println("*************************");*/

    // Calcula la velocidad
    //double difVel = Math.abs(velocidades[lastCerca / 2] - velocidad);

    //u[1] = k3 * difVel - k4 * Math.abs(anguloConsigna);

    u[1] = velocidades[lastCerca / 2] - velocidad;
    u[1] = velocidades[lastCerca / 2];

    System.out.println("*************************");
    System.out.println("Actual: " + velocidad);
    System.out.println("Deseada: " + velocidades[lastCerca / 2]);


    return u;
  }

  public void repintar() {
    canvas.repaint();
    frmRuta.toFront();
  }

  public void setIndependiente(boolean independiente, long refresco) {
    this.independiente = independiente;
    if (independiente) {
      this.refresco = refresco;
      Thread hilo = new Thread(this);
      hilo.start();
    }
  }

  public void setKAlfa(double kAlfa) {
    this.kAlfa = kAlfa;
  }

  public void setCocheActivo(boolean cocheActivo) {
    this.cocheActivo = cocheActivo;
  }

  public void setDllpath(String dllpath) {
    this.dllpath = dllpath;
  }

  public void setKConsigna(double KConsigna) {
    this.kConsigna = KConsigna;
  }

  public void setKDer(double KDer) {
    this.kDer = KDer;
  }

  public void setKIntegral(double KIntegral) {
    this.kIntegral = KIntegral;
  }

  public void setNextPos(int nextPos) {
    this.nextPos = nextPos * 2;
  }

  public void setAcelera(boolean acelera) {
    if (control != null)
      control.setAcelera(acelera);
  }

  public void setGira(boolean gira) {
    if (control != null)
      control.setGira(gira);
  }

  public static double getAnguloEntrePuntos(double p1[], double p2[]) {
    double angulo = 0;
    double distX = p2[0] - p1[0];
    double distY = p2[1] - p1[1];
    double hipotenusa = Math.sqrt(Math.pow(distX , 2) + Math.pow(distY , 2));

    double tg = distY / distX;
    double cos = distX / hipotenusa;

    double ang = Math.atan(tg);

    // Si la tg es > 0, estamos o en el 1er o en el 3er cuadrante
    // Si el cos es > 0, estamos o en el 1er o en el 4º cuadrante
    int cuadrante = 0;
    if (tg >= 0) {
      if (cos >= 0) {
        cuadrante = 1;
      } else {
        cuadrante = 3;
      }
    } else {
      if (cos >= 0) {
        cuadrante = 4;
      } else {
        cuadrante = 2;
      }
    }

    switch (cuadrante) {
      case 1:
        angulo = Math.abs(ang);
        break;
      case 2:
        angulo = Math.PI - Math.abs(ang);
        break;
      case 3:
        angulo = Math.PI + Math.abs(ang);
        break;
      case 4:
        angulo = 2 * Math.PI - Math.abs(ang);
    }

    return angulo;
  }

  public void relocalizar() {
    int maxc = MAX_CHECK;
    setMaxCheck(ruta.length / 2);
    setPuntoActual();
    setMaxCheck(maxc);
  }

  public void run() {
    while(independiente) {
      if (System.currentTimeMillis() - gps.getLastInstruccion() > 5000) {
        control.frenoEmergencia();
      }
      cc.setPuntoActual();
      try {
        Thread.sleep(refresco);
      } catch(Exception e) {}
    }
  }

  public void setMaxCheck(int valor) {
    if (valor == 0)
      valor = ruta.length;
    this.MAX_CHECK = valor;
  }

  public void simulaConduceSolo(double angulo, double velocidad) {
    control.simulaConduceSolo(angulo, velocidad, kAlfa, kDer, kIntegral,
            kVel, kAlfa2, kDer2, kIntegral2);
  }

  public void startCamaras(int camara1, int camara2, String dllpath) {
    if (dllpath != null && ! dllpath.equals("")) {
      this.dllpath = dllpath;
      if (media == null)
        media = new Media(dllpath);
      if (camara1 >= 0) {
        String disp1 = Media.listaDispositivos()[camara1];
        disp1 = disp1.replaceAll(" ", "&nbsp;");
        disp1 += ":" + camara1;
        System.out.println(disp1);
        String cadena = "dshow:// :dshow-vdev=" + disp1 + " :dshow-adev=none :dshow-size=640x480";
        int instancia = media.addInstancia(cadena);
        System.out.println(instancia);
        media.play(instancia);
      }
      if (camara2 >= 0) {
        String disp2 = Media.listaDispositivos()[camara2];
        disp2 = disp2.replaceAll(" ", "&nbsp;");
        disp2 += ":" + camara2;
        System.out.println(disp2);
        String cadena = "dshow:// :dshow-vdev=" + disp2+ " :dshow-adev=none :dshow-size=640x480";
        int instancia = media.addInstancia(cadena);
        System.out.println(instancia);
        media.play(instancia);
      }
    }
  }

  public void stopCamaras(int id) {
      if (media != null) {
          if (id < 0) {
              media.stopAll();
          } else {
              media.stop(id);
          }
      }
  }

  public void setMaxMedidas(int medidas) {
    gps.setMaxMedidas(medidas);
  }

  public void setAngulos(int next) {
    for (int i = 0; i < ruta.length - next * 2; i += 2) {
      int sig = i + next * 2;
      double ang = getAnguloEntrePuntos(new double[] { ruta[i], ruta[i + 1] },
                                             new double[] { ruta[sig], ruta[sig + 1] });
      angulos[i / 2] = ang;
    }
    angulos[angulos.length - 1] = getAnguloEntrePuntos(new double[] { ruta[ruta.length - 2], ruta[ruta.length - 1] },
                                           new double[] { ruta[0], ruta[1] });
    canvas.setRuta(ruta, angulos, velocidades, 800, 600);
  }

  public double[][] getRuta() {
    double[][] retorno = new double[ruta.length / 2][2];
    for (int i = 0; i < ruta.length; i += 2) {
      retorno[i / 2][0] = ruta[i];
      retorno[i / 2][1] = ruta[i + 1];
    }

    return retorno;
  }

  public static void main(String args[]) {
    /*String cadena = "msg_Err( p_demux, \""; //%i", p_data[i]);
    for (int i = 0; i < 28800; i++) {
      cadena += ", %i";
      System.out.println(i);
    }
    cadena += "\"";
    for (int i = 0; i < 28800; i++) {
      cadena += ", p_data[" + i + " + i]";
      System.out.println(i);
    }
    cadena += ");";
    System.out.println(cadena);*/
    //CambioCoordenadas cc = new CambioCoordenadas("COM3", "todo.dat");
    //cc.loadRuta("coche2.dat");
    CambioCoordenadas.showSerial();
    /*String listaDisp[] = Media.listaDispositivos();
    for (int i = 0; i < listaDisp.length; i++)
      System.out.println(listaDisp[i]);
*/
    CambioCoordenadas cc = new CambioCoordenadas("", "paramsInformatica.dat", "", false);
    cc.loadRuta("320e.dat", false);
    cc.showCanvas();
    double xy[] = cc.getGps().getXY();
    System.out.println(xy[0] + ", " + xy[1] + ", " + xy[2]);

    //cc.testRutaConImagenes(args[0], "C:\\Proyecto\\PruebasGPS\\IntegracionVideo\\testCaptura", 0);

    /*cc.loadRuta("iter2.dat", false);
    cc.setAngulos(1);
    cc.setMaxCheck(50);
    //cc.setIndependiente(true, 1000);
    cc.showCanvas();

    try {
      Thread.sleep(5000);
    } catch (Exception e) {}

    //cc.startRutaBD("pruebaBD2", 0, 1);
    cc.startRutaBD("prueba1234", 0, 1);

    try {
      Thread.sleep(20000);
    } catch (Exception e) {}

    cc.stopRutaBD();

    System.out.println("Finalizo");
    /*String dispositivo1 = "AVerTV USB 2.0:0";
    CapturaImagen ci1 = new CapturaImagen(args[0], dispositivo1, false);
    String dispositivo2 = "AVerTV USB 2.0:1";
    CapturaImagen ci2 = new CapturaImagen(args[0], dispositivo2, false);

    int i = 0;
    while (true) {
      i++;
      ImagenId ii1 = ci1.getImagen(i * 10 % 100, i * 20 % 100, i * 30 % 100);
      ii1.setNombre(dispositivo1);
      ii1.verImagen();
      ImagenId ii2 = ci2.getImagen(i * 10 % 100, i * 20 % 100, i * 30 % 100);
      ii2.setNombre(dispositivo2);
      ii2.verImagen();
      ci1.verImagen();
      try {
        Thread.sleep(10);
      } catch(Exception e) {}*/
    //}
    //cc.loadRuta2("coche3.dat");
    //cc.loadObstaculos("informaticaObs.dat");
    /*cc.setObstaculo("obs1", -7.73f, -10.39f, 0);
    cc.setObstaculo("obs1", 4.64f, -10.50f, 0);
    cc.setObstaculo("obs1", 5.21f, -50.08f, 0);
    cc.setObstaculo("obs1", -5.88f, -48.92f, 0);
    cc.addObstaculo("obs1");*/
    //cc.showCanvas();
    //cc.setIndependiente(true);
    /*while(true) {
      try {
        cc.setPuntoActual(false);
        Thread.sleep(1000);
      } catch (Exception e) {}
    }*/
  }

  public void getMaxMinSpeed() {
      double max = velocidades[0];
      double min = velocidades[0];

      for (int i = 0; i < velocidades.length; i++) {
          if (max < velocidades[i])
              max = velocidades[i];
          if (min > velocidades[i])
              min = velocidades[i];
      }

      System.out.println("Max: " + max);
      System.out.println("Min: " + min);
  }

  public void grabaRutaEnXY(String rutaOrigen, String rutaDest) {
      try {
        loadRuta(rutaOrigen, false);

        PrintWriter pw = new PrintWriter(new File(rutaDest));

        for (int i = 0; i < ruta.length; i++) {
          pw.println(ruta[i]);
        }

        pw.close();
      } catch(Exception e) {
          System.err.println("Excepción: " + e.getMessage());
      }
  }

  public double[] getRms() {
      return rms;
  }
}
