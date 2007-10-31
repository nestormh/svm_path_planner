package carrito.gps;

import java.io.File;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.FileOutputStream;
import java.io.BufferedWriter;
import java.io.FileWriter;public class TransformaFicheros {

  public static void transformaGPS2Bin(String fin, String fout) {
    try {
      File file = new File(fin);
      ObjectInputStream is = new ObjectInputStream(new FileInputStream(file));
      file = new File(fout);
      ObjectOutputStream os = new ObjectOutputStream(new FileOutputStream(file));

      long tiempo = -1;
      double x = -1;
      double y = -1;
      double z = -1;
      double speed = -1;
      double angulo = -1;
      double rms = -1;
      double latitud = -1;
      double longitud = -1;
      double altura = -1;
      int satelites = -1;
      double pdop = -1;
      double hdop = -1;
      double vdop = -1;
      double msl = -1;
      double hGeoide = -1;
      double hdgPoloN = -1;
      double hdgPoloM = -1;
      String hora = null;
      int age = -1;
      double desvEjeMayor = -1;
      double desvEjeMenor = -1;
      double orientacionMayor = -1;
      double desvLatitud = -1;
      double desvLongitud = -1;
      double desvAltura = -1;

      while (is.available() != 0) {
        tiempo = is.readLong();
        x = is.readDouble();
        y = is.readDouble();
        z = is.readDouble();
        speed = is.readDouble();
        angulo = is.readDouble();
        rms = is.readDouble();
        latitud = is.readDouble();
        longitud = is.readDouble();
        altura = is.readDouble();
        satelites = is.readInt();
        pdop = is.readDouble();
        hdop = is.readDouble();
        vdop = is.readDouble();
        msl = is.readDouble();
        hGeoide = is.readDouble();
        hdgPoloN = is.readDouble();
        hdgPoloM = is.readDouble();
        hora = is.readUTF();
        age = is.readInt();
        desvEjeMayor = is.readDouble();
        desvEjeMenor = is.readDouble();
        orientacionMayor = is.readDouble();
        desvLatitud = is.readDouble();
        desvLongitud = is.readDouble();
        desvAltura = is.readDouble();

        os.writeLong(tiempo);
        os.writeDouble(x);
        os.writeDouble(y);
        os.writeDouble(z);
        os.writeDouble(speed);
        os.writeDouble(angulo);
        os.writeDouble(rms);
        os.writeDouble(latitud);
        os.writeDouble(longitud);
        os.writeDouble(altura);
        os.writeInt(satelites);
        os.writeDouble(pdop);
        os.writeDouble(hdop);
        os.writeDouble(vdop);
        os.writeDouble(msl);
        os.writeDouble(hGeoide);
        os.writeDouble(hdgPoloN);
        os.writeDouble(hdgPoloM);
        os.writeUTF(hora);
        os.writeInt(age);
        os.writeDouble(desvEjeMayor);
        os.writeDouble(desvEjeMenor);
        os.writeDouble(orientacionMayor);
        os.writeDouble(desvLatitud);
        os.writeDouble(desvLongitud);
        os.writeDouble(desvAltura);
      }
      is.close();
      os.close();
    } catch (Exception e) {
      System.out.println("Error al cargar la ruta desde el fichero " + e.getMessage());
    }

  }

  public static void transformaGPS2Txt(String fin, String fout) {
    try {
      File file = new File(fin);
      ObjectInputStream is = new ObjectInputStream(new FileInputStream(file));
      BufferedWriter bw = new BufferedWriter(new FileWriter(fout, false));

      long tiempo = -1;
      double x = -1;
      double y = -1;
      double z = -1;
      double speed = -1;
      double angulo = -1;
      double rms = -1;
      double latitud = -1;
      double longitud = -1;
      double altura = -1;
      int satelites = -1;
      double pdop = -1;
      double hdop = -1;
      double vdop = -1;
      double msl = -1;
      double hGeoide = -1;
      double hdgPoloN = -1;
      double hdgPoloM = -1;
      String hora = null;
      int age = -1;
      double desvEjeMayor = -1;
      double desvEjeMenor = -1;
      double orientacionMayor = -1;
      double desvLatitud = -1;
      double desvLongitud = -1;
      double desvAltura = -1;

      while (is.available() != 0) {
        tiempo = is.readLong();
        x = is.readDouble();
        y = is.readDouble();
        z = is.readDouble();
        speed = is.readDouble();
        angulo = is.readDouble();
        rms = is.readDouble();
        latitud = is.readDouble();
        longitud = is.readDouble();
        altura = is.readDouble();
        satelites = is.readInt();
        pdop = is.readDouble();
        hdop = is.readDouble();
        vdop = is.readDouble();
        msl = is.readDouble();
        hGeoide = is.readDouble();
        hdgPoloN = is.readDouble();
        hdgPoloM = is.readDouble();
        hora = is.readUTF();
        age = is.readInt();
        desvEjeMayor = is.readDouble();
        desvEjeMenor = is.readDouble();
        orientacionMayor = is.readDouble();
        desvLatitud = is.readDouble();
        desvLongitud = is.readDouble();
        desvAltura = is.readDouble();

        bw.write("tiempo: " + tiempo);
        bw.write("getX: " + x);
        bw.write("getY: " + y);
        bw.write("getZ: " + z);
        bw.write("getSpeed: " + speed);
        bw.write("getAngulo: " + angulo);
        bw.write("getRms: " + rms);
        bw.write("getLatitud: " + latitud);
        bw.write("getLongitud: " + longitud);
        bw.write("getAltura: " + altura);
        bw.write("getSatelites: " + satelites);
        bw.write("getPdop: " + pdop);
        bw.write("getHdop: " + hdop);
        bw.write("getVdop: " + vdop);
        bw.write("getMsl: " + msl);
        bw.write("getHgeoide: " + hGeoide);
        bw.write("getHdgPoloN: " + hdgPoloN);
        bw.write("getHdgPoloM: " + hdgPoloM);
        bw.write("getHora: " + hora);
        bw.write("getAge: " + age);
        bw.write("getDesvEjeMayor: " + desvEjeMayor);
        bw.write("getDesvEjeMenor: " + desvEjeMenor);
        bw.write("getOrientacionMayor: " + orientacionMayor);
        bw.write("getDesvLatitud: " + desvLatitud);
        bw.write("getDesvLongitud: " + desvLongitud);
        bw.write("getDesvAltura: " + desvAltura);
      }
      is.close();
      bw.close();
    } catch (Exception e) {
      System.out.println("Error al cargar la ruta desde el fichero " + e.getMessage());
    }

  }

  public static void transformaRecibe2Bin(String fin, String fout) {
    try {
      File file = new File(fin);
      ObjectInputStream is = new ObjectInputStream(new FileInputStream(file));
      file = new File(fout);
      ObjectOutputStream os = new ObjectOutputStream(new FileOutputStream(file));

      long tiempo = -1;
      int volante = -1;
      int avance = -1;
      int avanceAnt = -1;
      int alarma = -1;
      int consignaVolanteH = -1;
      int consignaVolanteL = -1;
      int consignaFreno = -1;
      int consignaSentidoFreno = -1;
      int consignaVelocidad = -1;
      int consignaSentidoVelocidad = -1;
      int consignaVolante = -1;
      int consignaNumPasosFreno = -1;
      int numPasosFreno = -1;

      while (is.available() != 0) {
        tiempo = is.readLong();
        volante = is.readInt();
        avance = is.readInt();
        avanceAnt = is.readInt();
        alarma = is.readInt();
        consignaVolanteH = is.readInt();
        consignaVolanteL = is.readInt();
        consignaFreno = is.readInt();
        consignaSentidoFreno = is.readInt();
        consignaVelocidad = is.readInt();
        consignaSentidoVelocidad = is.readInt();
        consignaVolante = is.readInt();
        consignaNumPasosFreno = is.readInt();
        numPasosFreno = is.readInt();

        os.writeLong(tiempo);
        os.writeInt(volante);
        os.writeInt(avance);
        os.writeInt(avanceAnt);
        os.writeInt(alarma);
        os.writeInt(consignaVolanteH);
        os.writeInt(consignaVolanteL);
        os.writeInt(consignaFreno);
        os.writeInt(consignaSentidoFreno);
        os.writeInt(consignaVelocidad);
        os.writeInt(consignaSentidoVelocidad);
        os.writeInt(consignaVolante);
        os.writeInt(consignaNumPasosFreno);
        os.writeInt(numPasosFreno);
      }
      is.close();
      os.close();
    } catch (Exception e) {
      System.out.println("Error al cargar la ruta desde el fichero " + e.getMessage());
    }

  }

  public static void transformaRecibe2Txt(String fin, String fout) {
    try {
      File file = new File(fin);
      ObjectInputStream is = new ObjectInputStream(new FileInputStream(file));
      BufferedWriter bw = new BufferedWriter(new FileWriter(fout, false));

      long tiempo = -1;
      int volante = -1;
      int avance = -1;
      int avanceAnt = -1;
      int alarma = -1;
      int consignaVolanteH = -1;
      int consignaVolanteL = -1;
      int consignaFreno = -1;
      int consignaSentidoFreno = -1;
      int consignaVelocidad = -1;
      int consignaSentidoVelocidad = -1;
      int consignaVolante = -1;
      int consignaNumPasosFreno = -1;
      int numPasosFreno = -1;

      while (is.available() != 0) {
        tiempo = is.readLong();
        volante = is.readInt();
        avance = is.readInt();
        avanceAnt = is.readInt();
        alarma = is.readInt();
        consignaVolanteH = is.readInt();
        consignaVolanteL = is.readInt();
        consignaFreno = is.readInt();
        consignaSentidoFreno = is.readInt();
        consignaVelocidad = is.readInt();
        consignaSentidoVelocidad = is.readInt();
        consignaVolante = is.readInt();
        consignaNumPasosFreno = is.readInt();
        numPasosFreno = is.readInt();

        bw.write("tiempo: " + tiempo);
        bw.write("Volante: " + volante);
        bw.write("Avance: " + avance);
        bw.write("Avanceant: " + avanceAnt);
        bw.write("Alarma: " + alarma);
        bw.write("ConsignaVolanteHigh: " + consignaVolanteH);
        bw.write("ConsignaVolanteLow: " + consignaVolanteL);
        bw.write("ConsignaFreno: " + consignaFreno);
        bw.write("ConsignaSentidoFreno: " + consignaSentidoFreno);
        bw.write("ConsignaVelocidad: " + consignaVelocidad);
        bw.write("ConsignaSentidoVelocidad: " + consignaSentidoVelocidad);
        bw.write("ConsignaVolante: " + consignaVolante);
        bw.write("ConsignaNumPasosFreno: " + consignaNumPasosFreno);
        bw.write("NumPasosFreno: " + numPasosFreno);
      }
      is.close();
      bw.close();
    } catch (Exception e) {
      System.out.println("Error al cargar la ruta desde el fichero " + e.getMessage());
    }

  }

  public static void transformaEnvia2Bin(String fin, String fout) {
    try {
      File file = new File(fin);
      ObjectInputStream is = new ObjectInputStream(new FileInputStream(file));
      file = new File(fout);
      ObjectOutputStream os = new ObjectOutputStream(new FileOutputStream(file));

      long tiempo = -1;
      int giro = -1;
      int freno = -1;
      int desfreno = -1;
      int avance = -1;
      int retroceso = -1;
      int frenapasos = -1;
      int desfrenapasos = -1;

      while (is.available() != 0) {
        tiempo = is.readLong();
        giro = is.readInt();
        freno = is.readInt();
        desfreno = is.readInt();
        avance = is.readInt();
        retroceso = is.readInt();
        frenapasos = is.readInt();
        desfrenapasos = is.readInt();

        os.writeLong(tiempo);
        os.writeInt(giro);
        os.writeInt(freno);
        os.writeInt(desfreno);
        os.writeInt(avance);
        os.writeInt(retroceso);
        os.writeInt(frenapasos);
        os.writeInt(desfrenapasos);

      }
      is.close();
      os.close();
    } catch (Exception e) {
      System.out.println("Error al cargar la ruta desde el fichero " + e.getMessage());
    }

  }

  public static void transformaEnvia2Txt(String fin, String fout) {
    try {
      File file = new File(fin);
      ObjectInputStream is = new ObjectInputStream(new FileInputStream(file));
      BufferedWriter bw = new BufferedWriter(new FileWriter(fout, false));

      long tiempo = -1;
      int giro = -1;
      int freno = -1;
      int desfreno = -1;
      int avance = -1;
      int retroceso = -1;
      int frenapasos = -1;
      int desfrenapasos = -1;

      while (is.available() != 0) {
        tiempo = is.readLong();
        giro = is.readInt();
        freno = is.readInt();
        desfreno = is.readInt();
        avance = is.readInt();
        retroceso = is.readInt();
        frenapasos = is.readInt();
        desfrenapasos = is.readInt();

        bw.write("tiempo: " + tiempo);
        bw.write("giro: " + giro);
        bw.write("freno: " + freno);
        bw.write("desfreno: " + desfreno);
        bw.write("avance: " + avance);
        bw.write("retroceso: " + retroceso);
        bw.write("frenapasos: " + frenapasos);
        bw.write("desfrenapasos: " + desfrenapasos);

      }
      is.close();
      bw.close();
    } catch (Exception e) {
      System.out.println("Error al cargar la ruta desde el fichero " + e.getMessage());
    }
  }

  public static void main(String[] args) {
    TransformaFicheros transformaficheros = new TransformaFicheros();
  }
}
