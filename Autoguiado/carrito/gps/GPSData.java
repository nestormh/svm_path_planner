package carrito.gps;

/**
 * <p>Title: </p>
 *
 * <p>Description: </p>
 *
 * <p>Copyright: Copyright (c) 2007</p>
 *
 * <p>Company: </p>
 *
 * @author not attributable
 * @version 1.0
 */
public class GPSData {
    private double latitud;
    private double longitud;
    private double altura;
    private double x;
    private double y;
    private double z;
    private double pdop;
    private double hdop;
    private double vdop;
    private double speed;
    private int satelites;

    private String latitudg = "";
    private String longitudg = "";
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

    private double angulo = 0;


    public GPSData() {
        latitud = 0;
        longitud = 0;
        altura = 0;
        x = 0;
        y = 0;
        z = 0;
        pdop = 0;
        hdop = 0;
        vdop = 0;
        speed = 0;
        satelites = 0;
        latitudg = "";
        longitudg = "";
        msl = 0;
        hgeoide = 0;
        hdgPoloN = 0;
        hdgPoloM = 0;
        hora = "";
        age = 0;
        rms = 0;
        desvEjeMayor = 0;
        desvEjeMenor = 0;
        orientacionMayor = 0;
        desvLatitud = 0;
        desvLongitud = 0;
        desvAltura  = 0;
        angulo = 0;
    }

    public GPSData(double latitud, double longitud, double altura,
                   double x, double y, double z,
                   double pdop, double hdop, double vdop,
                   double speed, int satelites, String latitudg,
                   String longitudg, double msl, double hgeoide,
                   double hdgPoloN, double hdgPoloM, String hora, double age,
                   double rms, double desvEjeMayor, double desvEjeMenor,
                   double orientacionMayor, double desvLatitud,
                   double desvLongitud, double desvAltura, double angulo) {

        this.latitud = latitud;
        this.longitud = longitud;
        this.altura = altura;
        this.x = x;
        this.y = y;
        this.z = z;
        this.pdop = pdop;
        this.hdop = hdop;
        this.vdop = vdop;
        this.speed = speed;
        this.satelites = satelites;
        this.latitudg = latitudg;
        this.longitudg = longitudg;
        this.msl = msl;
        this.hgeoide = hgeoide;
        this.hdgPoloN = hdgPoloN;
        this.hdgPoloM = hdgPoloM;
        this.hora = hora;
        this.age = age;
        this.rms = rms;
        this.desvEjeMayor = desvEjeMayor;
        this.desvEjeMenor = desvEjeMenor;
        this.orientacionMayor = orientacionMayor;
        this.desvLatitud = desvLatitud;
        this.desvLongitud = desvLongitud;
        this.desvAltura  = desvAltura;
        this.angulo = angulo;
    }


    public void setLLA(double latitud, double longitud, double altura) {
        this.latitud = latitud;
        this.longitud = longitud;
        this.altura = altura;
    }

    public void setECEF(double x, double y, double z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    public void setError(double pdop, double hdop, double vdop) {
        this.pdop = pdop;
        this.hdop = hdop;
        this.vdop = vdop;
    }

    public void setLatitud(double latitud) {
        this.latitud = latitud;
    }

    public void setLongitud(double longitud) {
        this.longitud = longitud;
    }

    public void setAltura(double altura) {
        this.altura = altura;
    }

    public void setX(double x) {
        this.x = x;
    }

    public void setY(double y) {
        this.y = y;
    }

    public void setZ(double z) {
        this.z = z;
    }

    public void setPdop(double pdop) {
        this.pdop = pdop;
    }

    public void setHdop(double hdop) {
        this.hdop = hdop;
    }

    public void setVdop(double vdop) {
        this.vdop = vdop;
    }

    public void setSpeed(double speed) {
        this.speed = speed;
    }

    public void setSatelites(int satelites) {
        this.satelites = satelites;
    }

    public void setAge(double age) {
        this.age = age;
    }

    public void setRms(double rms) {
        this.rms = rms;
    }

    public void setOrientacionMayor(double orientacionMayor) {
        this.orientacionMayor = orientacionMayor;
    }

    public void setMsl(double msl) {
        this.msl = msl;
    }

    public void setLongitudg(String longitudg) {
        this.longitudg = longitudg;
    }

    public void setLatitudg(String latitudg) {
        this.latitudg = latitudg;
    }

    public void setHora(String hora) {
        this.hora = hora;
    }

    public void setHgeoide(double hgeoide) {
        this.hgeoide = hgeoide;
    }

    public void setHdgPoloN(double hdgPoloN) {
        this.hdgPoloN = hdgPoloN;
    }

    public void setHdgPoloM(double hdgPoloM) {
        this.hdgPoloM = hdgPoloM;
    }

    public void setDesvLongitud(double desvLongitud) {
        this.desvLongitud = desvLongitud;
    }

    public void setDesvLatitud(double desvLatitud) {
        this.desvLatitud = desvLatitud;
    }

    public void setDesvEjeMenor(double desvEjeMenor) {
        this.desvEjeMenor = desvEjeMenor;
    }

    public void setDesvEjeMayor(double desvEjeMayor) {
        this.desvEjeMayor = desvEjeMayor;
    }

    public void setDesvAltura(double desvAltura) {
        this.desvAltura = desvAltura;
    }

  public void setAngulo(double angulo) {
    this.angulo = angulo;
  }

  public double getLatitud() {
        return latitud;
    }

    public double getLongitud() {
        return longitud;
    }

    public double getAltura() {
        return altura;
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

    public double getPdop() {
        return pdop;
    }

    public double getHdop() {
        return hdop;
    }

    public double getVdop() {
        return vdop;
    }

    public double getSpeed() {
        return speed;
    }

    public int getSatelites() {
        return satelites;
    }

    public double getAge() {
        return age;
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

  public double getAngulo() {
    return angulo;
  }
}
