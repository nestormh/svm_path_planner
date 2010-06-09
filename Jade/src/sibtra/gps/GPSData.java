package sibtra.gps;

import java.io.Serializable;
import java.util.Locale;
import java.util.regex.Pattern;

import sibtra.imu.AngulosIMU;

import Jama.Matrix;

/**
 * Contiene toda la información que proporcina el GPS sobre un punto 
 */
public class GPSData implements Serializable, Cloneable {

    /**
     * Número de serie. IMPORTANTE porque vamos a salvarlo en fichero directamente.
     * Si cambiamos estructura del objeto tenemos que cambiar el número de serie y ver 
     * como se cargan versiones anteriores.
     */
    private static final long serialVersionUID = 4L;
    private static final double a = 6378137.0;
    private static final double b = 6356752.31424518d;
//	private static final double e1 = 1.4166d;
//
//	/** Vector del polo N */
//	double u[] = new double[] { 0, b };
    //private static final double e = 0.0821;//0.08181919084262032d;
    private static final double e = 0.08181919084262032d;

    /**
     * Convierte cadena de tiempo recibida del GPS al formato hh:mm:ss
     * @param cadena
     * @return tiempo en formato hh:mm:ss.ss
     */
    private static String cadena2Time(String cadena) {
        if (cadena.length() < 6) {
            return "";
        }
        return cadena.substring(0, 2) + ":" + cadena.substring(2, 4) + ":" + cadena.substring(4);
    }

    /**
     * Convierte cadena de caracteres que representa grados a double correspondiente
     * @param valor cadena a convertir
     * @param enteros número de dígitos que corresponden a grados (resto son minutos)
     * @return grados representados por la cadena
     */
    public static double sexagesimal2double(String valor, int enteros) {
        double grados = 0;
        double minutos = 0;
        if (valor.length() > enteros) {
            grados = Integer.parseInt(valor.substring(0, enteros));
            minutos = Double.parseDouble(valor.substring(enteros));
            return grados + (minutos / 60.0d);
        } else {
            return Double.parseDouble(valor);
        }
    }

    /** @return Cadena de caracteres de representación en grados y minutos	 */
    public static String double2sexagesimal(double grados) {
        double gr = (int)grados; //truncamos
        double minutos = (grados - gr) * 60.0;
        return String.format((Locale) null, "% 3.0fº %8.5f'", gr, Math.abs(minutos));
    }
    /**
     * Age of differential corrections in seconds
     */
    private double age = Double.NaN;
    /**
     * Altura.
     * Se puede calcular como {@link #hGeoide} o como suma de {@link #hGeoide} + {@link #msL}.
     */
    private double altura = Double.NaN;
    /**
     * Angulo del último desplazamiento con respecto sistemas de coordenadas locales.
     * Será el del vector que une el punto anterior con éste.
     */
    private double angulo = Double.NaN;
    /** cadena que se recibió del GPS y que dió lugar a este punto */
    private String cadenaNMEA;
    /** Vector columna que contiene coordenadas en sistema ECEF 
     * Se calcula en base a la {@link #altura}, {@link #longitud} y {@link #latitud}.
     */
    private Matrix coordECEF = null;
    /**
     * vector columna con las 3 coordenadas respecto sistema 
     * de coordenadas local (con 0 en centro de trayectoria)
     */
    private Matrix coordLocal = null;
    /**
     * Standard deviation of altitude error (meters)
     */
    private double desvAltura = Double.NaN;
    /**
     * Standard deviation of semi-major axis of error ellipse (meters)
     */
    private double desvEjeMayor = Double.NaN;
    /**
     * Standard deviation of semi-minor axis of error ellipse (meters)
     */
    private double desvEjeMenor = Double.NaN;
    /**
     * Standard deviation of latitude error (meteers)
     */
    private double desvLatitud = Double.NaN;
    /**
     * Standard deviation of longitude error (meters)
     */
    private double desvLongitud = Double.NaN;
    /**
     * Rumbo al norte magnético en grados (0-359)
     */
    private double hdgPoloM = Double.NaN;
    /**
     * Rumbo al norte verdadero en grados (0-359)
     */
    private double hdgPoloN = Double.NaN;
    /**
     * Horizontal Dilution of Precision (HDOP) =0.0 to 9.9
     * Describes the quality of the satellite geometry. A lower value is better than a
     * higher number. An HDOP of less than 1.0 indicates strong satellite geometry,
     *  which promotes good positioning accuracy. A value of over 3.0 indicates 
     *  weaker satellite geometry and accuracy may become affected. 
     *  This information is parsed from the GPGGA NMEA message.
     */
    private double hdoP = Double.NaN;
    /**
     * Separación de la geoide en metros con respecto al elipsoide (puede ser + o -)
     * Se recibe como '±xxxx.xx'
     */
    private double hGeoide = Double.NaN;
    /**
     * Cadena de caracteres con la hora en hh:mm:ss
     */
    private String hora;
    /**
     * Latitud del punto en grados con signo.
     */
    private double latitud = Double.NaN;
    /**
     * Logitud del punto en grados con signo.
     */
    private double longitud = Double.NaN;
    /**
     * Altura de la antena en metros.
     */
    private double msL = Double.NaN;
    /**
     * Orientation of semi-major axis of error ellipse (meters)
     */
    private double orientacionMayor = Double.NaN;
    /**
     * Position Dilution of Precision (PDOP) 1.0 to 9.9
     * La Dilución de la Precisión Posicional es una medida sin unidades que indica 
     * cuando la geometría satelital provee los resultados mas precisos. 
     * Cuando los satélites están desparramados por el espacio,
     *  el valor PDOP es bajo y las posiciones computadas son más precisas. 
     *  Cuando los satélites están agrupados el valor PDOP es alto y las posiciones imprecisas.
     *   Para obtener precisiones submétricas el PDOP debe ser de 4 o menos.
     */
    private double pdoP = Double.NaN;
    /**
     * Error cuadrático medio.
     * Root mean square (rms) value of the standard deviation of the range inputs to 
     * the navigation process. Range inputs include pseudoranges and differential 
     * GNSS (DGNSS) corrections
     */
    private double rms = Double.NaN;
    /**
     * Número de satelites disponibles cuando se obtuvo el punto.
     */
    private int satelites = 0;
    /**
     * Hora del ordenador cuando se obtuvo el punto (en milisegundos).
     * Como lo devuelve la llamada <code>System.currentTimeMillis()</code>.
     */
    private long sysTime = 0;
    /**
     *  Vertical Dilution of Precision (VDOP) = 1.0 to 9.9
     */
    private double vdoP = Double.NaN;
    /**
     * Velocidad obtenida de algún instrumento externo, no a partir de los datos del GPS
     * 
     */
    private double velocidad = Double.NaN;
    /**
     * Velocidad estimada por el GPS.
     * Speed over ground, 000 to 999 km/h
     */
    private double velocidadGPS = Double.NaN;
    /** Angulos leidos de la IMU. Serán null si no existe acceso a la IMU */
    private AngulosIMU angulosIMU = null;
    
    /** nombre por si es un punto significativo (waypoint)*/
    private String nombre=null;
    
	/** Matriz de cambio de coordenadas ECEF a coordenadas locales */
	private Matrix T = null;

    public AngulosIMU getAngulosIMU() {
        return angulosIMU;
    }

    public void setAngulosIMU(AngulosIMU angulosIMU) {
        this.angulosIMU = angulosIMU;
    }

    /** Constructor por defecto */
    public GPSData() {
    }

    /** Constructor de copia */
    public GPSData(GPSData aCopiar) {
        this.copy(aCopiar);
    }

    /** Constructuor al que se le pasan Latitud, Longitud, Altura */
    public GPSData(double Latitud, double Longitud, double Altura) {
        latitud = Latitud;
        longitud = Longitud;
        altura = Altura;
    }

    /** Constructor al que se pasa array de tres doubles que son  Latitud, Longitud, Altura */
    public GPSData(double[] lla) {
        if (lla.length != 3) {
            return;
        }
        latitud = lla[0];
        longitud = lla[1];
        altura = lla[2];
    }

    /**
     * Copia los datos del punto pasado a este
     * @param aCopiar punto que va haser copiado
     * @return este objeto
     */
    public GPSData copy(GPSData aCopiar) {
        age = aCopiar.age;
        altura = aCopiar.altura;
        angulo = aCopiar.angulo;
        cadenaNMEA = aCopiar.cadenaNMEA;
        if (aCopiar.coordECEF != null) {
            coordECEF = (Matrix) aCopiar.coordECEF.clone();
        }
        if (aCopiar.coordLocal != null) {
            coordLocal = (Matrix) aCopiar.coordLocal.clone();
        }
        desvAltura = aCopiar.desvAltura;
        desvEjeMayor = aCopiar.desvEjeMayor;
        desvEjeMenor = aCopiar.desvEjeMenor;
        desvLatitud = aCopiar.desvLatitud;
        desvLongitud = aCopiar.desvLongitud;
        hdgPoloM = aCopiar.hdgPoloM;
        hdgPoloN = aCopiar.hdgPoloN;
        hdoP = aCopiar.hdoP;
        hGeoide = aCopiar.hGeoide;
        hora = aCopiar.hora;
        latitud = aCopiar.latitud;
        longitud = aCopiar.longitud;
        msL = aCopiar.msL;
        orientacionMayor = aCopiar.orientacionMayor;
        pdoP = aCopiar.pdoP;
        rms = aCopiar.rms;
        satelites = aCopiar.satelites;
        sysTime = aCopiar.sysTime;
        vdoP = aCopiar.vdoP;
        velocidad = aCopiar.velocidad;
        velocidadGPS = aCopiar.velocidadGPS;
        if (aCopiar.angulosIMU != null) {
            angulosIMU = new AngulosIMU(aCopiar.angulosIMU);
        }
        nombre=aCopiar.nombre;
        T=aCopiar.T;
        return this;
    }

    /**
     * Calcula y actualiza el agulo y la velocidad del punto pasado con 
     * respecto a este punto.
     * @param val punto que se usa para el calculo y se actualiza con velocidad y ángulo calculados
     */
    public void calculaAngSpeed(GPSData val) {
        double x = val.getXLocal() - getXLocal();
        double y = val.getYLocal() - getYLocal();

        double ang = Math.atan2(y, x);
        //if (ang < 0) ang += 2 * Math.PI;
        //usamos valores entre -PI/2 y PI/2

        // En principio no diferencio entre angulo y angulo local
        val.setAngulo(ang);

        double vel = Math.sqrt(Math.pow(x, 2.0f) + Math.pow(y, 2.0f));
        vel /= (val.getSysTime() - getSysTime()) / 1000.0;
        val.setVelocidad(vel);
    }
    /** Método que calcula el ángulo del vehículo a partir de los puntos del
     * GPS. Se usará esté método cuando no esté disponible la IMU. 
     * @param val Se pasa el punto siguiente para comparar con el actual
     */
    public double calculaAnguloGPS(GPSData val){    	
        double x = val.getXLocal() - getXLocal();
        double y = val.getYLocal() - getYLocal();
        double ang = Math.atan2(y,x);
        val.setAngulo(ang);
        return ang;
    }
    /**Método que calcula la velocidad del vehículo a partir de los puntos del
     * GPS y del tiempo en el que se recogieron. Se utilizará cuando no esté 
     * disponible el sistema odométrico o el sensor de velocidad del vehículo
     * @param val Se pasa el punto siguiente para comparar con el actual
     */
    public double calculaVelocidadGPS(GPSData val){
        double x = val.getXLocal() - getXLocal();
        double y = val.getYLocal() - getYLocal();
        double vel = Math.sqrt(Math.pow(x, 2.0f) + Math.pow(y, 2.0f));
        vel /= (val.getSysTime() - getSysTime()) / 1000.0;
        val.setVelocidad(vel);
        return vel;
    }    

    /** Devuelve objeto {@link GPSData} identico al actual */
    public Object clone() {
        Object clone = null;
        try {
            clone = super.clone();
        } catch (CloneNotSupportedException e) {
        }
        return clone;
    }

    /** @return distancia (en sistema de coordenadas ECEF) entre punto actual y el pasado */
    public double distancia(GPSData data) {
        if (data == null || data.coordECEF == null || coordECEF == null) {
            return Double.NaN;
        }
        return data.coordECEF.minus(coordECEF).normF();
    }

    /** @return distancia (en sistema de coordenadas local) entre punto actual y el pasado */
    public double distanciaLocal(GPSData data) {
        if (data == null || data.coordLocal == null || coordLocal == null) {
            return Double.NaN;
        }
        return data.coordLocal.minus(coordLocal).normF();
    }

    /** @return {@link #age} la edad de la actualización */
    public double getAge() {
        return this.age;
    }

    /** @return {@link #altura} la altura sobre el elipsoide */
    public double getAltura() {
        return this.altura;
    }

    /** @return {@link #angulo} angulo con respecto al punto anterior */
    public double getAngulo() {
        return this.angulo;
    }

    /** @return the cadenaNMEA	 */
    public String getCadenaNMEA() {
        return cadenaNMEA;
    }

    /** @return the coordECEF */
    public Matrix getCoordECEF() {
        return coordECEF;
    }

    /** @return the coordLocal	 */
    public Matrix getCoordLocal() {
        return coordLocal;
    }

    /** @return {@link #desvAltura} */
    public double getDesvAltura() {
        return this.desvAltura;
    }

    public double getDesvEjeMayor() {
        return this.desvEjeMayor;
    }

    public double getDesvEjeMenor() {
        return this.desvEjeMenor;
    }

    public double getDesvLatitud() {
        return this.desvLatitud;
    }

    public double getDesvLongitud() {
        return this.desvLongitud;
    }

    public double getHdgPoloM() {
        return this.hdgPoloM;
    }

    public double getHdgPoloN() {
        return this.hdgPoloN;
    }

    public double getHDOP() {
        return this.hdoP;
    }

    /** @return {@link #hGeoide} desviación de la geoide resepecto al elipsoide */
    public double getHGeoide() {
        return this.hGeoide;
    }

    public String getHora() {
        return this.hora;
    }

    public double getLatitud() {
        return this.latitud;
    }

    public double getLongitud() {
        return this.longitud;
    }

    /** @return {@link #msL} altura de la antena sobre geoide */
    public double getMSL() {
        return this.msL;
    }

    public double getOrientacionMayor() {
        return this.orientacionMayor;
    }

    public double getPDOP() {
        return this.pdoP;
    }

    public double getRms() {
        return this.rms;
    }

    public int getSatelites() {
        return this.satelites;
    }

    public long getSysTime() {
        return this.sysTime;
    }

    public double getVDOP() {
        return this.vdoP;
    }

    public double getVelocidad() {
        return this.velocidad;
    }

    public double getVelocidadGPS() {
        return this.velocidadGPS;
    }

    /** @return componente del vector en sistema ECEF. NaN si no se ha creado */
    public double getX() {
        if (coordECEF == null) {
            return Double.NaN;
        }
        return coordECEF.get(0, 0);
    }

    /** @return componente del vector en sistema local. NaN si no se ha creado */
    public double getXLocal() {
        if (coordLocal == null) {
            return Double.NaN;
        }
        return coordLocal.get(0, 0);
    }

    /** @return componente del vector en sistema ECEF. NaN si no se ha creado */
    public double getY() {
        if (coordECEF == null) {
            return Double.NaN;
        }
        return coordECEF.get(1, 0);
    }

    /** @return componente del vector en sistema local. NaN si no se ha creado */
    public double getYLocal() {
        if (coordLocal == null) {
            return Double.NaN;
        }
        return coordLocal.get(1, 0);
    }

    /** @return componente del vector en sistema ECEF. NaN si no se ha creado */
    public double getZ() {
        if (coordECEF == null) {
            return Double.NaN;
        }
        return coordECEF.get(2, 0);
    }

    /** @return componente del vector en sistema local. NaN si no se ha creado */
    public double getZLocal() {
        if (coordLocal == null) {
            return Double.NaN;
        }
        return coordLocal.get(2, 0);
    }

    /**
     * Interpreta la cadena recibida del GPS y almacena los distintos datos contenidos.
     * Trata los mensajes GSA, GST, VTG y GGA.
     * @param cadena mensaje a interpretar
     * @return true si el paquete es GGA VÁLIDO
     */
    public boolean procesaCadena(String cadena) {//throws IOException, Exception {
        boolean esValido = false;
        String[] msj = cadena.split(",");

        if (Pattern.matches("\\$..GSA", msj[0])) {
            //System.out.println(System.currentTimeMillis() + "***" + cadena + "***");
            if (msj[15].equals("")) {
                setPDOP(0);
            } else {
                setPDOP(Double.parseDouble(msj[15]));
            }

            if (msj[16].equals("")) {
                setHDOP(0);
            } else {
                setHDOP(Double.parseDouble(msj[16]));
            }

            msj[17] = (msj[17].split("\\*"))[0];
            if (msj[17].equals("")) {
                setVDOP(0);
            } else {
                setVDOP(Double.parseDouble(msj[17]));
            }
        }

        if (Pattern.matches("\\$..GST", msj[0])) {
            //System.out.println(System.currentTimeMillis() + "***" + cadena + "***");

            if (msj[2].equals("")) {
                setRms(0);
            } else {
                setRms(Double.parseDouble(msj[2]));
            }

            if (msj[3].equals("")) {
                setDesvEjeMayor(0);
            } else {
                setDesvEjeMayor(Double.parseDouble(msj[3]));
            }

            if (msj[4].equals("")) {
                setDesvEjeMenor(0);
            } else {
                setDesvEjeMenor(Double.parseDouble(msj[4]));
            }

            if (msj[5].equals("")) {
                setOrientacionMayor(0);
            } else {
                setOrientacionMayor(Double.parseDouble(msj[5]));
            }

            if (msj[6].equals("")) {
                setDesvLatitud(0);
            } else {
                setDesvLatitud(Double.parseDouble(msj[6]));
            }

            if (msj[7].equals("")) {
                setDesvLongitud(0);
            } else {
                setDesvLongitud(Double.parseDouble(msj[7]));
            }

            msj[8] = (msj[8].split("\\*"))[0];
            if (msj[8].equals("")) {
                setDesvAltura(0);
            } else {
                setDesvAltura(Double.parseDouble(msj[8]));
            }

        }

        if (Pattern.matches("\\$..VTG", msj[0])) {
            //System.out.println(System.currentTimeMillis() + "***" + cadena + "***");
            if ((msj[2].equals("T") && (!msj[1].equals("")))) {
                setHdgPoloN(Math.toRadians(Double.parseDouble(msj[1])));
            } else {
                setHdgPoloN(0);
            }

            if ((msj[4].equals("M")) && (!msj[3].equals(""))) {
                setHdgPoloM(Math.toRadians(Double.parseDouble(msj[3])));
            } else {
                setHdgPoloM(0);
            }

            msj[8] = (msj[8].split("\\*"))[0];
            if ((msj[8].equals("K")) && (!msj[7].equals(""))) {
                setVelocidadGPS(Double.parseDouble(msj[7]));
            } else {
                setVelocidadGPS(0);
            }
        }

        if (Pattern.matches("\\$..GGA", msj[0])) {
            //System.out.println(System.currentTimeMillis() + "***" + cadena + "***");

            if (msj[1].equals("")) {
                setHora("");
            } else {
                setHora(cadena2Time(msj[1]));
            }
            if (msj[2].equals("")) {
                setLatitud(0);
            } else {
                setLatitud(sexagesimal2double(msj[2], 2));
            }
            if (!msj[3].equals("") && msj[3].equals("S")) {
                setLatitud(getLatitud() * -1);
            }
            if (msj[2].equals("")) {
                setLongitud(0);
            } else {
                setLongitud(sexagesimal2double(msj[4], 3));
            }
            if (!msj[5].equals("") && msj[5].equals("W")) {
                setLongitud(getLongitud() * -1);
            }

            if (msj[7].equals("")) {
                setSatelites(0);
            } else {
                setSatelites(Integer.parseInt(msj[7]));
            }

            if ((!msj[9].equals("")) && !msj[10].equals("") && msj[10].equals("M")) {
                setMSL(Double.parseDouble(msj[9]));
            } else {
                setMSL(0);
            }

            if ((!msj[11].equals("")) && !msj[12].equals("") && (msj[12].equals("M"))) {
                setHGeoide(Double.parseDouble(msj[11]));
            } else {
                setHGeoide(0);
            }

            //altura = msl + hgeoide;
            setAltura(getHGeoide() + getMSL());
            //setAltura(getHGeoide());


            if (msj[13].equals("")) {
                setAge(-1);
            } else {
                setAge(Double.parseDouble(msj[13]));
            }

            //calculaLLA(latitud, longitud, altura); 
            esValido = (getLongitud() != 0) && (getLatitud() != 0) && (getHGeoide() != 0);
        }
        setCadenaNMEA(cadena);
        return esValido;
    }

    public void setAge(double value) {
        this.age = value;
    }

    /** Actuliza altura. Si es distinta que la anterior anula coordenadas ECEF y locales*/
    public void setAltura(double value) {
    	if(value==altura)
    		return;
        this.altura = value;
        coordECEF=null;
        coordLocal=null;
    }

    public void setAngulo(double value) {
        this.angulo = value;
    }

    /** @param cadenaNMEA the cadenaNMEA to set	 */
    public void setCadenaNMEA(String cadenaNMEA) {
        this.cadenaNMEA = cadenaNMEA;
    }

    /** @param coordECEF the coordECEF to set	 */
    void setCoordECEF(Matrix coordECEF) {
        if (coordECEF != null && (coordECEF.getColumnDimension() != 1 || coordECEF.getRowDimension() != 3)) {
            throw new IllegalArgumentException("Parámetro debe ser vector columna de 3 componentes");
        }
        this.coordECEF = coordECEF;
    }

    /** @param coordLocal the coordLocal to set	 */
    void setCoordLocal(Matrix coordLocal) {
        if (coordLocal != null &&
                (coordLocal.getColumnDimension() != 1 || coordLocal.getRowDimension() != 3)) {
            throw new IllegalArgumentException("Parámetro debe ser vector columna de 3 componentes");
        }
        this.coordLocal = coordLocal;
    }

    public void setDesvAltura(double value) {
        this.desvAltura = value;
    }

    public void setDesvEjeMayor(double value) {
        this.desvEjeMayor = value;
    }

    public void setDesvEjeMenor(double value) {
        this.desvEjeMenor = value;
    }

    public void setDesvLatitud(double value) {
        this.desvLatitud = value;
    }

    public void setDesvLongitud(double value) {
        this.desvLongitud = value;
    }

    /**
     * Calcula y actualiza las coordenadas x,y,z (ECEF) del punto.
     */
    public GPSData calculaECEF() {
    	if(coordECEF!=null)
    		return this; //ya están las coordenadas
        //no aún no están las calculamos
        if (latitud == Double.NaN || longitud == Double.NaN || altura == Double.NaN) {
            throw (new IllegalArgumentException("El punto no tiene infomración suficiente para calcular ECEF (LLA)"));
        }
        double latitudRad = Math.toRadians(getLatitud());
        double longitudRad = Math.toRadians(getLongitud());

        double N = a / Math.sqrt(1 - (Math.pow(e, 2.0f) * Math.pow(Math.sin(latitudRad), 2.0f)));
        double x = (N + altura) * Math.cos(latitudRad) * Math.cos(longitudRad);
        double y = (N + altura) * Math.cos(latitudRad) * Math.sin(longitudRad);
        double z = (((Math.pow(b, 2.0f) / Math.pow(a, 2.0f)) * N) + altura) * Math.sin(latitudRad);
        setX(x);
        setY(y);
        setZ(z);

        return this;
    }
    
    /** Calcula y fija las coordenadas locales tomando como centro el punto pasado.
     * Al punto pasado (centro) se le calculan las coordenadas ECEF y la matriz T (si no la tiene)
     * Si el punto pasado es <code>null</code>, las coordenadas locales se ponen a <code>null</code>
     * @param centro punto a usar como centro de las coordenadas locales
     * @return el mimo punto (para poder encadenar métodos ;-) 
     */
    public GPSData calculaLocales(GPSData centro) {
    	if(centro==null) {
    		setCoordLocal(null);
    		return this;
    	}
    		
    	if(centro.T==null) {
    		// Matriz de rotación en torno a un punto
    		double v[][] = new double[3][];
    		double lonCenRad=Math.toRadians(centro.getLongitud());
    		double latCenRad=Math.toRadians(centro.getLatitud());
    		v[0] = new double[] { -Math.sin(lonCenRad), Math.cos(lonCenRad), 0 };
    		v[1] = new double[] { -Math.cos(lonCenRad) * Math.sin(latCenRad), -Math.sin(latCenRad) * Math.sin(lonCenRad), Math.cos(latCenRad) };
    		v[2] = new double[] { Math.cos(latCenRad) * Math.cos(lonCenRad), Math.cos(latCenRad) * Math.sin(lonCenRad), Math.sin(latCenRad)};

    		Matrix M1 = new Matrix(v);

    		// Matriz de inversión del eje z en torno al eje x (Norte)
    		double w[][] = new double[3][];
    		w[0] = new double[] { 0, 1, 0 };
    		w[1] = new double[] { -1, 0, 0 };
    		w[2] = new double[] { 0, 0, 1 };
    		Matrix M2 = new Matrix(w);

    		centro.T = M2.times(M1); 
    		centro.calculaECEF(); //Las necesitará para los cálculos posteriores
    	}
    	calculaECEF(); //por si no están ya que se necesitan
//    	System.out.println("Ecef:"+coordECEF.get(0,0)+","+coordECEF.get(1,0)+","+coordECEF.get(2,0));
    	Matrix res = getCoordECEF().minus(centro.getCoordECEF()); 
		res = centro.T.times(res); //dejamos como vector columna
		setCoordLocal(res);

    	return this;
    }

    public void setHdgPoloM(double value) {
        this.hdgPoloM = value;
    }

    public void setHdgPoloN(double value) {
        this.hdgPoloN = value;
    }

    public void setHDOP(double value) {
        this.hdoP = value;
    }

    public void setHGeoide(double value) {
        this.hGeoide = value;
    }

    public void setHora(String value) {
        this.hora = value;
    }

    /** Actuliza latitud. Si es distinta que la anterior anula coordenadas ECEF y locales*/
    public void setLatitud(double value) {
    	if(value==latitud)
    		return;
        this.latitud = value;
        coordECEF=null;
        coordLocal=null;
    }

    /** Actuliza longitud. Si es distinta que la anterior anula coordenadas ECEF y locales*/
    public void setLongitud(double value) {
    	if(value==longitud)
    		return;
        this.longitud = value;
        coordECEF=null;
        coordLocal=null;
    }

    public void setMSL(double value) {
        this.msL = value;
    }

    public void setOrientacionMayor(double value) {
        this.orientacionMayor = value;
    }

    public void setPDOP(double value) {
        this.pdoP = value;
    }

    public void setRms(double value) {
        this.rms = value;
    }

    public void setSatelites(int value) {
        this.satelites = value;
    }

    public void setSysTime(long sysTime) {
        this.sysTime = sysTime;
    }

    public void setVDOP(double value) {
        this.vdoP = value;
    }

    public void setVelocidad(double value) {
        this.velocidad = value;
    }

    public void setVelocidadGPS(double value) {
        this.velocidadGPS = value;
    }

    private void setX(double value) {
        if (coordECEF == null) {
            coordECEF = new Matrix(3, 1, Double.NaN);
        }
        coordECEF.set(0, 0, value);
    }

    public void setXLocal(double value) {
        if (coordLocal == null) {
            coordLocal = new Matrix(3, 1, Double.NaN);
        }
        coordLocal.set(0, 0, value);
    }

    private void setY(double value) {
        if (coordECEF == null) {
            coordECEF = new Matrix(3, 1, Double.NaN);
        }
        coordECEF.set(1, 0, value);
    }

    public void setYLocal(double value) {
        if (coordLocal == null) {
            coordLocal = new Matrix(3, 1, Double.NaN);
        }
        coordLocal.set(1, 0, value);
    }

    private void setZ(double value) {
        if (coordECEF == null) {
            coordECEF = new Matrix(3, 1, Double.NaN);
        }
        coordECEF.set(2, 0, value);
    }

    public void setZLocal(double value) {
        if (coordLocal == null) {
            coordLocal = new Matrix(3, 1, Double.NaN);
        }
        coordLocal.set(2, 0, value);
    }

    public String toString() {
        String retorno = "";
        
        if(nombre!=null)
        	retorno+="'"+nombre+"' ";

        //retorno += String.format((Locale)null,"LLA = [% 9.6f , % 9.6f , % 8.2f]",latitud ,longitud , altura);
        retorno += String.format((Locale) null, "LLA = [%s, %s , % 8.2f]", getLatitudText(), getLongitudText(), altura);
        if (coordLocal != null) {
            retorno += String.format((Locale) null, " PTP = [%.3f , %.3f , %.3f]", getXLocal(), getYLocal(), getZLocal());
        //retorno += "\n";
        //retorno += "ECEF = [" + x + ", " + y + ", " + z + "]\n";
        //\n";
        //retorno += "Angulo = " + angulo + " ::: " + "Velocidad = " + velocidad + "\n";
        }
        return retorno;
    }

    /** @return latitud en formato texto */
    public String getLatitudText() {
        return double2sexagesimal(latitud);
    }

    public String getLongitudText() {
        return double2sexagesimal(longitud);
    }

    /**
     * Para pruebas
     * @param args
     */
    public static void main(String[] args) {
        //probamos problema de los ángulos
        GPSData p1 = new GPSData();
        p1.setLatitud(28.0 + 28.93084 / 60.0);
        p1.setLongitud(-(16 + 19.27510 / 60));
        p1.setAltura(610);
        p1.calculaECEF();

        GPSData p2 = new GPSData();
        p2.setLatitud(28.0 + 28.93055 / 60.0);
        p2.setLongitud(-(16 + 19.27568 / 60));
        p2.setAltura(p1.getAltura());
        p2.calculaECEF();

        System.out.println("p1 ECEF=" + p1.getX() + "," + p1.getY() + "," + p1.getZ() + " p2 ECEF=" + p2.getX() + "," + p2.getY() + "," + p2.getZ());

        Ruta ruta = new Ruta();
        ruta.add(p1);
        ruta.add(p2);
        ruta.actualizaSistemaLocal(new GPSData(p1));
        ruta.actualizaCoordenadasLocales();

        p1.calculaAngSpeed(p2);
        double ang1 = p2.getAngulo();
        System.out.println(ruta + "Angulo: " + ang1 + "=>" + Math.toDegrees(ang1));
        System.out.println("Distancias: ECEF=" + p1.distancia(p2) + "  Local=" + p1.distanciaLocal(p2));



        p2.setAltura(p1.getAltura() + 100);
        p2.calculaECEF();
        ruta.actualizaCoordenadasLocales();
        p1.calculaAngSpeed(p2);

        double ang2 = p2.getAngulo();
        System.out.println("\n\n" + ruta + "\nAngulo: " + ang2 + " => " + Math.toDegrees(ang2));
        System.out.println("Distancias: ECEF=" + p1.distancia(p2) + "  Local=" + p1.distanciaLocal(p2));


        System.out.println("\nDiferencia en grados=" + Math.toDegrees(ang2 - ang1) + " => " + double2sexagesimal(Math.toDegrees(ang2 - ang1)));

        p1.setAltura(p1.getAltura() + 100);
        p1.calculaECEF();
        ruta.actualizaCoordenadasLocales();
        p1.calculaAngSpeed(p2);

        double ang3 = p2.getAngulo();
        System.out.println("\n\n" + ruta + "\nAngulo: " + ang3 + " => " + Math.toDegrees(ang3));
        System.out.println("Distancias: ECEF=" + p1.distancia(p2) + "  Local=" + p1.distanciaLocal(p2));


        System.out.println("\nDiferencia en grados=" + Math.toDegrees(ang3 - ang1) + " => " + double2sexagesimal(Math.toDegrees(ang3 - ang1)));

    }

	/**
	 * @return the nombre
	 */
	public String getNombre() {
		return nombre;
	}

	/**
	 * @param nombre the nombre to set
	 */
	public void setNombre(String nombre) {
		this.nombre = nombre;
	}

	/**
	 * @return the t
	 */
	public Matrix getT() {
		return T;
	}

}
