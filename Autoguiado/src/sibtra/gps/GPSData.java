package sibtra.gps;

import java.io.Serializable;

/**
* Protege name: GPSData
* @author ontology bean generator
* @version 2008/04/3, 16:58:25
*/
public class GPSData implements Serializable {

	/**
	 * Horizontal Dilution of Precision (HDOP) =0.0 to 9.9
	 * Describes the quality of the satellite geometry. A lower value is better than a
	 * higher number. An HDOP of less than 1.0 indicates strong satellite geometry,
	 *  which promotes good positioning accuracy. A value of over 3.0 indicates 
	 *  weaker satellite geometry and accuracy may become affected. 
	 *  This information is parsed from the GPGGA NMEA message.
	 */
	private double hdoP;
	
	public void setHDOP(double value) { 
		this.hdoP=value;
	}
	public double getHDOP() {
		return this.hdoP;
	}

	/**
	 * Position Dilution of Precision (PDOP) 1.0 to 9.9
	 * La Dilución de la Precisión Posicional es una medida sin unidades que indica 
	 * cuando la geometría satelital provee los resultados mas precisos. 
	 * Cuando los satélites están desparramados por el espacio,
	 *  el valor PDOP es bajo y las posiciones computadas son más precisas. 
	 *  Cuando los satélites están agrupados el valor PDOP es alto y las posiciones imprecisas.
	 *   Para obtener precisiones submétricas el PDOP debe ser de 4 o menos.
	 */
	private double pdoP;
	
	public void setPDOP(double value) { 
		this.pdoP=value;
	}
	public double getPDOP() {
		return this.pdoP;
	}

	/**
	 * Altura de la antena en metros.
	 */
	private double msL;
	
	public void setMSL(double value) { 
		this.msL=value;
	}
	public double getMSL() {
		return this.msL;
	}

	/**
	 *  Vertical Dilution of Precision (VDOP) = 1.0 to 9.9
	 */
	private double vdoP;
	
	public void setVDOP(double value) { 
		this.vdoP=value;
	}
	public double getVDOP() {
		return this.vdoP;
	}

	/**
	 * Rumbo al norte verdadero en grados (0-359)
	 */
	private double hdgPoloN;
	
	public void setHdgPoloN(double value) { 
		this.hdgPoloN=value;
	}
	public double getHdgPoloN() {
		return this.hdgPoloN;
	}

	/**
	 * Separación de la geoide en metros (puede ser + o -)
	 */
	private double hGeoide;
	
	public void setHGeoide(double value) { 
		this.hGeoide=value;
	}
	public double getHGeoide() {
		return this.hGeoide;
	}

	/**
	 * Cadena de caracteres con la hora en hh:mm:ss
	 */
	private String hora;
	
	public void setHora(String value) { 
		this.hora=value;
	}
	public String getHora() {
		return this.hora;
	}

	/**
	 * Rumbo al norte magnético en grados (0-359)
	 */
	private double hdgPoloM;
	
	public void setHdgPoloM(double value) { 
		this.hdgPoloM=value;
	}
	public double getHdgPoloM() {
		return this.hdgPoloM;
	}

	/**
	 * Error cuadrático medio.
	 * Root mean square (rms) value of the standard deviation of the range inputs to 
	 * the navigation process. Range inputs include pseudoranges and differential 
	 * GNSS (DGNSS) corrections
	 */
	private double rms;
	
	public void setRms(double value) { 
		this.rms=value;
	}
	public double getRms() {
		return this.rms;
	}

	/**
	 * Age of differential corrections in seconds
	 */
	private double age;
	
	public void setAge(double value) { 
		this.age=value;
	}
	public double getAge() {
		return this.age;
	}

	/**
	 * Standard deviation of semi-minor axis of error ellipse (meters)
	 */
	private double desvEjeMenor;
	
	public void setDesvEjeMenor(double value) { 
		this.desvEjeMenor=value;
	}
	public double getDesvEjeMenor() {
		return this.desvEjeMenor;
	}

	/**
	 * Standard deviation of semi-major axis of error ellipse (meters)
	 */
	private double desvEjeMayor;
	
	public void setDesvEjeMayor(double value) { 
		this.desvEjeMayor=value;
	}
	public double getDesvEjeMayor() {
		return this.desvEjeMayor;
	}

	/**
	 * Orientation of semi-major axis of error ellipse (meters)
	 */
	private double orientacionMayor;
	
	public void setOrientacionMayor(double value) { 
		this.orientacionMayor=value;
	}
	public double getOrientacionMayor() {
		return this.orientacionMayor;
	}

	/**
	 * Altura.
	 * Se puede calcular como {@link #hGeoide} o como suma de {@link #hGeoide} + {@link #msL}.
	 */
	private double altura;
	
	public void setAltura(double value) { 
		this.altura=value;
	}
	public double getAltura() {
		return this.altura;
	}

	/**
	 * Componente x de las coordenadas en sistema ECEF (Earth-Centered, Earth-Fixed).
	 * Se calcula en base a la {@link #altura}, {@link #longitud} y {@link #latitud}.
	 */
	private double x;
	
	public void setX(double value) { 
		this.x=value;
	}
	public double getX() {
		return this.x;
	}

	/**
	 * Componente y de las coordenadas en sistema ECEF (Earth-Centered, Earth-Fixed)
	 * Se calcula en base a la {@link #altura}, {@link #longitud} y {@link #latitud}.
	 */
	private double y;
	
	public void setY(double value) { 
		this.y=value;
	}
	public double getY() {
		return this.y;
	}

	/**
	 * Componente z de las coordenadas en sistema ECEF (Earth-Centered, Earth-Fixed)
	 * Se calcula en base a la {@link #altura}, {@link #longitud} y {@link #latitud}.
	 */
	private double z;
	
	public void setZ(double value) { 
		this.z=value;
	}
	public double getZ() {
		return this.z;
	}

	/**
	 * Angulo del último desplazamiento con respecto sistemas de coordenadas locales
	 */
	private double angulo;
	
	public void setAngulo(double value) { 
		this.angulo=value;
	}
	public double getAngulo() {
		return this.angulo;
	}

	/**
	 * Velocidad del último desplazamiento con respecto sistemas de coordenadas locales
	 */
	private double velocidad;
	
	public void setVelocidad(double value) { 
		this.velocidad=value;
	}
	public double getVelocidad() {
		return this.velocidad;
	}

	/**
	 * Velocidad estimada por el GPS.
	 * Speed over ground, 000 to 999 km/h
	 */
	private double velocidadGPS;
	
	public void setVelocidadGPS(double value) { 
		this.velocidadGPS=value;
	}
	public double getVelocidadGPS() {
		return this.velocidadGPS;
	}

	/**
	 * Componente x de la posición en el sistema de coordenadas loca (con 0 en centro de trayectoria) 
	 */
	private double xLocal;
	
	public void setXLocal(double value) { 
		this.xLocal=value;
	}
	public double getXLocal() {
		return this.xLocal;
	}

	/**
	 * Componente x de la posición en el sistema de coordenadas loca (con 0 en centro de trayectoria) 
	 */
	private double yLocal;
	
	public void setYLocal(double value) { 
		this.yLocal=value;
	}
	public double getYLocal() {
		return this.yLocal;
	}

	/**
	 * Componente x de la posición en el sistema de coordenadas loca (con 0 en centro de trayectoria) 
	 */
	private double zLocal;
	
	public void setZLocal(double value) { 
		this.zLocal=value;
	}
	public double getZLocal() {
		return this.zLocal;
	}

	/**
	 * Número de satelites disponibles cuando se obtuvo el punto.
	 */
	private int satelites;
	
	public void setSatelites(int value) { 
		this.satelites=value;
	}
	public int getSatelites() {
		return this.satelites;
	}

	/**
	 * Latitud del punto en grados con signo.
	 */
	private double latitud;
	
	public void setLatitud(double value) { 
		this.latitud=value;
	}
	public double getLatitud() {
		return this.latitud;
	}

	/**
	 * Logitud del punto en grados con signo.
	 */
	private double longitud;
	
	public void setLongitud(double value) { 
		this.longitud=value;
	}
	public double getLongitud() {
		return this.longitud;
	}

	/**
	 * Standard deviation of altitude error (meters)
	 */
	private double desvAltura;
	
	public void setDesvAltura(double value) { 
		this.desvAltura=value;
	}
	public double getDesvAltura() {
		return this.desvAltura;
	}

	/**
	 * Standard deviation of longitude error (meters)
	 */
	private double desvLongitud;
	
	public void setDesvLongitud(double value) { 
		this.desvLongitud=value;
	}
	public double getDesvLongitud() {
		return this.desvLongitud;
	}

	/**
	 * Standard deviation of latitude error (meteers)
	 */
	private double desvLatitud;
	
	public void setDesvLatitud(double value) { 
		this.desvLatitud=value;
	}
	public double getDesvLatitud() {
		return this.desvLatitud;
	}

	/**
	 * Hora del ordenador cuando se obtuvu el punto (en milisegundos).
	 * Como lo devuelve la llamada <code>System.currentTimeMillis()</code>.
	 */
	private long sysTime;
	
	public void setSysTime(long sysTime) { 
		this.sysTime=sysTime;
	}
	public long getSysTime() {
		return this.sysTime;
	}

	public String toString() {
		String retorno = "";

		retorno += "LLA = [" + latitud + ", " + longitud + ", " + altura + "]\n";
		//retorno += "ECEF = [" + x + ", " + y + ", " + z + "]\n";
		//retorno += "PTP = [" + xLocal + ", " + yLocal + ", " + zLocal + "]\n";
		//retorno += "Angulo = " + angulo + " ::: " + "Velocidad = " + velocidad + "\n";

		return retorno;
	}
}
