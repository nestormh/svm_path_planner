package sibtra.gps;

import Jama.Matrix;
import java.io.*;
import java.util.*;
import java.util.regex.*;
//import javax.comm.*;
import gnu.io.*;
import xbeedemo.SerialConnectionException;
import xbeedemo.SerialParameters;

/**
 * Maneja la conexión serial con el GPS. 
 * Al recibir un paquete recoge la información del nuevo punto y lo almacena en los distintos
 * buffers. 
 */
public class GPSConnection implements SerialPortEventListener {

	private static final long a = 6378137;
	private static final double b = 6356752.31424518d;
	private static final double e = 0.0821;//0.08181919084262032d;
	private static final double e1 = 1.4166d;
	
	/** Numero máximo de puntos acumulados en el buffer de ruta */
	private static final int MAXBUFFER = 1000;

	private static final double NULLANG = -5 * Math.PI;

	/** Vector del polo N */
	double u[] = new double[] { 0, b };

	/** si el puerto serial está abierto*/
	private boolean open;

	private SerialParameters parameters;    
	private InputStream is;    

	private CommPortIdentifier portId;
	private SerialPort sPort;

	/** Cadena en la que se van almacenando los trozos de mensajes que se van recibiendo */
	private String cadenaTemp = "";
	
	/** último punto recibido */
	GPSData data = new GPSData();    

	/** minima distancia que quienten que estar separados dos puntos para que se consideren distintos
	 * y se almacenen en el buffer espacial.
	 */
	public static double minDistOperativa = 0.7;

	/** Punto mas cercano al centro de la ruta */
	private GPSData centro = null;

	/** Matriz de cambio de coordenadas ECEF a coordenadas locales */
	Matrix T = null;

	/** Almacena ruta temporal cargada de fichero */
	private Vector<GPSData> rutaTemporal = new Vector<GPSData>();
	/** Almacena ruta espacial cargada de fichero */
	private Vector<GPSData> rutaEspacial = new Vector<GPSData>();

	/** Almacena los {@link #MAXBUFFER} últimos puntos que están separados al menos {@link #minDistOperativa} */
	private Vector<GPSData> bufferEspacial = new Vector<GPSData>();
	/** Almacena los {@link #MAXBUFFER} últimos puntos recibidos */
	private Vector<GPSData> bufferTemporal = new Vector<GPSData>();  

	/** indica si estamos capturando una ruta */
	private boolean enRuta = false;
	
	/** Si estamos {@link #enRuta}, almacena los puntos recibidos */
	private Vector<GPSData> bufferRutaTemporal = new Vector<GPSData>();
	/** Si estamos {@link #enRuta} almacena los puntos que estén separados al menos {@link #minDistOperativa} */
	private Vector<GPSData> bufferRutaEspacial = new Vector<GPSData>();


	/**
	 * Constructor por defecto no hace nada.
	 * Para usar el puerto hay que invocar a {@link #setParameters(SerialParameters)} 
	 * y luego {@link #openConnection()}
	 */
	public GPSConnection() {
		//lastPaquete = System.currentTimeMillis();
	}

	/**
	 * Crea conexión a GPS en puerto serial indicado.
	 * Se utilizan los parámetros <code>SerialParameters(portName, 9600, 0, 0, 8, 1, 0)</code>
	 * Si se quieren especificar otros parámetros se debe utilizar
	 * el {@link #GPSConnection() constructor por defecto}.
	 * @param portName nombre puerto donde encontrar al GPS
	 */
	public GPSConnection(String portName) throws SerialConnectionException {
		parameters = new SerialParameters(portName, 9600, 0, 0, 8, 1, 0);
		openConnection();
		if (isOpen()) {
			System.out.println("Puerto Abierto " + portName);
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
     Sets the connection parameters to the setting in the {@link #parameters} object.
         If set fails return the parameters object to original settings and
         throw exception.
	 */
	public void setConnectionParameters() throws SerialConnectionException {

		// Save state of parameters before trying a set.
		int oldBaudRate = sPort.getBaudRate();
		int oldDatabits = sPort.getDataBits();
		int oldStopbits = sPort.getStopBits();
		int oldParity = sPort.getParity();
		//int oldFlowControl = sPort.getFlowControlMode();

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

	/**
	 * Maneja los eventos seriales {@link SerialPortEvent#DATA_AVAILABLE}.
	 * Si se recibe un mensaje completo del GPS {@link #procesaCadena(String)}
	 */
	public synchronized void serialEvent(SerialPortEvent e) {
		if (e.getEventType() == SerialPortEvent.DATA_AVAILABLE) {
			try {
				while (is.available() != 0) {
					int val = is.read();
					if (val != 10) {
						cadenaTemp += (char) val;
					} else {
						procesaCadena(cadenaTemp);
						cadenaTemp = "";
					}
				}
			} catch (IOException ioe) {
				System.err.println("\nError al recibir los datos");
			} catch (Exception ex) {
				System.err.println("\nGPSConnection Error: Cadena fragmentada " + ex.getMessage());
				cadenaTemp = "";
			}
		}
	}

	/**
	 * Interpreta la cadena recibida del GPS y obtiene los distintos datos contenidos.
	 * Al final invoca a {@link #calculaValores(String)}.
	 * Trata los mensajes GSA, GST, VTG y GGA.
	 * @param cadena
	 */
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
	 * Calcula varios valores sobre el punto actual y actualiza los buffers. 
	 * Al final crea un nuevo {@link #data} para almacenar al siguiente punto.
	 */
	private void calculaValores(String cadena) {
		data.setSysTime(System.currentTimeMillis());
		setECEF();
		setCoordenadasLocales();
		anadeBufferTemporal();
		anadeBufferEspacial();
		//data = new GPSData();
                data = (GPSData)data.clone();
	}

	/**
	 * Actualiza las coordenadas locales de todos los elementos de los buffers y rutas 
	 * que estén creadas.
	 */
	private void updateBuffers() {

		if (rutaEspacial != null && rutaEspacial.size() != 0) {
			for (int i = 0; i < rutaEspacial.size(); i++)
				setCoordenadasLocales(rutaEspacial.elementAt(i));
		}

		if (rutaTemporal != null && rutaTemporal.size() != 0) {
			for (int i = 0; i < rutaTemporal.size(); i++) 
				setCoordenadasLocales(rutaTemporal.elementAt(i)); 
		}

		if (bufferRutaEspacial != null && bufferRutaEspacial.size() != 0) {
			for (int i = 0; i < bufferRutaEspacial.size(); i++)
				setCoordenadasLocales(bufferRutaEspacial.elementAt(i));
		}

		if (bufferRutaTemporal != null && bufferRutaTemporal.size() != 0) {
			for (int i = 0; i < bufferRutaTemporal.size(); i++)
				setCoordenadasLocales(bufferRutaTemporal.elementAt(i));
		}

		if (bufferEspacial != null && bufferEspacial.size() != 0) {
			for (int i = 0; i < bufferEspacial.size(); i++)
				setCoordenadasLocales(bufferEspacial.elementAt(i));
		}

		if (bufferTemporal != null && bufferTemporal.size() != 0) {
			for (int i = 0; i < bufferTemporal.size(); i++)
				setCoordenadasLocales(bufferTemporal.elementAt(i));
		}

	}

	/**
	 * Calcula el {@link #centro} de la ruta y prepara {@link #T las matriz de traslación}.
	 * Al final se llama a {@link #updateBuffers()}.
	 * @param ruta sobre la que se calcula en centro.
	 */
	private void setParams(Vector<GPSData> ruta) {
		if (ruta == null || ruta.size() == 0)
			return;                

		// Primero buscamos el punto central exacto
		double xCentral = 0, yCentral = 0, zCentral = 0;
		for (int i = 0; i < ruta.size(); i++) {
			xCentral += ruta.elementAt(i).getX();
			yCentral += ruta.elementAt(i).getY();
			zCentral += ruta.elementAt(i).getZ();            
		}
		xCentral /= ruta.size();
		yCentral /= ruta.size();
		zCentral /= ruta.size();

		// Ahora buscamos el punto que más cerca esté de ese centro
		double dist = Math.sqrt(Math.pow(ruta.elementAt(0).getX() - xCentral, 2.0f) + 
				Math.pow(ruta.elementAt(0).getY() - yCentral, 2.0f) + 
				Math.pow(ruta.elementAt(0).getZ() - zCentral, 2.0f));
		centro = ruta.elementAt(0);
		for (int i = 0; i < ruta.size(); i++) {
			double myDist = Math.sqrt(Math.pow(ruta.elementAt(i).getX() - xCentral, 2.0f) + 
					Math.pow(ruta.elementAt(i).getY() - yCentral, 2.0f) + 
					Math.pow(ruta.elementAt(i).getZ() - zCentral, 2.0f));
			if (myDist < dist) {
				dist = myDist;
				centro = ruta.elementAt(i);
			}            
		}

		// Matriz de rotación en torno a un punto
		double v[][] = new double[3][];
		v[0] = new double[] { -Math.sin(centro.getLongitud()), Math.cos(centro.getLongitud()), 0 };
		v[1] = new double[] { -Math.cos(centro.getLongitud()) * Math.sin(centro.getLatitud()), -Math.sin(centro.getLatitud()) * Math.sin(centro.getLongitud()), Math.cos(centro.getLatitud()) };
		v[2] = new double[] { Math.cos(centro.getLatitud()) * Math.cos(centro.getLongitud()), Math.cos(centro.getLatitud()) * Math.sin(centro.getLongitud()), Math.sin(centro.getLatitud())};

		Matrix M1 = new Matrix(v);

		// Matriz de inversión del eje z en torno al eje x (Norte)
		double w[][] = new double[3][];
		w[0] = new double[] { -1, 0, 0 };
		w[1] = new double[] { 0, 1, 0 };
		w[2] = new double[] { 0, 0, -1 };
		Matrix M2 = new Matrix(w);

		T = M2.times(M1);  

		updateBuffers();
	}

	/**
	 * Calcula y actualiza las coordenadas x,y,z (ECEF) del úlmimo punto (en {@link #data}).
	 */
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

	/**
	 * Calcula y actualiza las coordenadas locales del nuevo punto (en {@link #data}).
	 * Hace uso de la {@link #T las matriz de traslación}. Si no existe aún, y hay más de 100 puntos, 
	 * llama a {@link #setParams(Vector)} para calcularla.
	 */
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

	/**
	 * Calcula y actualiza las coordenadas locales del punto pasado.
	 * Hace uso de la {@link #T las matriz de traslación}.
	 * @param pto punto a actualizar
	 */
	public void setCoordenadasLocales(GPSData pto) {
		if (T == null) {       
			pto.setXLocal(-1);
			pto.setYLocal(-1);
			pto.setZLocal(-1);
			return;
		}

		Matrix res = new Matrix(new double[][] { {pto.getX() - centro.getX()}, 
				{pto.getY() - centro.getY()}, 
				{pto.getZ() - centro.getZ()} });            
		res = T.times(res).transpose();
		pto.setXLocal(res.getArray()[0][0]);
		pto.setYLocal(res.getArray()[0][1]);
		pto.setZLocal(res.getArray()[0][2]);                    
	}

	/**
	 * Añade el punto en {@link #data} al {@link #bufferTemporal} y si se está {@link #enRuta} 
	 * también al {@link #bufferRutaTemporal}.
	 */
	private void anadeBufferTemporal() {
		calculaAngSpeed(bufferTemporal, data);
		bufferTemporal.add(data);

		if (enRuta) {
			bufferRutaTemporal.add(data);
		}

		while (bufferTemporal.size() > GPSConnection.MAXBUFFER) {
			bufferTemporal.remove(0);
		}
	}

	/**
	 * Añade {@link #data} al {@link #bufferEspacial} si está a distancia mayor que {@link #minDistOperativa}
	 * del último puto almacenado.
	 * Si estamos {@link #enRuta} también se añade a {@link #bufferRutaEspacial}
	 */
	private void anadeBufferEspacial() {       
		if (bufferEspacial.size() == 0) {
			calculaAngSpeed(bufferEspacial, data);
			bufferEspacial.add(data);
			return;
		}

		double dist = Math.sqrt(Math.pow(data.getX() - bufferEspacial.lastElement().getX(), 2.0f) + 
				Math.pow(data.getY() - bufferEspacial.lastElement().getY(), 2.0f) + 
				Math.pow(data.getZ() - bufferEspacial.lastElement().getZ(), 2.0f));
		if (dist > minDistOperativa) {
			calculaAngSpeed(bufferEspacial, data);
			bufferEspacial.add(data);

			if (enRuta) {
				bufferRutaEspacial.add(data);
			}

			while (bufferEspacial.size() > GPSConnection.MAXBUFFER) {
				bufferEspacial.remove(0);
			}
		}
	}


	/**
	 * Calcula y actualiza el agulo y la velocidad del punto <code>val</code> con 
	 * respecto a el último punto del <code>buffer</code> 
	 * @param buffer
	 * @param val punto que se usa para el calculo y se actualiza con velocidad y ángulo calculados
	 */
	private void calculaAngSpeed(Vector<GPSData> buffer, GPSData val) {
		if (buffer.size() == 0) {
			val.setAngulo(0);
			val.setVelocidad(0);
			return;
		}
		double x = val.getXLocal() - buffer.lastElement().getXLocal();
		double y = val.getYLocal() - buffer.lastElement().getYLocal();       

		double ang = Math.atan2(x, y);
		if (ang < 0) ang += 2 * Math.PI;

		// En principio no diferencio entre angulo y angulo local
		val.setAngulo(ang);        

		double vel = Math.sqrt(Math.pow(x , 2.0f) + Math.pow(y , 2.0f));
		vel /= (val.getSysTime() - buffer.lastElement().getSysTime()) / 1000.0;
		val.setVelocidad(vel);                
	}

	/**
	 * Convierte cadena de caracteres que representa grados a double correspondiente
	 * @param valor cadena a convertir
	 * @param enteros número de dígitos que corresponden a grados (resto son minutos)
	 * @return grados representados por la cadena
	 */
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

	/**
	 * Convierte cadena de tiempo recibida del GPS al formato hh:mm:ss
	 * @param cadena
	 * @return
	 */
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

	/**
	 * @param minDistOperativa nuevo valor para {@link #minDistOperativa}
	 */
	public void setMinDistOperativa(double minDistOperativa) {
		GPSConnection.minDistOperativa = minDistOperativa;
	}

	/**
	 * @return valor de {@link #minDistOperativa}
	 */
	public double getMinDistOperativa() {
		return GPSConnection.minDistOperativa;
	}

	/**
	 * @return último punto del {@link #bufferEspacial}. null si no hay.
	 */
	public GPSData getPuntoActualEspacial() {
		if (bufferEspacial == null || bufferEspacial.size() == 0) return null;
		return bufferEspacial.lastElement();
	}

	/**
	 * @return último punto del {@link #bufferTemporal}. null si no hay.
	 */
	public GPSData getPuntoActualTemporal() {
		if (bufferTemporal == null || bufferTemporal.size() == 0) return null;
		return (GPSData)(bufferTemporal.lastElement());
	}

	/**
	 * Devuelve lós último <code>n</code> puntos del del {@link #bufferEspacial}.
	 * @param n
	 * @return vector con los puntos
	 */
	public Vector<GPSData> getLastPuntosEspacial(int n) {
		if (bufferEspacial == null || bufferEspacial.size() == 0) return null;

		Vector<GPSData> retorno = new Vector<GPSData>();
		Vector<GPSData> buffer = (Vector<GPSData>)(bufferEspacial.clone());

		for (int i = n; i > 0; i--) {          
			if (buffer.size() - i < 0) continue;
			retorno.add(buffer.elementAt(buffer.size() - i));
		}        

		return retorno;
	}

	/**
	 * Devuelve los <code>n</code> últimos puntos del {@link #bufferTemporal}
	 * @param n
	 * @return vector con los puntos
	 */
	public Vector<GPSData> getLastPuntosTemporal(int n) {
		if (bufferTemporal == null || bufferTemporal.size() == 0) return null;

		Vector<GPSData> retorno = new Vector<GPSData>();
		Vector<GPSData> buffer = (Vector<GPSData>)(bufferTemporal.clone());

		for (int i = n; i > 0; i--) {          
			if (buffer.size() - i < 0) continue;
			retorno.add(buffer.elementAt(buffer.size() - i));
		}   

		return retorno;
	}

	/** @return {@link #bufferEspacial}	 */
	public Vector<GPSData> getBufferEspacial() {
		return bufferEspacial;
	}

	/** @return {@link #bufferTemporal}	 */
	public Vector<GPSData> getBufferTemporal() {
		return bufferTemporal;
	}

	/** @return {@link #bufferRutaEspacial}	 */
	public Vector<GPSData> getBufferRutaEspacial() {
		return bufferRutaEspacial;
	}

	/** @return {@link #bufferRutaTemporal} */
	public Vector<GPSData> getBufferRutaTemporal() {        
		return bufferRutaTemporal;
	}

	/** @return {@link #rutaEspacial} */
	public Vector<GPSData> getRutaEspacial() {
		return rutaEspacial;
	}

	/** @return {@link #rutaTemporal} */
	public Vector<GPSData> getRutaTemporal() {        
		return rutaTemporal;
	}

	/**
	 * Comienza la captura de nuevas rutas {@link #bufferRutaEspacial espacial} 
	 * y {@link #bufferRutaTemporal temporal}.
	 */
	public void startRuta() {
		bufferRutaEspacial = new Vector<GPSData>();
		bufferRutaTemporal = new Vector<GPSData>();

		enRuta = true;
	}    

	/**
	 * Detine la actualización de las rutas especial y temporal
	 */
	public void stopRuta() {
		enRuta = false;
	}

	/**
	 * Detiene actualización de las rutas especial y temporal y las salva en fichero indicado
	 * @param fichero
	 */
	public void stopRutaAndSave(String fichero) {
		enRuta = false;
		saveRuta(fichero);
	}

	/**
	 * Salva las rutas {@link #bufferRutaEspacial espacial} y {@link #bufferRutaTemporal temporal}
	 * en formato binario en el fichero indicado.
	 * @param fichero
	 */
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

	/**
	 * Carga {@link #rutaEspacial} y {@link #rutaTemporal} desde fichero indicado.
	 * Deben estar salvadas en formato binario (con {@link #saveRuta(String)})
	 * @param fichero
	 */
	public void loadRuta(String fichero) {
		rutaEspacial = new Vector<GPSData>();
		rutaTemporal = new Vector<GPSData>();
		//    Vector valores = new Vector();
		try {
			File file = new File(fichero);
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file));
			int tamEspacial = -1;
			int tamTemporal = -1;
			tamEspacial = ois.readInt();
			tamTemporal = ois.readInt();
			for (int i = 0; i < tamEspacial; i++) {
				rutaEspacial.add((GPSData)ois.readObject());
			}
			for (int i = 0; i < tamTemporal; i++) {
				rutaTemporal.add((GPSData)ois.readObject());
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

	/**
	 * Carga {@link #rutaTemporal} de fichero en formato anterior (no binario).
	 * @param fichero
	 */
	public void loadOldRuta(String fichero) {
		rutaTemporal = new Vector<GPSData>();
		//    Vector valores = new Vector();
		try {
			File file = new File(fichero);
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file));
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

	/**
	 * @return the parameters
	 */
	public SerialParameters getParameters() {
		return parameters;
	}

	/**
	 * @param parameters the parameters to set
	 */
	public void setParameters(SerialParameters parameters) {
		this.parameters = parameters;
	}

}

