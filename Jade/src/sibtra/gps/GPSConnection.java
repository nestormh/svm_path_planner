package sibtra.gps;

import gnu.io.*;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.TooManyListenersException;

import sibtra.imu.ConexionSerialIMU;

/**
 * Maneja la conexión serial con el GPS. 
 * Al recibir un paquete recoge la información del nuevo punto y lo almacena en los distintos
 * buffers. Si se define conexión a la IMU se almacena información águlos en el momento.
 */
public class GPSConnection implements SerialPortEventListener {

	
	/** Numero máximo de puntos acumulados en el buffer de ruta */
	private static final int MAXBUFFER = 1000;

//	private static final double NULLANG = -5 * Math.PI;


	/** si el puerto serial está abierto*/
	boolean open;

	/** Los parámetros seriales instalados en el puerto */
	SerialParameters parameters=null;
	
	InputStream is;    
	OutputStream os;

	CommPortIdentifier portId;
	SerialPort sPort;

	/** Cadena en la que se van almacenando los trozos de mensajes que se van recibiendo */
	String cadenaTemp = "";
	
	/** último punto recibido */
	GPSData data = new GPSData();    


	/** Almacena ruta temporal cargada de fichero */
	Ruta rutaTemporal = null;
	/** Almacena ruta espacial cargada de fichero */
	Ruta rutaEspacial = null;

	/** Almacena los {@link #MAXBUFFER} últimos puntos de la ruta espacial */
	Ruta bufferEspacial = new Ruta(MAXBUFFER,true);
	/** Almacena los {@link #MAXBUFFER} últimos puntos recibidos */
	Ruta bufferTemporal = new Ruta(MAXBUFFER);  

	/** indica si estamos capturando una ruta */
	boolean enRuta = false;
	
	/** Si estamos {@link #enRuta}, almacena los puntos recibidos */
	Ruta bufferRutaTemporal = null;
	/** Si estamos {@link #enRuta} almacena los puntos en ruta espacial */
	Ruta bufferRutaEspacial = null;
	
	/** Conexión serial IMU de la que leer ángulos*/
	ConexionSerialIMU csIMU=null;

	/** contador de paquetes recibidos del GPS */
	int cuentaPaquetesRecibidos=0;

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
	 * Se utilizan los parámetros <code>SerialParameters(portName, 115200, 0, 0, 8, 1, 0)</code>
	 * Si se quieren especificar otros parámetros se debe utilizar
	 * el {@link #GPSConnection() constructor por defecto}.
	 * @param portName nombre puerto donde encontrar al GPS
	 */
	public GPSConnection(String portName) throws SerialConnectionException {
		this(portName,115200);
	}
	/**
	 * Crea conexión a GPS en puerto serial indicado.
	 * Se utilizan los parámetros <code>SerialParameters(portName, baudios, 0, 0, 8, 1, 0)</code>
	 * Si se quieren especificar otros parámetros se debe utilizar
	 * el {@link #GPSConnection() constructor por defecto}.
	 * @param portName nombre puerto donde encontrar al GPS
	 * @param baudio velocidad de la comunicacion en baudios
	 */
	public GPSConnection(String portName, int baudios) throws SerialConnectionException {
		openConnection(new SerialParameters(portName, baudios, 0, 0, 8, 1, 0));
		if (isOpen()) {
			System.out.println("Puerto Abierto " + portName);
		}
	}

	/**
        Attempts to open a serial connection and streams using the parameters
        in the SerialParameters object. If it is unsuccesfull at any step it
        returns the port to a closed state, throws a
        <code>SerialConnectionException</code>, and returns.

     Gives a timeout of 30 seconds on the portOpen to allow other applications
        to reliquish the port if have it open and no longer need it.
	 */
	public void openConnection(SerialParameters serPar) throws SerialConnectionException {
		parameters=serPar;
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
			sPort = (SerialPort) portId.open("Sibtra", 30000);
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

		//Obtenemos el flujo de salida
		try {
			os = sPort.getOutputStream();
		} catch (IOException e) {
			System.err.println("\n No se pudo obtener flujo de salida para puerto ");
			throw new SerialConnectionException("Error obteniendo flujo de salida");
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
	protected void setConnectionParameters() throws SerialConnectionException {

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
			sPort.setInputBufferSize(1);
//			sPort.setLowLatency();
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
	 * Si se recibe un mensaje completo del GPS {@link #nuevaCadenaNMEA(String)}
	 * Sirve para la gestión de mensajes NMEA
	 */
	public synchronized void serialEvent(SerialPortEvent e) {
		if (e.getEventType() == SerialPortEvent.DATA_AVAILABLE) {
			try {
				while (is.available() != 0) {
					int val = is.read();
					if (val != 10) {
						cadenaTemp += (char) val;
					} else {
						nuevaCadenaNMEA(cadenaTemp);						
						cadenaTemp = "";
					}
				}
			} catch (IOException ioe) {
				System.err.println("\nError al recibir los datos");
			} catch (Exception ex) {
				System.err.println("\nGPSConnection Error al procesar >"+cadenaTemp+"< : " + ex.getMessage());
				ex.printStackTrace();
				cadenaTemp = "";
			}
		}
	}


	/**
	 * Actuliza toda los buffers con la información contenida en la nueva cadena recibida
	 *  sólo si es GGA válido.
	 * Fija el sistema local al primer punto de buffer espacial.
	 * @param cadena cadena recibida (del GPS)
	 */
	void nuevaCadenaNMEA(String cadena) {
		
		if(!data.procesaCadena(cadena))
			return;  //sólo será nuevo punto si es paquete GGA valido
		
		data.setSysTime(System.currentTimeMillis());
		data.calculaECEF();
		nuevoPunto();
	}
	
	/** Gestiona acciones a realizar cuando se tiene nuevo punto. 
	 * El nuevo punto está en campo {@link #data} 
	 */
	void nuevoPunto() {
		if (csIMU!=null)  //si existe conexión seriañ a la IMU copiamos el ángulo  
			data.setAngulosIMU(csIMU.getAngulo());
		else
			data.setAngulosIMU(null);
		if(!bufferEspacial.tieneSistemaLocal())  {
			System.out.println("Se actuliza local de buffer Espacial");
			bufferEspacial.actualizaSistemaLocal(data);
			updateBuffers(bufferEspacial);
		}
		bufferEspacial.setCoordenadasLocales(data);

		/**
		 * Añade el punto en {@link #data} al {@link #bufferTemporal} y {@link #bufferEspacial} y si 
		 * se está {@link #enRuta} también al {@link #bufferRutaTemporal} y {@link #bufferRutaEspacial} 
		 * Si se está {@link #enRuta} se usa primer punto de ruta espacial para fijar sistema local
		 * de todos.
		 * 
		 */
		boolean seAñadeEspacial=bufferEspacial.add(data);
		bufferTemporal.add(data);
		if(enRuta) {
			bufferRutaEspacial.add(data);
			bufferRutaTemporal.add(data);
			if(!bufferRutaEspacial.tieneSistemaLocal()) {
				//sistema local con primer punto de la ruta espacial
				bufferRutaEspacial.actualizaSistemaLocal(bufferRutaEspacial.getPunto(0));
				//todos los demás con ese sistema local
				updateBuffers(bufferRutaEspacial);
			}

		}
		avisaListeners(seAñadeEspacial); //avisamos a todos los listeners
        data = new GPSData(data); //creamos nuevo punto copia del anterior

	}
	
	/**
	 * Actualiza las coordenadas locales de todos los elementos de los buffers y rutas 
	 * que estén creadas. Nos basamos en sistema local de pasado como parámetro.
	 * @param rutaRef ruta cuyo sistema local se usa como referencia.
	 */
	private void updateBuffers(Ruta rutaRef) {
		if(rutaRef==null)
			return;

		rutaRef.actualizaCoordenadasLocales();
		Ruta ba;
		ba=bufferEspacial;
		if(ba!=null && ba!=rutaRef) {
			ba.actualizaSistemaLocal(rutaRef);
			ba.actualizaCoordenadasLocales();
		}
		ba=bufferTemporal;
		if(ba!=null && ba!=rutaRef) {
			ba.actualizaSistemaLocal(rutaRef);
			ba.actualizaCoordenadasLocales();
		}
		ba=bufferRutaEspacial;
		if(ba!=null && ba!=rutaRef) {
			ba.actualizaSistemaLocal(rutaRef);
			ba.actualizaCoordenadasLocales();
		}
		ba=bufferRutaTemporal;
		if(ba!=null && ba!=rutaRef) {
			ba.actualizaSistemaLocal(rutaRef);
			ba.actualizaCoordenadasLocales();
		}
		ba=rutaEspacial;
		if(ba!=null && ba!=rutaRef) {
			ba.actualizaSistemaLocal(rutaRef);
			ba.actualizaCoordenadasLocales();
		}
		ba=rutaTemporal;
		if(ba!=null && ba!=rutaRef) {
			ba.actualizaSistemaLocal(rutaRef);
			ba.actualizaCoordenadasLocales();
		}
	}


	/**
	 * @return último punto del {@link #bufferEspacial}. null si no hay.
	 */
	public GPSData getPuntoActualEspacial() {
		return bufferEspacial.getUltimoPto();
	}

	/**
	 * @return último punto del {@link #bufferTemporal}. null si no hay.
	 */
	public GPSData getPuntoActualTemporal() {
		return bufferTemporal.getUltimoPto();
	}


	/** @return {@link #bufferEspacial}	 */
	public Ruta getBufferEspacial() {
		return bufferEspacial;
	}

	/** @return {@link #bufferTemporal}	 */
	public Ruta getBufferTemporal() {
		return bufferTemporal;
	}

	/** @return {@link #bufferRutaEspacial}	 */
	public Ruta getBufferRutaEspacial() {
		return bufferRutaEspacial;
	}

	/** @return {@link #bufferRutaTemporal} */
	public Ruta getBufferRutaTemporal() {        
		return bufferRutaTemporal;
	}

	/** @return {@link #rutaEspacial} */
	public Ruta getRutaEspacial() {
		return rutaEspacial;
	}

	/** @return {@link #rutaTemporal} */
	public Ruta getRutaTemporal() {        
		return rutaTemporal;
	}

	/**
	 * Comienza la captura de nuevas rutas {@link #bufferRutaEspacial espacial} 
	 * y {@link #bufferRutaTemporal temporal}.
	 */
	public void startRuta() {
		bufferRutaEspacial = new Ruta(true);
		bufferRutaTemporal = new Ruta(false);
		
		if(rutaEspacial!=null) {
			//si se ha cargado una ruta se usa el sistema local de la ruta espacial cargada
			bufferRutaEspacial.actualizaSistemaLocal(rutaEspacial);
			bufferRutaTemporal.actualizaSistemaLocal(rutaEspacial);
		}

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
			oos.writeObject(bufferRutaEspacial);
			oos.writeObject(bufferRutaTemporal);
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
		try {
			File file = new File(fichero);
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file));
			rutaEspacial=(Ruta)ois.readObject();
			rutaEspacial.actualizaSistemaLocal();
			updateBuffers(rutaEspacial);
			rutaTemporal=(Ruta)ois.readObject();
			ois.close();
		} catch (IOException ioe) {
			System.err.println("Error al abrir el fichero " + fichero);
			System.err.println(ioe.getMessage());
		} catch (ClassNotFoundException cnfe) {
			System.err.println("Objeto leído inválido: " + cnfe.getMessage());            
		}     

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
	
	/** @return {@link #cuentaPaquetesRecibidos} */
	public int getCuentaPaquetesRecibidos(){
		return cuentaPaquetesRecibidos;
	}

	
	/** mantiene la lista de listeners */
	private ArrayList<GpsEventListener> listeners = new ArrayList<GpsEventListener>();

	/**
	 * Para añadir objeto a la lista de {@link GpsEventListener}
	 * @param gel objeto a añadir
	 */
	public void addGpsEventListener( GpsEventListener gel ) {
		listeners.add( gel );
	}
	
	/** avisa a todos los listeners con un evento. Siempre se manda evento temporal.
	 * Si ha habido cambio espacial se manda también evento espacial 
	 * @param seAñadeEspacial si se añadió en buffer espacial*/
	private void avisaListeners(boolean seAñadeEspacial) {
	    for ( int j = 0; j < listeners.size(); j++ ) {
	        GpsEventListener gel = listeners.get(j);
	        if ( gel != null ) {
	        	//siempre mandamos evento temporal
	          GpsEvent me = new GpsEvent(this,bufferTemporal.getUltimoPto(),false);
	          gel.handleGpsEvent(me);
	          
	        }
	    }
	    if(seAñadeEspacial)
	    	//mandamos eventos espaciales (si es el caso)
	    	for ( int j = 0; j < listeners.size(); j++ ) {
	    		GpsEventListener gel = listeners.get(j);
	    		if ( gel != null ) {
	    			GpsEvent me = new GpsEvent(this,bufferEspacial.getUltimoPto(),true);
	    			gel.handleGpsEvent(me);

	    		}
	    }
	}

	/** @return la conexión serial a IMU {@link #csIMU} */
	public ConexionSerialIMU getCsIMU() {
		return csIMU;
	}

	/** 
	 * fija valor para {@link #csIMU}. Si es !=null se leerá último angulo para cada nuevo 
	 * dato. 
	 */ 
	public void setCsIMU(ConexionSerialIMU csIMU) {
		this.csIMU = csIMU;
	}
	
}

