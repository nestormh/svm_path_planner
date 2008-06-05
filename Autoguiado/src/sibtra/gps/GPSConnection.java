package sibtra.gps;

import gnu.io.*;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.TooManyListenersException;

/**
 * Maneja la conexión serial con el GPS. 
 * Al recibir un paquete recoge la información del nuevo punto y lo almacena en los distintos
 * buffers. 
 */
public class GPSConnection implements SerialPortEventListener {

	
	/** Numero máximo de puntos acumulados en el buffer de ruta */
	private static final int MAXBUFFER = 1000;

//	private static final double NULLANG = -5 * Math.PI;


	/** si el puerto serial está abierto*/
	private boolean open;

	private SerialParameters parameters;    
	private InputStream is;    

	private CommPortIdentifier portId;
	private SerialPort sPort;

	/** Cadena en la que se van almacenando los trozos de mensajes que se van recibiendo */
	private String cadenaTemp = "";
	
	/** último punto recibido */
	private GPSData data = new GPSData();    


	/** Almacena ruta temporal cargada de fichero */
	private Ruta rutaTemporal = null;
	/** Almacena ruta espacial cargada de fichero */
	private Ruta rutaEspacial = null;

	/** Almacena los {@link #MAXBUFFER} últimos puntos que están separados al menos {@link #minDistOperativa} */
	private Ruta bufferEspacial = new Ruta(MAXBUFFER,true);
	/** Almacena los {@link #MAXBUFFER} últimos puntos recibidos */
	private Ruta bufferTemporal = new Ruta(MAXBUFFER);  

	/** indica si estamos capturando una ruta */
	private boolean enRuta = false;
	
	/** Si estamos {@link #enRuta}, almacena los puntos recibidos */
	private Ruta bufferRutaTemporal = null;
	/** Si estamos {@link #enRuta} almacena los puntos que estén separados al menos {@link #minDistOperativa} */
	private Ruta bufferRutaEspacial = null;


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
						actualizaNuevaCadena(cadenaTemp);
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
	 * Actuliza toda los buffers con la información contenida en la nueva cadena recibida.
	 * @param cadena cadena recibida (del GPS)
	 */
	void actualizaNuevaCadena(String cadena) {
		
		data.procesaCadena(cadena);
		data.setSysTime(System.currentTimeMillis());
		data.setECEF();
		if(!bufferEspacial.tieneSistemaLocal() && bufferEspacial.getNumPuntos()>100) {
			bufferEspacial.actualizaSistemaLocal();
			updateBuffers(bufferEspacial);
		}
		bufferEspacial.setCoordenadasLocales(data);

		añadeABuffers();
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
		bufferEspacial.actualizaCoordenadasLocales();
		Ruta ba;
		ba=bufferEspacial;
		if(ba!=null && ba!=rutaRef) {
			//usamos sistema local de bufferEspacial
			ba.actualizaSistemaLocal(bufferEspacial);
			//actualizamos todos los puntos
			ba.actualizaCoordenadasLocales();
		}
		ba=bufferTemporal;
		if(ba!=null && ba!=rutaRef) {
			//usamos sistema local de bufferEspacial
			ba.actualizaSistemaLocal(bufferEspacial);
			//actualizamos todos los puntos
			ba.actualizaCoordenadasLocales();
		}
		ba=bufferRutaEspacial;
		if(ba!=null && ba!=rutaRef) {
			ba.actualizaSistemaLocal(bufferEspacial);
			ba.actualizaCoordenadasLocales();
		}
		ba=bufferRutaTemporal;
		if(ba!=null && ba!=rutaRef) {
			ba.actualizaSistemaLocal(bufferEspacial);
			ba.actualizaCoordenadasLocales();
		}
		ba=rutaEspacial;
		if(ba!=null && ba!=rutaRef) {
			ba.actualizaSistemaLocal(bufferEspacial);
			ba.actualizaCoordenadasLocales();
		}
		ba=rutaTemporal;
		if(ba!=null && ba!=rutaRef) {
			ba.actualizaSistemaLocal(bufferEspacial);
			ba.actualizaCoordenadasLocales();
		}
	}

//	/**
//	 * Calcula el {@link #centro} de la ruta y prepara {@link #T las matriz de traslación}.
//	 * Al final se llama a {@link #updateBuffers()}.
//	 * @param ruta sobre la que se calcula en centro.
//	 */
//	private void setParams(Ruta ruta) {
//		ruta.actualizaLocal();
//		updateBuffers();
//	}

//	/**
//	 * Calcula y actualiza las coordenadas locales del nuevo punto (en {@link #data}).
//	 * Hace uso de la {@link #T las matriz de traslación}. Si no existe aún, y hay más de 100 puntos, 
//	 * llama a {@link #setParams(Vector)} para calcularla.
//	 */
//	public void setCoordenadasLocales() {
//	}


	/**
	 * Añade el punto en {@link #data} al {@link #bufferTemporal} y {@link #bufferEspacial} y si 
	 * se está {@link #enRuta} también al {@link #bufferRutaTemporal} y {@link #bufferRutaEspacial}
	 */
	private void añadeABuffers() {
		bufferEspacial.add(data);
		bufferTemporal.add(data);
		if(enRuta) {
			bufferRutaEspacial.add(data);
			bufferRutaTemporal.add(data);
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

//	/**
//	 * Devuelve lós último <code>n</code> puntos del del {@link #bufferEspacial}.
//	 * @param n
//	 * @return vector con los puntos
//	 */
//	public Vector<GPSData> getLastPuntosEspacial(int n) {
//		if (bufferEspacial == null || bufferEspacial.size() == 0) return null;
//
//		Vector<GPSData> retorno = new Vector<GPSData>();
//		Vector<GPSData> buffer = (Vector<GPSData>)(bufferEspacial.clone());
//
//		for (int i = n; i > 0; i--) {          
//			if (buffer.size() - i < 0) continue;
//			retorno.add(buffer.elementAt(buffer.size() - i));
//		}        
//
//		return retorno;
//	}
//
//	/**
//	 * Devuelve los <code>n</code> últimos puntos del {@link #bufferTemporal}
//	 * @param n
//	 * @return vector con los puntos
//	 */
//	public Vector<GPSData> getLastPuntosTemporal(int n) {
//		if (bufferTemporal == null || bufferTemporal.size() == 0) return null;
//
//		Vector<GPSData> retorno = new Vector<GPSData>();
//		Vector<GPSData> buffer = (Vector<GPSData>)(bufferTemporal.clone());
//
//		for (int i = n; i > 0; i--) {          
//			if (buffer.size() - i < 0) continue;
//			retorno.add(buffer.elementAt(buffer.size() - i));
//		}   
//
//		return retorno;
//	}

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
			rutaTemporal=(Ruta)ois.readObject();
			ois.close();
		} catch (IOException ioe) {
			System.err.println("Error al abrir el fichero " + fichero);
			System.err.println(ioe.getMessage());
		} catch (ClassNotFoundException cnfe) {
			System.err.println("Objeto leído inválido: " + cnfe.getMessage());            
		}     

		rutaEspacial.actualizaSistemaLocal();
		updateBuffers(rutaEspacial);
	}

//	/**
//	 * Carga {@link #rutaTemporal} de fichero en formato anterior (no binario).
//	 * @param fichero
//	 */
//	public void loadOldRuta(String fichero) {
//		rutaTemporal = new Vector<GPSData>();
//		//    Vector valores = new Vector();
//		try {
//			File file = new File(fichero);
//			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file));
//			while (ois.available() != 0) {
//				GPSData valor = new GPSData();
//				valor.setX(ois.readDouble());
//				valor.setY(ois.readDouble());
//				valor.setZ(ois.readDouble());
//				valor.setLatitud(ois.readDouble());
//				valor.setLongitud(ois.readDouble());
//				valor.setAltura(ois.readDouble());
//				valor.setAngulo(ois.readDouble());
//				valor.setVelocidad(ois.readDouble());
//
//				rutaTemporal.add(valor);
//			}
//			ois.close();
//		} catch (IOException ioe) {
//			System.err.println("Error al abrir el fichero " + fichero);
//			System.err.println(ioe.getMessage());
//		}
//
//		setParams(rutaTemporal);
//	}

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
