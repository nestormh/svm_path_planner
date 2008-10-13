package sibtra.triumph;

import java.io.IOException;
import java.io.InputStream;
import java.util.TooManyListenersException;

import sibtra.gps.SerialConnectionException;
import sibtra.gps.SerialParameters;
import gnu.io.CommPortIdentifier;
import gnu.io.NoSuchPortException;
import gnu.io.PortInUseException;
import gnu.io.SerialPort;
import gnu.io.SerialPortEvent;
import gnu.io.SerialPortEventListener;
import gnu.io.UnsupportedCommOperationException;

/** Clase para probar los mensajes estandar de la familia Triumph
 * Como:
 * out,,jps/RT,jps/PO,jps/ET
 * out,,jps/RT,jps/PO,jps/RD
 * @author alberto
 *
 */

public class GPSConnection implements SerialPortEventListener {

	
	/** Tamaño máximo del mensaje */
	private static final int MAXMENSAJE = 1000;

//	private static final double NULLANG = -5 * Math.PI;


	/** si el puerto serial está abierto*/
	private boolean open;

	private SerialParameters parameters;    
	private InputStream is;    

	private CommPortIdentifier portId;
	private SerialPort sPort;

	/** Cadena en la que se van almacenando los trozos de mensajes que se van recibiendo */
	private byte cadenaTemp[] = new byte[MAXMENSAJE];

	private int indCadenaTemp;
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
		parameters = new SerialParameters(portName, 115200, 0, 0, 8, 1, 0);
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
	 * Maneja los eventos seriales {@link SerialPortEvent#DATA_AVAILABLE}.
	 * Si se recibe un mensaje completo del GPS {@link #actualizaNuevaCadena(String)}
	 */
	public synchronized void serialEvent(SerialPortEvent e) {
		if (e.getEventType() == SerialPortEvent.DATA_AVAILABLE) {
			try {
				while (is.available() != 0) {
					int val = is.read();
					//añadimos nuevo byte recibido
					indCadenaTemp++;
					cadenaTemp[indCadenaTemp]= (byte) val;
					//vemos si ya terminó la cadena
					if (indCadenaTemp>0 
							&& (cadenaTemp[0]>=33 && cadenaTemp[0]<=47) 
							&& (cadenaTemp[indCadenaTemp] == 10 || cadenaTemp[indCadenaTemp]==13)
					) {
						//Tenemos un mensaje de texto
						actualizaNuevaCadenaTexto();
						indCadenaTemp=-1;
					}
					else if (indCadenaTemp>=4) { //ya tenemos longitud
						int logitud=(cadenaTemp[2]-0x30)*32+(cadenaTemp[3]-0x30)*16+(cadenaTemp[2]-0x30);
						if(indCadenaTemp==(logitud+4)) {
							//tenemos el mensaje estandar completo
							actualizaNuevaCadenaBinaria();						
							indCadenaTemp=-1;
						}
					}
				}
			} catch (IOException ioe) {
				System.err.println("\nError al recibir los datos");
			} catch (Exception ex) {
				System.err.println("\nGPSConnection Error al procesar >"+cadenaTemp+"< : " + ex.getMessage());
				ex.printStackTrace();
				indCadenaTemp=-1;
			}
		}
	}

	/**
	 * Actuliza toda los buffers con la información contenida en la nueva cadena recibida.
	 * Fija el sistema local al primer punto de buffer espacial.
	 * @param cadena cadena recibida (del GPS)
	 */
	void actualizaNuevaCadenaTexto() {
		
	}
	void actualizaNuevaCadenaBinaria() {
		
	}
}
