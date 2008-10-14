package sibtra.triumph;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.TooManyListenersException;

import sibtra.gps.SerialConnectionException;
import sibtra.gps.SerialParameters;
import sibtra.imu.UtilMensajesIMU;
import sibtra.util.EligeSerial;
import gnu.io.CommPortIdentifier;
import gnu.io.NoSuchPortException;
import gnu.io.PortInUseException;
import gnu.io.SerialPort;
import gnu.io.SerialPortEvent;
import gnu.io.SerialPortEventListener;
import gnu.io.UnsupportedCommOperationException;

/** Clase para probar los mensajes estandar de la familia Triumph
 * Como:
 * 	,jps/PO,jps/ET
 * out,,jps/RT,jps/PO,jps/RD
 * 
 * em,,{jps/RT,nmea/GGA}:2
 * @author alberto
 *
 */

public class GPSConnection implements SerialPortEventListener {

	
	/** Tamaño máximo del mensaje */
	private static final int MAXMENSAJE = 5000;

//	private static final double NULLANG = -5 * Math.PI;


	/** si el puerto serial está abierto*/
	private boolean open;

	private SerialParameters parameters;    
	private InputStream is;    

	private CommPortIdentifier portId;
	private SerialPort sPort;

	/** Buffer en la que se van almacenando los trozos de mensajes que se van recibiendo */
	private byte buff[] = new byte[MAXMENSAJE];

	/** Indice inicial de un mensaje correcto */
	private int indIni;
	/** Indice final de un mensaje correcto */
	private int indFin;
	
	/** largo del mensaje binario */
	private int largoMen;

	/** Banderín que indica que el mensaje es binario */
	private boolean esBinario;

	/** Banderín que indica que el mensaje es de texto */
	private boolean esTexto;

	private OutputStream flujoSalida;

	private static byte ascii0=0x30;
	private static byte ascii9=0x39;
	private static byte asciiA=0x41;
	private static byte asciiF=0x46;

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
		this(portName,115200);
	}

	/**
	 * Crea conexión a GPS en puerto serial indicado.
	 * Se utilizan los parámetros <code>SerialParameters(portName, baudios, 0, 0, 8, 1, 0)</code>
	 * Si se quieren especificar otros parámetros se debe utilizar
	 * el {@link #GPSConnection() constructor por defecto}.
	 * @param portName nombre puerto donde encontrar al GPS
	 * @param baudios baudios de la conexion
	 */
	public GPSConnection(String portName, int baudios) throws SerialConnectionException {
		parameters = new SerialParameters(portName, baudios, 0, 0, 8, 1, 0);
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

		try {
			flujoSalida = sPort.getOutputStream();
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
		indIni=0; indFin=-1; esBinario=false; esTexto=false;
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
					if(indFin==(buff.length-1)) {
						System.err.println("Buffer se llenó. Resetamos");
						indIni=0; indFin=-1; esBinario=false; esTexto=false;
					}
					indFin++;
					buff[indFin]= (byte) val;
					if(esTexto) {
						if ( buff[indFin] == 10 || buff[indFin]==13)
						{
							//mensaje de texto completo
							indFin--; //quitamos caracter del salto
							actualizaNuevaCadenaTexto();
							indIni=0; indFin=-1; esBinario=false; esTexto=false;
						} 
					} else if (esBinario) {
						//terminamos si ya está el tamaño
						if ( (indFin-indIni+1)==(largoMen+5) ) {
							//tenemos el mensaje estandar completo
							actualizaNuevaCadenaBinaria();						
							indIni=0; indFin=-1; esBinario=false; esTexto=false;
						}
					} else { //Todavía no sabemos si es texto o binario
						boolean sincronizado=false;
						while(!sincronizado && (indFin>=indIni)) {
							int larAct=indFin-indIni+1;
							larAct=indFin-indIni+1;
							if (larAct==1) {
								if (isCabTex(indIni)) {
									esTexto=true;
									sincronizado=true;
								}
								else 
									if (!isCabBin(indIni)) {
									//no es cabecera permitida, nos resincronizamos
									indIni=0; indFin=-1; //reiniciamos el buffer
								} else //por ahora puede ser binaria
									sincronizado=true;
								continue;
							} 
							if (larAct==2) {
								if  (!isCabBin(indIni) || !isCabBin(indIni+1))  {
									//no es cabecera permitida, nos resincronizamos
									indIni++;
								} else //por ahora puede ser binaria 
									sincronizado=true;
								continue;
							}
							if (larAct==3) {
								if (!isCabBin(indIni) || !isCabBin(indIni+1) || !isHexa(indIni+2)) {
									//no es caracter hexa, nos resincronizamos
									indIni++;
								} else  //por ahora puede ser binaria 
									sincronizado=true;
								continue;
							}
							if (larAct==4) {
								if (!isCabBin(indIni) || !isCabBin(indIni+1) 
										|| !isHexa(indIni+2) || !isHexa(indIni+3)) {
									//no es caracter hexa, nos resincronizamos
									indIni++;
								} else  //por ahora puede ser binaria 
									sincronizado=true;
								continue;
							} 
							//caso de largo 5
							if (!isCabBin(indIni) || !isCabBin(indIni+1) 
									|| !isHexa(indIni+2) || !isHexa(indIni+3) || !isHexa(indIni+4)) {
								//no es caracter hexa, nos resincronizamos
								indIni++;
							} else { //estamos seguros que es binaria 
								sincronizado=true;
								esBinario=true;
								largoMen=largo();
							}

						}
					}
				}
			} catch (IOException ioe) {
				System.err.println("\nError al recibir los datos");
			} catch (Exception ex) {
				System.err.println("\nGPSConnection Error al procesar >"+buff+"< : " + ex.getMessage());
				ex.printStackTrace();
				indIni=-1;
			}
		}
	}

	private int largo() {
		int lar=0;
		int indAct=indIni+2;
		if(buff[indAct]<ascii9) lar+=32*(buff[indAct]-ascii0);
		else lar+=32*(buff[indAct]-asciiA+10);
		indAct++;
		if(buff[indAct]<ascii9) lar+=16*(buff[indAct]-ascii0);
		else lar+=16*(buff[indAct]-asciiA+10);
		indAct++;
		if(buff[indAct]<ascii9) lar+=(buff[indAct]-ascii0);
		else lar+=(buff[indAct]-asciiA+10);
		return lar;
	}

	/**
	 * Mira si es caracter hexadecimal MAYÚSCULA
	 * @param ind indice en {@link #buff} del dato a mirar
	 * @return true si es caracter hexadecimal mayúscula
	 */
	private boolean isHexa(int ind) {
		return ((buff[ind]>=ascii0) && (buff[ind]<=ascii9)) || ((buff[ind]>=asciiA && buff[ind]<=asciiF)) ;
	}

	/**
	 * Mira si es caracter válido en la cabecera de paquete binario (entre 48 y 126)
	 * @param ind indice en {@link #buff} del dato a mirar
	 * @return true si es caracter es válido
	 */
	private boolean isCabBin(int ind) {
		return ((buff[ind]>=48) && (buff[ind]<=126)) ;
	}

	/**
	 * Mira si es caracter válido en la cabecera de paquete texto (entre 33 y 47)
	 * @param ind indice en {@link #buff} del dato a mirar
	 * @return true si es caracter es válido
	 */
	private boolean isCabTex(int ind) {
		return ((buff[ind]>=33) && (buff[ind]<=47)) ;
	}

	/**
	 * Actuliza toda los buffers con la información contenida en la nueva cadena recibida.
	 * Fija el sistema local al primer punto de buffer espacial.
	 * @param cadena cadena recibida (del GPS)
	 */
	void actualizaNuevaCadenaTexto() {
		int larMen=indFin-indIni+1;
		System.out.print("Texto ("+larMen+"):>");
		for(int i=indIni; i<=indFin; i++)
			System.out.print((char)buff[i]);
		System.out.println("<");
	}
	void actualizaNuevaCadenaBinaria() {
		int larMen=indFin-indIni+1;
		System.out.print("Binaria ("+larMen+"):"+new String(buff,indIni,5)+" >");
		for(int i=indIni+5; i<=indFin; i++)
			if (buff[i]<32 || buff[i]>126)
				//no imprimible
				System.out.print('.');
			else
				System.out.print((char)buff[i]);
		System.out.println("< >"+UtilMensajesIMU.hexaString(buff, indIni+5, larMen-5)+"<");		
	}

	public void comandoGPS(String comando) {
		try {
		flujoSalida.write(comando.getBytes());
		} catch (Exception e) {
			// TODO: handle exception
		}
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {

		GPSConnection gpsC;
		
		try {
			gpsC=new GPSConnection("/dev/ttyUSB0",9600);

			try { Thread.sleep(2000); } catch (Exception e) {}

			gpsC.comandoGPS("em,,{jps/RT,nmea/GGA,jps/PO,jps/BL,nmea/GST,jps/ET}:2\n");
		} catch (Exception e) {
		}
		
		
	}
}