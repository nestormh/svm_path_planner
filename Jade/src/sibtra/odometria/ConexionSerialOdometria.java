package sibtra.odometria;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.TooManyListenersException;

import sibtra.controlcarro.ControlCarroConnectionException;
import sibtra.controlcarro.ControlCarroSerialParameters;
import sibtra.gps.GpsEventListener;
import sibtra.imu.IMUEvent;
import sibtra.imu.IMUEventListener;
import sibtra.log.LoggerFactory;
import sibtra.util.UtilCalculos;

import gnu.io.CommPortIdentifier;
import gnu.io.NoSuchPortException;
import gnu.io.PortInUseException;
import gnu.io.SerialPort;
import gnu.io.SerialPortEvent;
import gnu.io.SerialPortEventListener;
import gnu.io.UnsupportedCommOperationException;
/**
 * Clase que se encargará de gestionar la conexión serial con la electrónica
 * del sistema odométrico
 * **/
public class ConexionSerialOdometria implements SerialPortEventListener {

	private InputStream inputStream=null;
	private int TotalBytes = 0;
	private long encoder1;
	private long encoder2;
	static int MAX_VALOR_INT = 65536;
	/** Mutex donde se bloquean los hilos que quieren espera por un nuevo dato */
	private Object mutexDatos;
	private int NumPaquetes = 0;
	private ControlCarroSerialParameters parameters = null;
	private CommPortIdentifier portId;
	private SerialPort sPort;
	private OutputStream outputStream;
	private boolean open;
	private DatosOdometria datos = new DatosOdometria(0,0,0);
	int bufferAnt[] = new int[5];

	/**
	 * Crea la conexión serial al carro en el puerto indicado.
	 * Una vez abierto queda a la espera de los eventos de recepción de caracteres.
	 * @param portName nombre del puerto serial 
	 */
	public ConexionSerialOdometria(String portName) {
		if(!portName.equals("/dev/null")) { 
			parameters = new ControlCarroSerialParameters(portName, 9600, 0, 0, 8,
					1, 0);
			try {
				openConnection();
			} catch (ControlCarroConnectionException e2) {

				System.out.println("Error al abrir el puerto " + portName);
				System.out.flush();
			}
			if (isOpen())
				System.out.println("Puerto Abierto " + portName);
		} else {
			System.out.println(this.getClass().getName()+": Trabajamos sin conexión ");
		}
//		logAngVel=LoggerFactory.nuevoLoggerArrayDoubles(this, "carroAngVel",12);
//		logAngVel.setDescripcion("Carro [Ang en Rad,Vel en m/s]");
//		
//		logMenRecibidos= LoggerFactory.nuevoLoggerArrayInts(this, "mensajesRecibidos",(int)(1/T)+1);
//		logMenRecibidos.setDescripcion("volante,avance,(int)velocidadCS,alarma,incCuentas,incTiempo");
//		logMenEnviados= LoggerFactory.nuevoLoggerArrayInts(this, "mensajesEnviados",(int)(1/T)+1);
//		logMenEnviados.setDescripcion("ConsignaVolante,ComandoVelocidad,ConsignaFreno,ConsignaNumPasosFreno");
//		logControl=LoggerFactory.nuevoLoggerArrayDoubles(this, "controlPID",(int)(1/T)+1);
//		logControl.setDescripcion("consignaVel,velocidadCS,derivativo,integral,comandotemp,comando,apertura,avanceAplicado");
//		logParamPID=LoggerFactory.nuevoLoggerArrayDoubles(this, "ParamPID",(int)(1/T)+1);
//		logParamPID.setDescripcion("[kPAvance,kIAvance,kDAvance,maxInc, FactorFreno, maxDec]");
	}

	/**
	 * Attempts to open a serial connection and streams using the parameters in
	 * the SerialParameters object. If it is unsuccesfull at any step it returns
	 * the port to a closed state, throws a
	 * <code>SerialConnectionException</code>, and returns.
	 * 
	 * Gives a timeout of 30 seconds on the portOpen to allow other applications
	 * to reliquish the port if have it open and no longer need it.
	 */
	private void openConnection() throws ControlCarroConnectionException {

		// Obtain a CommPortIdentifier object for the port you want to open.
		try {
			portId = CommPortIdentifier.getPortIdentifier(parameters
					.getPortName());
		} catch (NoSuchPortException e) {
			throw new ControlCarroConnectionException(e.getMessage());
		}

		// Open the port represented by the CommPortIdentifier object. Give
		// the open call a relatively long timeout of 30 seconds to allow
		// a different application to reliquish the port if the user
		// wants to.
		try {
			sPort = (SerialPort) portId.open("SerialDemo", 30000);
		} catch (PortInUseException e) {
			throw new ControlCarroConnectionException(e.getMessage());
		}

		// Set the parameters of the connection. If they won't set, close the
		// port before throwing an exception.
		try {
			setConnectionParameters();
		} catch (ControlCarroConnectionException e) {
			sPort.close();
			throw e;
		}

		// Open the input and output streams for the connection. If they won't
		// open, close the port before throwing an exception.
		try {
			outputStream = sPort.getOutputStream();
			inputStream = sPort.getInputStream();
		} catch (IOException e) {
			sPort.close();
			throw new ControlCarroConnectionException(
					"Error opening i/o streams");
		}

		// Add this object as an event listener for the serial port.
		try {
			sPort.addEventListener(this);
		} catch (TooManyListenersException e) {
			sPort.close();
			throw new ControlCarroConnectionException(
					"too many listeners added");
		}

		// Set notifyOnDataAvailable to true to allow event driven input.
		sPort.notifyOnDataAvailable(true);

		// Set notifyOnBreakInterrup to allow event driven break handling.
		//No se están atendiendo los break
//		sPort.notifyOnBreakInterrupt(true);

		// Set receive timeout to allow breaking out of polling loop during
		// input handling.
		// try {
		// sPort.enableReceiveTimeout(30);
		// } catch (UnsupportedCommOperationException e) {
		// }

		open = true;

		sPort.disableReceiveTimeout();
	}

	/**
	 * Sets the connection parameters to the setting in the parameters object.
	 * If set fails return the parameters object to origional settings and throw
	 * exception.
	 */
	private void setConnectionParameters()
			throws ControlCarroConnectionException {

		// Save state of parameters before trying a set.
		int oldBaudRate = sPort.getBaudRate();
		int oldDatabits = sPort.getDataBits();
		int oldStopbits = sPort.getStopBits();
		int oldParity = sPort.getParity();
		int oldFlowControl = sPort.getFlowControlMode();

		// Set connection parameters, if set fails return parameters object
		// to original state.
		try {
			sPort.setSerialPortParams(parameters.getBaudRate(), parameters
					.getDatabits(), parameters.getStopbits(), parameters
					.getParity());
		} catch (UnsupportedCommOperationException e) {
			parameters.setBaudRate(oldBaudRate);
			parameters.setDatabits(oldDatabits);
			parameters.setStopbits(oldStopbits);
			parameters.setParity(oldParity);
			throw new ControlCarroConnectionException("Unsupported parameter");
		}

		// Set flow control.
		try {
			sPort.setFlowControlMode(parameters.getFlowControlIn()
					| parameters.getFlowControlOut());
		} catch (UnsupportedCommOperationException e) {
			throw new ControlCarroConnectionException(
					"Unsupported flow control");
		}
	}

	/** Cierra el puerto y libera los elementos asociados */
	public void closeConnection() {
		// If port is alread closed just return.
		if (!open) {
			return;
		}

		// Check to make sure sPort has reference to avoid a NPE.
		if (sPort != null) {
			try {
				// close the i/o streams.
				outputStream.close();
				inputStream.close();
			} catch (IOException e) {
				System.err.println(e);
			}
			// Close the port.
			sPort.close();
		}
		open = false;
	}

	/** Send a one second break signal. */
	private void sendBreak() {
		sPort.sendBreak(1000);
	}

	/** @return true if port is open, false if port is closed.*/
	public boolean isOpen() {
		return open;
	}
	@Override
	public void serialEvent(SerialPortEvent e) {
		int buffer[] = new int[5]; //porque el 0 no se usa :-(		
		int newData = 0;
//		System.out.println("Llega paquete a la serialC");
		//Sólo atendemos la disponibilidad de datos
		if (e.getEventType() !=  SerialPortEvent.DATA_AVAILABLE)
			return;
		while (newData != -1) {
			try {
				newData = inputStream.read();
				TotalBytes++;
				if (newData == -1) {
					break;
				}
				if (newData!=250)
					continue;
				//newdata vale 250
				newData = inputStream.read();
				if (newData != 251)
					continue; //TODO faltaría considerar el caso de varios 255 seguidos
				//newdata vale 251
				//Ya tenemos la cabecera de un mensaje válido
				//leemos los datos del mensaje en buffer				
				
				
				
				buffer[1] = inputStream.read();
				buffer[2] = inputStream.read();
				buffer[3] = inputStream.read();
				buffer[4] = inputStream.read();				
				newData = inputStream.read();
				
				if (newData != 255)  //no está la marca de fin de paquete
					continue; //no lo consideramos
				//Ya tenemos paquete valido en buffer
				trataMensaje(buffer);
				bufferAnt = buffer.clone();				
				
			} catch (IOException ex) {
				System.err.println(ex);
				return;
			}
		}		
	}
	
	private void trataMensaje(int buffer[]) {	
		int encoder1Ant = UtilCalculos.byte2entero(bufferAnt[1], bufferAnt[2]);
		int encoder2Ant = UtilCalculos.byte2entero(bufferAnt[3], bufferAnt[4]);

		int encoder1leido = UtilCalculos.byte2entero(buffer[1], buffer[2]);
		int encoder2leido = UtilCalculos.byte2entero(buffer[3], buffer[4]);
		
		int incEncoder1 = 0;
		int incEncoder2 = 0;
		
		//Sobrepasamiento por arriba encoder 1
		if (buffer[1] - bufferAnt[1] < -128){ // la mitad de 2 Bytes
			incEncoder1 = encoder1leido + (MAX_VALOR_INT - encoder1Ant);
			
			//Sobrepasamiento por abajo
		}else if (buffer[1] - bufferAnt[1] > 128){
			incEncoder1 = - (MAX_VALOR_INT - encoder1leido) - encoder1Ant; 
		} else {
			incEncoder1 = encoder1leido - encoder1Ant;
		}
		
		encoder1 = encoder1 + incEncoder1;
		
		//Sobrepasamiento por arriba encoder 2
		if (buffer[3] - bufferAnt[3] < -128){ // la mitad de 2 Bytes	
			incEncoder2 = encoder2leido + (MAX_VALOR_INT - encoder2Ant);
			//Sobrepasamiento por abajo
		}else if (buffer[3] - bufferAnt[3] > 128){
			incEncoder2 = - (MAX_VALOR_INT - encoder2leido) - encoder2Ant; 
		} else {
			incEncoder2 = encoder2leido - encoder2Ant;
		}
		
		encoder2 = encoder2 + incEncoder2;
		
		NumPaquetes++;
		//despertamos hilos pendientes de nuevos datos
		if ((encoder1Ant != encoder1leido) || (encoder2Ant != encoder2leido)){
			datos.calculaDatos(incEncoder1, incEncoder2, 0.2);
		}
		
//		mutexDatos.notifyAll();
		avisaListeners();
		
		//	logMenRecibidos.add(volante,avance,(int)velocidadCS,alarma,incCuentas,(int)incTiempo);
	}
	/**
	 * Bloquea el thread si no se ha recibido mensaje desde el que se pasa
	 * @param numeroDePaquete numero del último paquete recibido
	 * @return el número del nuevo paquete recibido
	 */
	public int esperaNuevosDatos(int numeroDePaquete) {
		synchronized (mutexDatos) {
			while(numeroDePaquete==NumPaquetes)
				try {
					mutexDatos.wait(); //nos quedamos bloqueados en mutexDatos
				}catch (InterruptedException e) {
					//No hacemos nada si somos interrumpidos
				}
		}
		return NumPaquetes;  
	}
	
	/** mantiene la lista de listeners */
	private ArrayList<OdometriaEventListener> listeners = new ArrayList<OdometriaEventListener>();

	/**
	 * Para añadir objeto a la lista de {@link GpsEventListener}
	 * @param iel objeto a añadir
	 */
	public void addOdometriaEventListener( OdometriaEventListener iel ) {
		listeners.add( iel );
	}
	
	/** avisa a todos los listeners con un evento */
	private void avisaListeners() {
	    for ( int j = 0; j < listeners.size(); j++ ) {
	        OdometriaEventListener iel = listeners.get(j);
	        if ( iel != null ) {
	          OdometriaEvent me = new OdometriaEvent(this,datos);
	          iel.handleOdometriaEvent(me);
	        }
	    }
	}

	
	public long getEncoder1() {
		return encoder1;
	}

	public void setEncoder1(long encoder1) {
		this.encoder1 = encoder1;
	}

	public long getEncoder2() {
		return encoder2;
	}

	public void setEncoder2(long encoder2) {
		this.encoder2 = encoder2;
	}

	public int getNumPaquetes() {
		return NumPaquetes;
	}

	public void setNumPaquetes(int numPaquetes) {
		NumPaquetes = numPaquetes;
	}

}
