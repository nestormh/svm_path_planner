/* @(#)SerialConnection.java	1.6 98/07/17 SMI
 *
 * Clase de Control del veh�culo Guistub
 */
package sibtra.controlcarro;

import gnu.io.CommPortIdentifier;
import gnu.io.NoSuchPortException;
import gnu.io.PortInUseException;
import gnu.io.SerialPort;
import gnu.io.SerialPortEvent;
import gnu.io.SerialPortEventListener;
import gnu.io.UnsupportedCommOperationException;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.TooManyListenersException;

import sibtra.log.LoggerArrayDoubles;
import sibtra.log.LoggerArrayInts;
import sibtra.log.LoggerFactory;
import sibtra.util.UtilCalculos;



/**
 * Clase para gestionar la comunicación con el PIC que controla el carro.
 * El lazo del control del volante se cierra a bajo nivel (en el PIC) por lo que 
 * basta mandar la consigna de ángulo.
 * El control de velocidad se realiza en esta clase. Para ello hay que combinar la 
 * fuerza aplicada al motor (avance) y el freno.
 * 
 * @author jonay
 */
public class ControlCarro implements SerialPortEventListener {

	/*
	 * En el freno existen 3 variables a modificar Sentido_freno = indica si se
	 * debe frenar o desfrenar Valor_Freno = indica la fuerza, la amplitud con
	 * la que se abre la valvula que abre el circuito de frenado. Tiempo_freno =
	 * Tiempo en el que esta actuando el freno, funciona a la inversa donde el
	 * tiempo que se pone tiene que saturar a MaxpasosFreno,FFh para parar el
	 * frenado. MaxpasosFreno = 10 decimal, EN REALIDAD SE FIJA EN EL MICRO LA
	 * PARTE ALTA DE LA PALABRA A MAXPASOSFRENO -1 POR LO QUE SE HACE SOLO LA
	 * CUENTA BAJA. Hay que ajustar los parametros de tiempo para conseguir que
	 * funcione correctamente /
	 */


	/** Maximo comando en el avance */
	public final static int MINAVANCE = 100;
	/** Minimo comando en el avance */
	public final static int MAXAVANCE = 240;
	
	/** Numero de cuentas necesarias para alcanzar un metro	 */
	public final static double PULSOS_METRO = 74;

	// public static final double RADIANES_POR_CUENTA =
	// 2*Math.PI/MAX_CUENTA_VOLANTE;
	/** Punto central del volante del vehiculo */
//	public final static int CARRO_CENTRO = 5280;
	// 2927, 3188, 3897
	public final static int CARRO_CENTRO = 3800;
	/** Radianes que suponen cada cuenta del sensor del volante */
//	public static final double RADIANES_POR_CUENTA = 0.25904573048913979374 / (CARRO_CENTRO - 3300);
	private static int CUENTAS_PARA_15_GRADOS_DESDE_EL_CENTRO=2156;
//	public static final double RADIANES_POR_CUENTA 
//		= Math.toRadians(15) / (CARRO_CENTRO - CUENTAS_PARA_15_GRADOS_DESDE_EL_CENTRO);
	public static final double RADIANES_POR_CUENTA = Math.toRadians(45) / CARRO_CENTRO ;

	/** Periodo de envío de mensajes por parte del PIC ¿? */
	static double T = 0.096; // Version anterior 0.087

	/** Número de puntos que se usan para calcular la velocidad */
	static private int freqVel = 8;
	/** Array cuentas para el cálculo de la velocidad */
	int Cuentas[] = new int[freqVel];
	/** Array de tiempos para el cálculo de la velocidad */
	long tiempos[] = new long[freqVel];
	/** Puntero sobre los array de cálculo de la velocidad {@link #Cuentas} y {@link #tiempos} */
	int indiceCuentas = 0;

	/** Error del controlador PID de la velocidad en {@link #controlVel()}*/
	double errorAnt = 0;

	/** Derivada del controlador PID de la velocidad  en {@link #controlVel()}*/
	double derivativoAnt = 0;
	
	/** Comando calculado por el controlador PID de la velocidad  en {@link #controlVel()}*/
	double comando = CARRO_CENTRO;

	/** comando anterior calculado por el PID de control de la velocidad  en {@link #controlVel()}*/
	private double comandoAnt = MINAVANCE;

	/** Integral del controlador PID de la velocidad  en {@link #controlVel()}*/
	double integral = 0;

	// Campos relativos a la conexión serial =========================================================
	private ControlCarroSerialParameters parameters=null;

	private OutputStream outputStream=null;

	private InputStream inputStream=null;

	private CommPortIdentifier portId=null;

	private SerialPort sPort=null;
	
	/** Para saber si el puerto serial está abierto */
	private boolean open=false;

	// Fin de campos relativos a conexión serial =======================================================


	/** Ultima velocidad de avance calculada en cuentas por segundo */
	private double velocidadCS = 0;
	/** Ultima velocidad de avance calculada en metros por segundo */
	private double velocidadMS = 0;

	/** Total de bytes recibidos desde el PIC por la serial */
	private int TotalBytes = 0;

	/** Posición del volante recibida desde el PIC. 
	 * El 0 (valor más pequeño) está con el volante maś a la izquierda.
	 * A partir de esa posición el valor irá creciendo.
	 * Cuando esté en medio tendrá el valor {@link #CARRO_CENTRO}.
	 * */
	private int volante = 32768;

	/** Cuentas del encoder de avance de la tracción, tiene corregido el desbordamiento */
	private int avance = 0;

	/** Valora anterior recibido del encoder de tracción. Necesario para corregir desbordamiento contador */
	private int avanceant = 0;

	/** Contiene byte, recibido del PIC, con los bits correspondientes a las distintas alarmas */
	private int alarma = 0;

        
	/** Valor de la última consigna para el volante enviada.
	 * Se envía al PIC descomponiéndola en la parte alta y la baja 
	 */
	private int ConsignaVolante = 32767;
	/* Comandos enviados hacia el coche para el microcontrolador */
	/** Uno de los 8 bytes que se envían en mensaje al PIC */
	private int ConsignaFreno = 0;
	/** Uno de los 8 bytes que se envían en mensaje al PIC. 
	 * Indica que acción se va ha hacer con el freno
	 * <code>
	 * 0 - Parar el freno, lo que cerraria todas las valvulas y terminaria  
	 * todos los tiempos, dejando el freno como esta.
	 * 1 - Abrir valvula de desfrenar
	 * 2 - Abrir valvula de frenar
	 * 3 - Desfrenar
	 * 4 - No hacer nada, dejar el freno en el comportamiento antes fijado,  
	 * cerrara las valvulas cuando toque.
	 *  </code>
	 *  */
	private int ConsignaSentidoFreno = PARAR_FRENO;
	
	static public int PARAR_FRENO=0;
	static public int ABRIR_DESFRENAR=1;
	static public int ABRIR_FRENAR=2;
	static public int DESFRENAR=3;
	static public int NOCAMBIAR_FRENO=4;
	
	/** Uno de los 8 bytes que se envían en mensaje al PIC */
	private int ComandoVelocidad = 0;
	/** Uno de los 8 bytes que se envían en mensaje al PIC */
	private int ConsignaSentidoVelocidad = 0;
	/** Uno de los 8 bytes que se envían en mensaje al PIC */
	private int ConsignaNumPasosFreno = 0;


	/** Numero de pasos de freno que se han dado */
	private int NumPasosFreno = 0;

	/** Logger para registrar mensaje recibidos */
	private LoggerArrayInts logMenRecibidos;
	/** Logger para registrar los mensajece enviados */
	private LoggerArrayInts logMenEnviados;
//	private int RVolante;

	
        
//	private int refresco = 300;

//	/** Ganancia proporcional del PID de freno */
//	private double kPFreno = 0.5;
//	/** Ganancia derivativa del PID de freno */
//	private double kPDesfreno = 1.5;
	
	/** Ganancia proporcional del PID de avance */
	private double kPAvance = 1.0;
	/** Ganancia defivativa del PID de avance */
	private double kDAvance = 0.3;
	/** Ganancia integral del PID de avance */
	private double kIAvance = 0.01;

	/** Para indicar si se está aplicando control de velocidad */
	private boolean controlando = false;

	/** Consigna de velocidad a aplicar en cuentas por segundo */
	private double consignaVel = 0;

	/** Maximo incremento permitido en el comando para evitar aceleraciones bruscas */
	private int maxInc = 2;
	
	/** Máximo decremento permitido del comando aplicado cuando este es negativo.
	 * Hace que el decremento sea constante al principio y se aplique el freno seguido.
	 */
	private int maxDec=4;
        
	/** Zona Muerta donde el motor empieza a actuar realmente 	 */
	static final int ZonaMuerta = 60;
	private static final double limiteIntegral = 40;
        
	/** Numero de paquetes recibidos validos */
	private int NumPaquetes = 0;

	/** Milisegundos al crearse el objeto. Se usa para tener tiempos relativos */
//	private long TiempoInicial = System.currentTimeMillis();


	/** Para llevar la cuenta de la aplicación del freno */
//	private int contadorFreno = 0;

	
	/** Registrador del angulo y velocidad medidos al recivir cada paquete */
	private LoggerArrayDoubles logAngVel;

	/**	Registrador de todos los datos del PID de avance*/
	private LoggerArrayDoubles logControl;

	/**	Registrador de todos los parámetros del PID de avance*/
	private LoggerArrayDoubles logParamPID;

	double FactorFreno=30;
	
	/** Mutex donde se bloquean los hilos que quieren espera por un nuevo dato */
	private Object mutexDatos;

	/**
	 * Crea la conexión serial al carro en el puerto indicado.
	 * Una vez abierto queda a la espera de los eventos de recepción de caracteres.
	 * @param portName nombre del puerto serial 
	 */
	public ControlCarro(String portName) {
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
		logAngVel=LoggerFactory.nuevoLoggerArrayDoubles(this, "carroAngVel",12);
		logAngVel.setDescripcion("Carro [Ang en Rad,Vel en m/s]");
		
		logMenRecibidos= LoggerFactory.nuevoLoggerArrayInts(this, "mensajesRecibidos",(int)(1/T)+1);
		logMenRecibidos.setDescripcion("volante,avance,(int)velocidadCS,alarma,incCuentas,incTiempo");
		logMenEnviados= LoggerFactory.nuevoLoggerArrayInts(this, "mensajesEnviados",(int)(1/T)+1);
		logMenEnviados.setDescripcion("ConsignaVolante,ComandoVelocidad,ConsignaFreno,ConsignaNumPasosFreno");
		logControl=LoggerFactory.nuevoLoggerArrayDoubles(this, "controlPID",(int)(1/T)+1);
		logControl.setDescripcion("consignaVel,velocidadCS,derivativo,integral,comandotemp,comando,apertura,avanceAplicado");
		logParamPID=LoggerFactory.nuevoLoggerArrayDoubles(this, "ParamPID",(int)(1/T)+1);
		logParamPID.setDescripcion("[kPAvance,kIAvance,kDAvance,maxInc, FactorFreno]");
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

	/**
	 * Maneja la llegada de datos por la serial
	 */
	public void serialEvent(SerialPortEvent e) {
		int buffer[] = new int[6]; //porque el 0 no se usa :-(
		int newData = 0;

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
				buffer[5] = inputStream.read();
				newData = inputStream.read();
				if (newData != 255)  //no está la marca de fin de paquete
					continue; //no lo consideramos
				//Ya tenemos paquete valido en buffer
				trataMensaje(buffer);
			} catch (IOException ex) {
				System.err.println(ex);
				return;
			}
		}

	}

	/** Función que hace el tratamiento de un mensaje válido */
	private void trataMensaje(int buffer[]) {
		
		volante = UtilCalculos.byte2entero(buffer[1], buffer[2]);

		//si no hay consigna de volante, dejamos el volante como está
		if (ConsignaVolante == -1) {
			ConsignaVolante = volante;
		}

		//controlamos desbordamiento contador de avance en el PIC
		if (avanceant <= buffer[3])
			//no se ha producido
			avance = avance + (buffer[3] - avanceant);
		else
			//se ha producido
			avance = avance + (buffer[3] + (255 - avanceant) + 1);
		avanceant = buffer[3];  //preparamos para siguiente iteración

		
		// Calculamos la velocidad usando ventana de freqVel mensajes.
		Cuentas[indiceCuentas] = avance;
		tiempos[indiceCuentas] = System.currentTimeMillis();

		int  incCuentas=Cuentas[indiceCuentas]
		                        - Cuentas[(indiceCuentas + 1) % freqVel];
		long incTiempo=tiempos[indiceCuentas]
		                       - tiempos[(indiceCuentas + 1) % freqVel];
		velocidadCS=1000.0 *(double) incCuentas / (double)incTiempo;
		
		indiceCuentas = (indiceCuentas + 1) % freqVel; //preparamos siguiente iteración

		velocidadMS = velocidadCS / PULSOS_METRO;

		//apuntamos angulo volante en radianes y velocidad en m/sg
		logAngVel.add(getAnguloVolante(),velocidadMS);

		// System.out.println("T: " +
		// (System.currentTimeMillis() - lastPaquete) +
		// " Tmedio: " + (TiempoTotal/Recibidos));

		controlVel();

		alarma = buffer[4];
		NumPasosFreno = buffer[5];

//		if (ConsignaSentidoFreno != 0)
//			System.out.println("NumPasosFreno = "+ NumPasosFreno);
//		if (NumPasosFreno == 255) {
//			if (ConsignaSentidoFreno == 2) {
//				System.out.println("Frenando");
//				if (contadorFreno > 0) {
//					System.out.println("ContadorFreno1 = "+ contadorFreno);
//					if (getFreno() == 1) {
//						contadorFreno = 0;
//					} else {
//						NumPasosFreno = 0;
//						contadorFreno--;
//					}
//					System.out.println("ContadorFreno2 = "+ contadorFreno);
//				}
//			} else if (ConsignaSentidoFreno == 1) {
//				System.out.println("DesFrenando");
//				if (contadorFreno > 0) {
//					System.out.println("ContadorFreno1 = "+ contadorFreno);
//					if (getDesfreno() == 1) {
//						contadorFreno = 0;
//					} else {
//						NumPasosFreno = 0;
//						contadorFreno--;
//					}
//					System.out.println("ContadorFreno2 = "+ contadorFreno);
//				}
//			}
//		}


		NumPaquetes++;
		//despertamos hilos pendientes de nuevos datos
		mutexDatos.notifyAll();
		logMenRecibidos.add(volante,avance,(int)velocidadCS,alarma,incCuentas,(int)incTiempo);
	}
	
	/** @return Objeto de escritura del puerto serie */
	public OutputStream getOutputStream() {
		return outputStream;

	}

	/**
	 * Devuelve el numero de Bytes que hemos recibido del coche
	 * 
	 */
	public int getBytes() {
		return TotalBytes;
	}

	/**
	 * Devuelve el numero de Cuentas del encoder de la tracci?n, lo que se ha
	 * desplazado el coche
	 * 
	 */

	public int getAvance() {
		return avance;
	}

	/**
	 * Devuelve la posicion actual del volante en cuentas
	 * 
	 */
	public int getVolante() {
		return volante;
	}

	/**
	 * @return Devuelve el angulo del volante en radianes respecto al centro (+izquierda, - derecha).
	 */
	public double getAnguloVolante() {
		return (CARRO_CENTRO-volante) * RADIANES_POR_CUENTA;
	}

	/**
	 * @return Devuelve el angulo del volante en grados respecto al centro (+izquierda, - derecha).
	 */
	public double getAnguloVolanteGrados() {
		return Math.toDegrees(getAnguloVolante());
	}

	/**
	 * Devuelve si esta pulsada la alarma de desfrenado
	 * 
	 */
	public int getDesfreno() {
		if ((alarma & 16) == 16)
			return 1;
		return 0;
	}

	/**
	 * Devuelve si esta pulsada la alarma de frenado
	 * 
	 */
	public int getFreno() {
		if ((alarma & 8) == 8)
			return 1;
		return 0;
	}

	/**
	 * Devuelve si esta pulsada la alarma del fin de carrera izquierdo
	 * 
	 */
	public int getIzq() {
		if ((alarma & 4) == 4)
			return 1;
		return 0;
	}

	/**
	 * Devuelve si esta pulsada la alarma del fin de carrera derecho
	 * 
	 */
	public int getDer() {
		if ((alarma & 2) == 2)
			return 1;
		return 0;
	}

	/**
	 * Devuelve si esta pulsada la alarma global del coche, en este caso el
	 * sistema bloquea todas las posibles opciones
	 * 
	 */
	public int getAlarma() {
		if ((alarma & 1) == 1)
			return 1;
		return 0;
	}

	/**
	 * Obtiene la consigna del volante
	 * 
	 * @return Devuelve la consigna del volante en cuentas
	 */
	public int getConsignaVolante() {
		return ConsignaVolante;
	}

	/**
	 * Obtiene la consigna del volante en ángulo
	 * 
	 * @return Devuelve la consigna del volante en radianes
	 */
	public double getConsignaAnguloVolante() {
		return ( CARRO_CENTRO-ConsignaVolante) * RADIANES_POR_CUENTA;
	}

	/**
	 * @return Devuelve la consigna del volante en grados
	 */
	public double getConsignaAnguloVolanteGrados() {
		return Math.toDegrees(getConsignaAnguloVolante());
	}

	
	
	/**
	 * @return Devuelve el comando de la velocidad
	 */
	public int getComandoVelocidad() {
		return ComandoVelocidad;
	}

	
	
	/**
	 * Fija consigna del volante
	 * 
	 * @param comandoVolante angulo deseado en radianes desde el centro (+izquierda, -derecha)
	 */
	public void setAnguloVolante(double comandoVolante) {
		setVolante( CARRO_CENTRO - (int)Math.floor(comandoVolante / RADIANES_POR_CUENTA) );
	}

	/**
	 * Fija la posicion del volante a las cuentas indicadas
	 * 
	 * @param Angulo Numero de cuentas a las que fijar el volante
	 */
	private void setVolante(int Angulo) {
		ConsignaVolante=UtilCalculos.limita(Angulo, 0, 65535);
		ConsignaSentidoFreno=NOCAMBIAR_FRENO;  //No modifica el freno
		ConsignaNumPasosFreno = NumPasosFreno; //TODO sobra
		Envia();
	}

//	/**
//	 * Fija la posicion del volante a n cuentas a partir de la posicion actual
//	 * 
//	 * @param deltaAngulo
//	 *            Numero de cuentas a desplazar a partir de la posicion actual
//	 */
//	public void setRVolante(int deltaAngulo) {
//		ConsignaVolante = UtilCalculos.limita(ConsignaVolante + deltaAngulo, 0, 65535);
//		ConsignaSentidoFreno=NOCAMBIAR_FRENO;
//		ConsignaNumPasosFreno = NumPasosFreno; //TODO sobra
//		Envia();
//	}


	/**
	 * Fija la velocidad hacia delante con una Fuerza dada entre 0-255
	 */
	public void Avanza(int Fuerza) {
		Fuerza=UtilCalculos.limita(Fuerza,0,255);

		/*if (getDesfreno() != 1) {
			DesFrena(255);
			return;
		}*/

		ComandoVelocidad = Fuerza;
		ConsignaSentidoVelocidad = 2;  //para alante
//		ConsignaFreno = 4;  //dejar el freno como está
		ConsignaSentidoFreno = NOCAMBIAR_FRENO;
		ConsignaNumPasosFreno = 0; //TODO sobra
		Envia();
	}

	/**
	 * Fija la velocidad hacia delante con una Fuerza dada entre 0-255
	 */
	public void Retrocede(int Fuerza) {

		Fuerza=UtilCalculos.limita(Fuerza,0,255);

		if (getDesfreno() != 1) {
			DesFrena(255);
			return;
		}

		ConsignaSentidoFreno=NOCAMBIAR_FRENO;
		ConsignaNumPasosFreno = 0; //TODO sobra
		ComandoVelocidad = Fuerza;
		ConsignaSentidoVelocidad = 1;  //para atrás

		Envia();
	}

	/**
	 * @param tiempo 255 maximo, 0 minimo tiempo
	 */
	public void DesFrena(int tiempo) {

		tiempo=UtilCalculos.limita(tiempo, 0, 255);		
		tiempo = 255 - tiempo;

		ConsignaFreno = 255;
		ConsignaSentidoFreno = DESFRENAR;  //abrir las 2 válvulas ??
		ConsignaNumPasosFreno = tiempo;
		ComandoVelocidad = 0;
		ConsignaSentidoVelocidad = 0;

		Envia();

		ConsignaFreno = 0;
		ConsignaNumPasosFreno = 0;
		ConsignaSentidoFreno = PARAR_FRENO; //TODO mejor NOCAMBIAR_FRENO ¿?
	}

	/**
	 * Aumenta el frenado del sistema
	 * @param valor apertura de la valvula
	 * @param tiempo tiempo de esta apertura
	 */
	public void masFrena(int valor, int tiempo) {
		valor=UtilCalculos.limita(valor,0,255);
		tiempo=UtilCalculos.limita(tiempo, 0, 255);
		
		tiempo = 255 - tiempo;

		ConsignaSentidoFreno = ABRIR_FRENAR;  //Abrir valvula de frenar
		ConsignaNumPasosFreno = tiempo;
		ConsignaFreno = valor;
		ComandoVelocidad = 0;
		ConsignaSentidoVelocidad = 0;

		Envia();

		ConsignaFreno = 0;
		ConsignaNumPasosFreno = 0;
		ConsignaSentidoFreno = PARAR_FRENO; //TODO mejor NOCAMBIAR_FRENO ¿?
	}
	/**
	 * Disminuye la presion del frenado
	 * @param valor Apertura de la electrovalvula
	 * @param tiempo tiempo de apertura
	 */
	public void menosFrena(int valor, int tiempo) {
		valor=UtilCalculos.limita(valor,0,255);
		tiempo=UtilCalculos.limita(tiempo, 0, 255);

		tiempo = 255 - tiempo;

		
		ConsignaSentidoFreno = ABRIR_DESFRENAR; //abrir la valvula de desfrenar
		ConsignaFreno = valor;
		ConsignaNumPasosFreno = tiempo;
		ComandoVelocidad = 0;
		ConsignaSentidoVelocidad = 0;

		Envia();

		ConsignaFreno = 0;
		ConsignaNumPasosFreno = 0;
		ConsignaSentidoFreno = PARAR_FRENO; //TODO mejor NOCAMBIAR_FRENO ¿?
	}


	private void Envia() {
		
		logMenEnviados.add(ConsignaVolante,ComandoVelocidad,ConsignaFreno,ConsignaNumPasosFreno);
		int a[] = new int[10];

		a[0] = 250;
		a[1] = 251;
		a[2] = ConsignaVolante & 255;
		a[3] = (ConsignaVolante & 65280) >> 8;
		a[4] = ConsignaFreno;
		a[5] = ConsignaSentidoFreno;
		a[6] = ComandoVelocidad;
		a[7] = ConsignaSentidoVelocidad;
		a[8] = ConsignaNumPasosFreno;
		a[9] = 255;

		if(isOpen()) {
		for (int i = 0; i < 10; i++)
			try {
				outputStream.write(a[i]);

			} catch (Exception e) {
				System.out.println("Error al enviar, " + e.getMessage());
				System.out.println(e.getStackTrace());
				try {
					System.out.println("Se va a proceder a vaciar el buffer");
					outputStream.flush();
				} catch (Exception e2) {
					System.out.println("Error al vaciar el buffer, "
							+ e2.getMessage());
				}
			}
		} else {
			System.err.println(getClass().getName()+": Conexion no abierta al tratar de enviar ");
		}
	}


	/**
	 * Función que implementa PID para el control de la velociad.
	 * Debe convinar la fuerza de avance con el freno.
	 */
	public void controlVel() {

		
		if (!controlando) 
			return;

		if (consignaVel == 0 && velocidadCS<20)
			stopControlVel();

		double error = consignaVel - velocidadCS;
		//derivativo como y(k)-y(k-2)
		double derivativo = error - errorAnt +  derivativoAnt;

//		if ( (integral < limiteIntegral)) //
//		if ((comandoAnt > 0 )&& (comandoAnt < MAXAVANCE))
		if((comandoAnt<MAXAVANCE) || ((errorAnt<0) && (integral>0)) )
			integral += errorAnt;
//		integral = UtilCalculos.limita(integral, -limiteIntegral, limiteIntegral);

		double comandotemp = kPAvance * error + kDAvance * derivativo + kIAvance * integral;
		double IncComando = comandotemp - comandoAnt;
		
		//Limitamos el incremento de comando
		if(comandotemp>0)
			if (comandoAnt<0)
				IncComando=-comandoAnt+Math.min(maxInc,comandotemp);
			else
				IncComando=UtilCalculos.limita(IncComando,-255,maxInc);
		else if (comandotemp<0)//el comando temp es negativo
			if(comandoAnt>0)
				IncComando=-comandoAnt+Math.max(-maxDec, comandotemp);
			else
				IncComando=UtilCalculos.limita(IncComando,-maxDec,255);
		comando=comandoAnt+IncComando;
		//Limitamos el comando maximo a aplicar
		comando=UtilCalculos.limita(comando, -25500, MAXAVANCE);
		//umbralizamos la zona muerta
		//comando=UtilCalculos.zonaMuertaCon0(comando, comandoAnt, ZonaMuerta, -1);

		int apertura=0;
		int avanceAplicado=0;
		if (comando > 0) {
			if(comandoAnt<=0) {
				menosFrena(apertura=255, 255);
//				System.err.println("========== Abrimos :"+comandoAnt+" > "+comando);
			}
			avanceAplicado=(int)comando;
//			avanceAplicado=(int)UtilCalculos.zonaMuertaCon0(comando, comandoAnt, ZonaMuerta, -1);
			Avanza(avanceAplicado);
		}
		else if (comando<0) {
			double IncCom=comando-comandoAnt;
			if(IncCom<0) {
				apertura=-(int)(IncCom*FactorFreno);
				apertura=UtilCalculos.limita(apertura, 20, 150);
				masFrena( apertura,30); /** Es un comando negativo, por lo que hay que frenar */
				System.err.println("Mas frena "+apertura);
			} else if(IncCom> maxInc){
				apertura=(int)(IncCom*FactorFreno);
				apertura=UtilCalculos.limita(apertura, 20, 150);
				menosFrena(apertura,30);
				System.err.println("menos frena "+apertura);
				apertura=-apertura;
			}
			//ente 0 y masInc no hace nada
		}
		//comando 0 no hacemos nada
		
		//guardamos todo para la iteración siguiente
		errorAnt = error;
		derivativoAnt = derivativo;
		comandoAnt = comando;
		logControl.add((double)consignaVel,velocidadCS,derivativo,integral,comandotemp,comando,(double)apertura,avanceAplicado);
		logParamPID.add(kPAvance,kIAvance,kDAvance,maxInc, FactorFreno);
	}
	
	public double getComando() {
		return comando;
	}

	/**
	 * Fija el valor de la velocidad de avance en cuentas Segundo y activa el
	 * control
	 * 
	 * @param valor
	 *            consigna en cuentas/Seg
	 */
	public void setConsignaAvanceCS(double valor) {
		if(consignaVel==valor) return; //Si es la misma consigna no cambiamos nada
		consignaVel = valor;
		controlando = true;
	}

	/** @return la consigna fijada en cuentas por segundo */
	public double getConsignaAvanceCS() {
		return consignaVel;
	}
	
	/**
	 * Fija el valor de la velocidad de avance en Metros Segundo y activa el
	 * control
	 * 
	 * @param valor
	 *            consigna en metros/seg
	 */
	public void setConsignaAvanceMS(double valor) {
		consignaVel = valor * PULSOS_METRO;
//		System.out.println("Consigna Avance " + consignaVel);
		controlando = true;
	}

	/** @return la consigna fijada en metros por segundo */
	public double getConsignaAvanceMS() {
		return consignaVel/PULSOS_METRO;
	}
	
	/**
	 * Fija el valor de la velocidad de avance en Kilometros hora y activa el
	 * control
	 * 
	 * @param valor
	 *            consigna en Kilometros hora
	 */
	public void setConsignaAvanceKH(double valor) {
		consignaVel = valor * PULSOS_METRO * 1000 / 3600;
		controlando = true;
	}

	/** @return la consigna fijada en Kilometros por hora */
	public double getConsignaAvanceKH() {
		return consignaVel/PULSOS_METRO*3600.0/1000.0;
	}
	/** @return Velocidad actual en el vehiculo en Cuentas Segundo */
	public double getVelocidadCS() {
		return velocidadCS;
	}

	/** @return Velocidad actual en el vehiculo en Metros Segundo */
	public double getVelocidadMS() {
		return velocidadMS;
	}

	/** @return Velocidad actual en el vehiculo en Kilometros Hora */
	public double getVelocidadKH() {
		return velocidadMS * 3600 / 1000;
	}


	/**
	 * Fija los parametros del controlador PID de la velocidad
	 * 
	 * @param kp Constante proporcional
	 * @param kd Constante derivativa
	 * @param ki Constante integral
	 */
	public void setKVel(double kp, double kd, double ki) {
		kPAvance = kp;
		kDAvance = kd;
		kIAvance = ki;
	}

	/**
	 * Detiene el control de la velocidad
	 */
	public void stopControlVel() {
		controlando = false;
		comandoAnt = 0;
		integral = 0;
		Avanza(0);
		masFrena(150, 60);
	}

	/**
	 * Fija el incremento máximo entre dos comandos de tracción consecutivos
	 * para evitar aceleraciones muy bruscas
	 * 
	 * @param incremento   diferencia máxima entre dos comandos
	 */
	public void setMaxIncremento(int incremento) {
		maxInc = incremento;
	}

	/** @return el valor de {@link #maxInc}, máximo incrmento de comando premitido */
	public int getMaxIncremento() {
		return maxInc;
	}
	
	public void setMaxDecremento(int decremento) {
		maxDec=decremento;
	}
	
	public int getMaxDecremento() {
		return maxDec;
	}

	public int getPaquetes() {
		return NumPaquetes;
	}

	/**
	 * @return el factorFreno
	 */
	public double getFactorFreno() {
		return FactorFreno;
	}

	/**
	 * @param factorFreno el factorFreno a establecer
	 */
	public void setFactorFreno(double factorFreno) {
		FactorFreno = factorFreno;
	}

	/**
	 * @return el kDAvance
	 */
	public double getKDAvance() {
		return kDAvance;
	}

	/**
	 * @param avance el kDAvance a establecer
	 */
	public void setKDAvance(double avance) {
		kDAvance = avance;
	}

	/**
	 * @return el kIAvance
	 */
	public double getKIAvance() {
		return kIAvance;
	}

	/**
	 * @param avance el kIAvance a establecer
	 */
	public void setKIAvance(double avance) {
		kIAvance = avance;
	}

	/**
	 * @return el kPAvance
	 */
	public double getKPAvance() {
		return kPAvance;
	}

	/**
	 * @param avance el kPAvance a establecer
	 */
	public void setKPAvance(double avance) {
		kPAvance = avance;
	}

	/**
	 * @return the controlando
	 */
	public boolean isControlando() {
		return controlando;
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

}
