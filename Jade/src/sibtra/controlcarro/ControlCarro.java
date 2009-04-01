//package carrito.server.serial;
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

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.util.TooManyListenersException;
import java.util.Vector;

import sibtra.log.LoggerArrayDoubles;
import sibtra.log.LoggerArrayInts;
import sibtra.log.LoggerFactory;


//import com.sun.xml.internal.fastinfoset.util.CharArrayArray;

/**
 * Clase para gestionar la comunicación con el PIC que controla el carro.
 * El lazo del control del volante se cierra a bajo nivel (en el PIC) por lo que 
 * basta mandar la consigna de ángulo.
 * El control de velocidad se realiza en esta clase. Para ello hay que combinar la 
 * fuerza aplicada al motor (avance) y el freno.
 * 
 * @autor jonay
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

	/** Para saber si el puerto serial está abierto */
	private boolean open;

	/** Punto central del volante del vehiculo */
	public final static int CARRO_CENTRO = 5280;

	/** Maximo comando en el avance */
	public final static int MINAVANCE = 100;
	/** Minimo comando en el avance */
	public final static int MAXAVANCE = 255;
	
	/** Numero de cuentas necesarias para alcanzar un metro	 */
	public final static double PULSOS_METRO = 74;

	// public static final double RADIANES_POR_CUENTA =
	// 2*Math.PI/MAX_CUENTA_VOLANTE;
	/** Radianes que suponen cada cuenta del sensor del volante */
	public static final double RADIANES_POR_CUENTA = 0.25904573048913979374 / (5280 - 3300);

	//double TiempoTotal = 0;

	/** Contador de los bytes recibidos por la serial */
	//int Recibidos = 0;


	/** Indica si la ultima vez se estaba acelerando o se estaba frenando */

	/**
	 * Indica el sentido que llevaba la aceleracion del vehiculo en la ultima
	 * instruccion
	 */

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
	int comando = CARRO_CENTRO;

	/** comando anterior calculado por el PID de control de la velocidad  en {@link #controlVel()}*/
	private double comandoAnt = MINAVANCE;

	/** Integral del controlador PID de la velocidad  en {@link #controlVel()}*/
	double integral = 0;

	private ControlCarroSerialParameters parameters;

	private OutputStream os;

	private InputStream is;

	private CommPortIdentifier portId;

	private SerialPort sPort;

	/** Ultima velocidad de avance calculada en cuentas por segundo */
	private double velocidadCS = 0;
	/** Ultima velocidad de avance calculada en metros por segundo */
	private double velocidadMS = 0;
	/** Ultima velocidad de avance calculada en Kilometros por hora */
	private double velocidadKH = 0;

	/** Total de bytes recibidos desde el PIC por la serial */
	private int TotalBytes = 0;

	/** Posición del volante recibida desde el PIC */
	private int volante = 32768;

	/** Cuentas del encoder de avance de la tracción, tiene corregido el desbordamiento */
	private int avance = 0;

	/** Valora anterior recibido del encoder de tracción. Necesario para corregir desbordamiento contador */
	private int avanceant = 0;

	/** Contiene byte, recibido del PIC, con los bits correspondientes a las distintas alarmas */
	private int alarma = 0;

        
	/** Valor de la última consigna para el volante enviada.
	 * Se envía al PIC descomponiéndola en {@link #ConsignaVolanteHigh} y {@link #ConsignaVolanteLow} 
	 */
	private int ConsignaVolante = 32767;
	/* Comandos enviados hacia el coche para el microcontrolador */
	/** Uno de los 8 bytes que se envían en mensaje al PIC */
	//private int ConsignaVolanteHigh;
	/** Uno de los 8 bytes que se envían en mensaje al PIC */
	//private int ConsignaVolanteLow;
	/** Uno de los 8 bytes que se envían en mensaje al PIC */
	private int ConsignaFreno = 0;
	/** Uno de los 8 bytes que se envían en mensaje al PIC */
	private int ConsignaSentidoFreno = 0;
	/** Uno de los 8 bytes que se envían en mensaje al PIC */
	private int ComandoVelocidad = 0;
	/** Uno de los 8 bytes que se envían en mensaje al PIC */
	private int ConsignaSentidoVelocidad = 0;
	/** Uno de los 8 bytes que se envían en mensaje al PIC */
	private int ConsignaNumPasosFreno = 0;


	/** Numero de pasos de freno que se han dado */
	private int NumPasosFreno = 0;




	/**Tamaño del Buffer de Entrada/Salida de datos para el microcontrolador */
	static private int MAXBUFFER = 10000;
	/** Buffer de Comandos enviados: Tiempo, Comando volante, comando velocidad, Fuerza freno, Pasos freno */
	private int BufferComandos[][] = new int[MAXBUFFER][5]; 
	/** Buffer de mensajes recibidos: Tiempo, Volante, cuentas, velocidad,	Alarma */
	private double BufferRecibe[][] = new double[MAXBUFFER][5]; 

	/** Puntero a los buffer de envío y recepción */
	private int PtroBufferRecibir = 1, PtroBufferEnviar = 1;

	/** Logger para registrar mensaje recibidos */
	private LoggerArrayInts logMenRecibidos;
	/** Logger para registrar los mensajece enviados */
	private LoggerArrayInts logMenEnviados;
//	private int RVolante;

	
        
//	private int refresco = 300;

	/** Ganancia proporcional del PID de freno */
	private double kPFreno = 0.5;
	/** Ganancia derivativa del PID de freno */
	private double kPDesfreno = 1.5;
	
	/** Ganancia proporcional del PID de avance */
	private double kPAvance = 2.5;
	/** Ganancia defivativa del PID de avance */
	private double kDAvance = 0.2;
	/** Ganancia integral del PID de avance */
	private double kIAvance = 0.1;

	/** Para indicar si se está aplicando control de velocidad */
	private boolean controla = false;

	/** Consigna de velocidad a aplicar en cuentas por segundo */
	private static double consignaVel = 0;

	/** Maximo incremento permitido en el comando para evitar aceleraciones bruscas */
	private int maxInc = 10;
        
	/** Zona Muerta donde el motor empieza a actuar realmente 	 */
	static final int ZonaMuerta = 80;
        
	/** Numero de paquetes recibidos validos */
	private int NumPaquetes = 0;

	/** Milisegundos al crearse el objeto. Se usa para tener tiempos relativos */
	private long TiempoInicial = System.currentTimeMillis();


	/** Para llevar la cuenta de la aplicación del freno */
	private int contadorFreno = 0;

	
	/** Indica si se están guardando los datos para salvar a fichero. DEPRECADO */
	private boolean capturando = false;
	/** Vector donde se guardarán los datos a salver en fichero. DEPRECADO */
	private Vector captura = null;
	/** Nombre fichero donde se guardarán los datos. DEPRECADO */
	private String ficheroCaptura = "";

	/** Registrador del angulo y velocidad medidos al recivir cada paquete */
	private LoggerArrayDoubles logAngVel;

	double FactorFreno=20;

	/** @return devuelve entero SIN SIGNO tomando a como byte para parte alta y b como parte baja */
	static int byte2entero(int a, int b) {
		return a * 256 + b;
	}

	/** @return devuelve entero CON SIGNO tomando a como byte para parte alta y b como parte baja */
	static int byte2enteroSigno(int a, int b) {
		if (a > 129) {
			int atemp = a * 256 + b - 1;
			atemp = atemp ^ 65535;

			return -1 * atemp;
		}

		return a * 256 + b;
	}

	/**
	 * Crea la conexión serial al carro en el puerto indicado.
	 * Una vez abierto queda a la espera de los eventos de recepción de caracteres.
	 * @param portName nombre del puerto serial 
	 */
	public ControlCarro(String portName) {

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

		logAngVel=LoggerFactory.nuevoLoggerArrayDoubles(this, "carroAngVel",12);
		logAngVel.setDescripcion("Carro [Ang en Rad,Vel en m/s]");
		
		logMenRecibidos= LoggerFactory.nuevoLoggerArrayInts(this, "mensajesRecibidos",(int)(1/T)+1);
		logMenRecibidos.setDescripcion("volante,avance,(int)velocidadCS,alarma,incCuentas,incTiempo");
		logMenEnviados= LoggerFactory.nuevoLoggerArrayInts(this, "mensajesEnviados",(int)(1/T)+1);
		logMenEnviados.setDescripcion("ConsignaVolante,ComandoVelocidad,ConsignaFreno,ConsignaNumPasosFreno");
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
			os = sPort.getOutputStream();
			is = sPort.getInputStream();
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
				os.close();
				is.close();
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
				newData = is.read();
				/*
				 * if(capturando) { captura.add(System.currentTimeMillis() -
				 * lastTime); captura.add(new Integer(newData)); lastTime =
				 * System.currentTimeMillis(); }
				 */
				TotalBytes++;
				if (newData == -1) {
					break;
				}
				if (newData!=250)
					continue;
				//newdata vale 250
				newData = is.read();
				if (newData != 251)
					continue; //TODO faltaría considerar el caso de varios 255 seguidos
				//newdata vale 251
				//Ya tenemos la cabecera de un mensaje válido
				//leemos los datos del mensaje en buffer
				buffer[1] = is.read();
				buffer[2] = is.read();
				buffer[3] = is.read();
				buffer[4] = is.read();
				buffer[5] = is.read();
				newData = is.read();
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
		
		volante = byte2entero(buffer[1], buffer[2]);

		//si no hay consigna de volante, dejamos el volante como está
		if (ConsignaVolante == -1) {
			ConsignaVolante = volante;
//			ConsignaVolanteLow = volante & 255;
//			ConsignaVolanteHigh = (volante & 65280) >> 8;
		}

		//controlamos desbordamiento contador de avance en el PIC
		if (avanceant <= buffer[3])
			//no se ha producido
			avance = avance + (buffer[3] - avanceant);
		else
			//se ha producido
			avance = avance + (buffer[3] + (255 - avanceant) + 1);
		avanceant = buffer[3];  //preparamos para siguiente iteración

		Cuentas[indiceCuentas] = avance;
		tiempos[indiceCuentas] = System.currentTimeMillis();

//		int IncCuentas = Cuentas[indiceCuentas]
//		                         - Cuentas[(indiceCuentas + 1) % freqVel];
//		velocidadCS = 1000.0 * (double)IncCuentas / 
//			(double)((System.currentTimeMillis()-this.TiempoInicial) - BufferRecibe[PtroBufferRecibir-1][0]);  
		/* (freqVel * T); */
		//TODO forma que creo que es más correcto de hacerlo
			int  incCuentas=Cuentas[indiceCuentas]
			                         - Cuentas[(indiceCuentas + 1) % freqVel];
			long incTiempo=tiempos[indiceCuentas]
			                         - tiempos[(indiceCuentas + 1) % freqVel];
			velocidadCS=1000.0 *(double) incCuentas / (double)incTiempo;
//			if(velocidadCS!=velCS) {
//				System.err.println("Diferencia calculo velocidad:"+velocidadCS+"<>"+velCS);				
//			}
		
		
		indiceCuentas = (indiceCuentas + 1) % freqVel; //preparamos siguiente iteración

		velocidadMS = velocidadCS / PULSOS_METRO;
		velocidadKH = velocidadMS * 3600 / 1000;

		//apuntamos angulo volante en radianes y velocidad en m/sg
		logAngVel.add((volante - CARRO_CENTRO) * RADIANES_POR_CUENTA,velocidadMS);

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


//		System.currentTimeMillis();
		//TODO sustituir estos buffes por loggers
		BufferRecibe[PtroBufferRecibir][0] = System.currentTimeMillis()	- this.TiempoInicial;
		BufferRecibe[PtroBufferRecibir][1] = volante;
		BufferRecibe[PtroBufferRecibir][2] = avance;
		BufferRecibe[PtroBufferRecibir][3] = velocidadCS;
		BufferRecibe[PtroBufferRecibir][4] = alarma;
		PtroBufferRecibir = (PtroBufferRecibir+1)% MAXBUFFER;
		NumPaquetes++;
		logMenRecibidos.add(volante,avance,(int)velocidadCS,alarma,incCuentas,(int)incTiempo);
	}
	
	/** @return Objeto de escritura del puerto serie */
	public OutputStream getOutputStream() {
		return os;

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
	 * Devuelve la posicion actual del volante
	 * 
	 */
	public int getVolante() {
		return volante;
	}

	/**
	 * @return Devuelve el angulo del volante en radianes respecto al centro (+
	 *         izquierda, - derecha).
	 */
	public double getAnguloVolante() {
		return (volante - CARRO_CENTRO) * RADIANES_POR_CUENTA;
	}

	/**
	 * @return Devuelve el angulo del volante en grados respecto al centro (+
	 *         izquierda, - derecha).
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

//	/**
//	 * Obtiene el giro del volante
//	 * 
//	 * @return Devuelve el giro del volante
//	 */
//	public int getRVolante() {
//		return RVolante;
//	}

	/**
	 * Obtiene la consigna del volante
	 * 
	 * @return Devuelve la consigna del volante
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
		return (ConsignaVolante - CARRO_CENTRO) * RADIANES_POR_CUENTA;
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
	 * @param comandoVolante
	 *            angulo deseado en radianes desde el centro (+izquierda, -
	 *            derecha)
	 */
	public void setAnguloVolante(double comandoVolante) {
		setVolante((int) Math.floor(comandoVolante / RADIANES_POR_CUENTA)
				+ CARRO_CENTRO);
	}

	/**
	 * Fija la posicion del volante a las cuentas indicadas
	 * 
	 * @param Angulo
	 *            Numero de cuentas a las que fijar el volante
	 */
	public void setVolante(int Angulo) {



		int a[] = new int[10];

		if (Angulo > 65535)
			Angulo = 65535;

		if (Angulo < 0)
			Angulo = 0;

		ConsignaVolante = Angulo;

		a[0] = 250;
		a[1] = 251;
		a[2] = ConsignaVolante & 255;
		a[3] = (ConsignaVolante & 65280) >> 8;
		a[4] = ConsignaFreno;
		a[5] = ConsignaSentidoFreno=4;
		a[6] = ComandoVelocidad;
		a[7] = ConsignaSentidoVelocidad;
		a[8] = ConsignaNumPasosFreno = NumPasosFreno;
		/*
		 * if (NumPasosFreno != 0) { a[5] = ConsignaSentidoFreno; a[4] =
		 * ConsignaFreno; a[8] = ConsignaNumPasosFreno = NumPasosFreno; } else {
		 * a[5] = ConsignaSentidoFreno = 0; a[4] = ConsignaFreno = 0; a[8] =
		 * ConsignaNumPasosFreno = NumPasosFreno; }
		 */
		a[9] = 255;

		Envia(a);

	}

	/**
	 * Fija la posicion del volante a n cuentas a partir de la posicion actual
	 * 
	 * @param Angulo
	 *            Numero de cuentas a desplazar a partir de la posicion actual
	 */
	public void setRVolante(int Angulo) {
		int a[] = new int[10];

		Angulo = ConsignaVolante + Angulo;

		if (Angulo > 65535)
			Angulo = 65535;
		if (Angulo < 0)
			Angulo = 0;
		ConsignaVolante = Angulo;

		a[0] = 250;
		a[1] = 251;
//		a[2] = ConsignaVolanteLow = Angulo & 255;
//		a[3] = ConsignaVolanteHigh = (Angulo & 65280) >> 8;
		a[2] = ConsignaVolante & 255;
		a[3] = (ConsignaVolante & 65280) >> 8;
		a[6] = ComandoVelocidad;
		a[7] = ConsignaSentidoVelocidad;
		a[5] = ConsignaSentidoFreno=4;
		a[4] = ConsignaFreno;
		a[8] = ConsignaNumPasosFreno = NumPasosFreno;

		a[9] = 255;

		Envia(a);
	}




	/***************************************************************************
	 * 
	 * @param tiempo 0 maximo tiempo, 255 minimo tiempo
	 */
	public void DesFrena(int tiempo) {

		int a[] = new int[10];

		if (tiempo > 255)
			tiempo = 255;

		if (tiempo < 0)
			tiempo = 0;
		
		tiempo = 255 - tiempo;

		a[0] = 250;
		a[1] = 251;
//		a[2] = ConsignaVolanteLow;
//		a[3] = ConsignaVolanteHigh;
		a[2] = ConsignaVolante & 255;
		a[3] = (ConsignaVolante & 65280) >> 8;
		a[4] = ConsignaFreno = 255;
		a[5] = ConsignaSentidoFreno = 3;
		a[6] = ComandoVelocidad = 0;
		a[7] = ConsignaSentidoVelocidad = 0;
		a[8] = ConsignaNumPasosFreno = tiempo;
		a[9] = 255;

		Envia(a);

		ConsignaFreno = 0;
		ConsignaNumPasosFreno = 0;
		ConsignaSentidoFreno = 0;
	}

	/**
	 * Fija la velocidad hacia delante con una Fuerza dada entre 0-255
	 */

	public void Avanza(int Fuerza) {

		int a[] = new int[10];

		if (Fuerza > 255)
			Fuerza = 255;

		if (Fuerza < 0)
			Fuerza = 0;

		/*if (getDesfreno() != 1) {
			DesFrena(255);
			return;
		}*/

		a[0] = 250;
		a[1] = 251;
//		a[2] = ConsignaVolanteLow;
//		a[3] = ConsignaVolanteHigh;
		a[2] = ConsignaVolante & 255;
		a[3] = (ConsignaVolante & 65280) >> 8;
		a[4] = ConsignaFreno = 4;  //dejar el freno como está
		a[5] = ConsignaSentidoFreno = 0;
		a[6] = ComandoVelocidad = Fuerza;
		a[7] = ConsignaSentidoVelocidad = 2;  //para alante
		a[8] = ConsignaNumPasosFreno = 0;
		a[9] = 255;

		Envia(a);
	}

	/**
	 * Fija la velocidad hacia delante con una Fuerza dada entre 0-255
	 * 
	 * 
	 */

	public void Retrocede(int Fuerza) {

		int a[] = new int[10];

		if (Fuerza > 255)
			Fuerza = 255;

		if (Fuerza < 0)
			Fuerza = 0;

		if (getDesfreno() != 1) {
			DesFrena(255);
			return;
		}

		a[0] = 250;
		a[1] = 251;
//		a[2] = ConsignaVolanteLow;
//		a[3] = ConsignaVolanteHigh;
		a[2] = ConsignaVolante & 255;
		a[3] = (ConsignaVolante & 65280) >> 8;
		a[4] = ConsignaFreno;
		a[5] = ConsignaSentidoFreno=4;
		a[6] = ComandoVelocidad = Fuerza;
		a[7] = ConsignaSentidoVelocidad = 1;  //para atrás
		a[8] = ConsignaNumPasosFreno = 0;
		a[9] = 255;

		Envia(a);
	}

	/**
	 * Aumenta el frenado del sistema
	 * @param valor apertura de la valvula
	 * @param tiempo tiempo de esta apertura
	 */
	public void masFrena(int valor, int tiempo) {
		int a[] = new int[10];

		if (valor > 255)
			valor = 255;

		if (valor < 0)
			valor = 0;
		
		if (tiempo > 255)
			tiempo = 255;
		
		if (tiempo < 0)
			tiempo = 0;
		
		tiempo = 255 - tiempo;

		a[0] = 250;
		a[1] = 251;
//		a[2] = ConsignaVolanteLow;
//		a[3] = ConsignaVolanteHigh;
		a[2] = ConsignaVolante & 255;
		a[3] = (ConsignaVolante & 65280) >> 8;
		a[4] = ConsignaFreno = valor;
		a[5] = ConsignaSentidoFreno = 2;  //Abrir valvula de frenar
		a[6] = ComandoVelocidad = 0;
		a[7] = ConsignaSentidoVelocidad = 0;
		a[8] = ConsignaNumPasosFreno = tiempo;
		a[9] = 255;

		Envia(a);

		ConsignaFreno = 0;
		ConsignaNumPasosFreno = 0;
		ConsignaSentidoFreno = 0;
	}
/**
 * Disminuye la presion del frenado
 * @param valor Apertura de la electrovalvula
 * @param tiempo tiempo de apertura
 */
	public void menosFrena(int valor, int tiempo) {
		int a[] = new int[10];

		if (valor > 255)
			valor = 255;

		if (valor < 0)
			valor = 0;

		if (tiempo > 255)
			tiempo = 255;
		
		if (tiempo < 0)
			tiempo = 0;
		
		tiempo = 255 - tiempo;

		
		a[0] = 250;
		a[1] = 251;
//		a[2] = ConsignaVolanteLow;
//		a[3] = ConsignaVolanteHigh;
		a[2] = ConsignaVolante & 255;
		a[3] = (ConsignaVolante & 65280) >> 8;
		a[4] = ConsignaFreno = valor;
		a[5] = ConsignaSentidoFreno = 1; //abrir la valvula de desfrenar
		a[6] = ComandoVelocidad = 0;
		a[7] = ConsignaSentidoVelocidad = 0;
		a[8] = ConsignaNumPasosFreno = tiempo;
		a[9] = 255;

		Envia(a);

		ConsignaFreno = 0;
		ConsignaNumPasosFreno = 0;
		ConsignaSentidoFreno = 0;
	}


	private void Envia(int a[]) {
		//TODO sustituir estos buffes por loggers
		BufferComandos[PtroBufferEnviar][0] = (int) (System.currentTimeMillis() - this.TiempoInicial);
		BufferComandos[PtroBufferEnviar][1] = ConsignaVolante;
		BufferComandos[PtroBufferEnviar][2] = ComandoVelocidad;
		BufferComandos[PtroBufferEnviar][3] = ConsignaFreno;
		BufferComandos[PtroBufferEnviar][4] = ConsignaNumPasosFreno;

		PtroBufferEnviar = (PtroBufferEnviar+1) % MAXBUFFER;
		
		logMenEnviados.add(ConsignaVolante,ComandoVelocidad,ConsignaFreno,ConsignaNumPasosFreno);

		
		for (int i = 0; i < 10; i++)
			try {
				os.write(a[i]);

			} catch (Exception e) {
				System.out.println("Error al enviar, " + e.getMessage());
				System.out.println(e.getStackTrace());
				try {
					System.out.println("Se va a proceder a vaciar el buffer");
					os.flush();
				} catch (Exception e2) {
					System.out.println("Error al vaciar el buffer, "
							+ e2.getMessage());
				}
			}
		;

	}

	/***************************************************************************
	 * Recupera el historial de comandos enviados, 0.- Posicion Volante 1.-
	 * Fuerza de freno 2.- Sentido de freno 3.-  4.- Sentido
	 * velocidad 5.- Num Pasos Freno
	 * 
	 **************************************************************************/
	public int getBufferEnvio(int i, int j) {
		return BufferComandos[i][j];

	}

	/***************************************************************************
	 * Recupera el historial de comandos enviados, 0.- Volante 1.- Avance del
	 * Encoder 2.- Alarmas
	 * 
	 **************************************************************************/
	public double getBufferRecibir(int i, int j) {
		return BufferRecibe[i][j];

	}

	/**
	 * Funci�n para el calculo de la velocidad a partir del encoder incluido en
	 * el vehiculo.
	 */
	public void controlVel() {

		
		if (!controla) 
			return;

		if (consignaVel == 0)
			stopControlVel();


		// System.out.println("Control de velocidad activo");
		// System.out.println("***********************");
		// System.out.println("Velocidad: " + velocidad);
		// System.out.println("Diferencia: " + dif);
		// System.out.println("Velocidad(m/s): " + (velocidad / 78 /
		// (refresco / 1000)));
		// System.out.println("Velocidad(Km/h): " + (((velocidad / 78 /
		// (refresco / 1000)) * 1000) / 3600));
		double error = consignaVel - velocidadCS;
		//TODO entender esta acción
//		double derivativo = kDAvance * (error - errorAnt) + kDAvance* derivativoAnt;
		double derivativo = kDAvance * (error - errorAnt +  derivativoAnt);

		if ((comandoAnt > 0 )&& (comandoAnt < 254 ))
//			integral += kIAvance * errorAnt;
			integral += errorAnt;
		// System.out.println("Error: " + error);

		errorAnt = error;
		derivativoAnt = derivativo;

		double comandotemp = kPAvance * error + derivativo + kIAvance*integral;
		double IncComando = comandotemp - comandoAnt;

		if (Math.abs(IncComando) > maxInc) {
			if (IncComando >= 0) {
				comando = (int) (comandoAnt + maxInc);
				if ((comando > 0) && (comando < ZonaMuerta))
					comando = ZonaMuerta;
			} else {
				comando = (int) (comandoAnt - maxInc);
				if ((comando > 0) && (comando < ZonaMuerta))
					comando = 0;
			}
		} else
			comando = (int) (comandoAnt + IncComando);



		if (comando >= 255)
			comando = 255;
		if (comando <= -255)
			comando = -255;


//		System.out.println("Avanzando: " + comando + " error " + (consignaVel - velocidadCS));
		if (comando >= 0) {
			if(comandoAnt<=0) {
				menosFrena(255, 255);
//				DesFrena(255);
				System.err.
				
				
				
				
				println("========== Abrimos :"+comandoAnt+" > "+comando);
			}
			Avanza(comando);
		}
		else {
//			double IncCom=comando-comandoAnt;
//			if(IncCom<=0) {
//				int apertura=-(int)(IncCom*FactorFreno);
//				if(apertura>150)
//					apertura=150;
//				masFrena( apertura,20); /** Es un comando negativo, por lo que hay que frenar */
//				System.err.println("Mas frena "+apertura);
//
//			} else {
//				int apertura=(int)(IncCom*FactorFreno);
//				if(apertura>150) apertura=150;
//				menosFrena(apertura,20);
//				System.err.println("menos frena "+apertura);
//			}
		}
		comandoAnt = comando;
	}
	
	public int getComando() {
		return comando;
	}

	/*
	 * public void setConsigna(double valor) { boolean retroceso = false;
	 * 
	 * if (valor < 0) { retroceso = true; valor *= -1; } double error = valor -
	 * velocidad; System.out.println("Error: " + error);
	 * 
	 * if (error >= 0) { if (getDesfreno() == 0) { comando += (int)(kPDesfreno *
	 * error);
	 * 
	 * DesFrenaPasos(comando); System.out.println("Desfrenando: " + comando); }
	 * else { comando += (int)(kPAvance * error);
	 * 
	 * if (retroceso == true) { System.out.println("Retrocediendo: " + comando);
	 * Retrocede(comando); } else { //System.out.println("Avanzando: " +
	 * comando); Avanza(comando); } } } else { comando += (int)(kPFreno *
	 * error);
	 * 
	 * Avanza(0); FrenaPasos(comando); } }
	 */
	/**
	 * Fija el valor de la velocidad de avance en cuentas Segundo y activa el
	 * control
	 * 
	 * @param valor
	 *            consigna en cuentas/Seg
	 */
	public void setConsignaAvanceCS(double valor) {
		consignaVel = valor;
		controla = true;
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
		System.out.println("Consigna Avance " + consignaVel);
		controla = true;
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
		controla = true;
	}

	/** @return la consigna fijada en Kilometros por hora */
	public double getConsignaAvanceKH() {
		return consignaVel/PULSOS_METRO*3600.0/1000.0;
	}
	/**
	 * Velocidad actual en el vehiculo en Cuentas Segundo
	 * 
	 * @return
	 */
	public double getVelocidadCS() {
		return velocidadCS;
	}

	/**
	 * Velocidad actual en el vehiculo en Metros Segundo
	 * 
	 * @return
	 */
	public double getVelocidadMS() {
		return velocidadMS;
	}

	/**
	 * Velocidad actual en el vehiculo en Kilometros Hora
	 * 
	 * @return
	 */
	public double getVelocidadKH() {
		return velocidadKH;
	}

//	/**
//	 * Velocidad de refresco en el calculo de la velocidad
//	 * 
//	 * @return
//	 */
//	public int getRefrescoVel() {
//		return refresco;
//	}

	/**
	 * Fija los parametros del controlador PID de la velocidad
	 * 
	 * @param kp
	 *            Constante proporcional
	 * @param kd
	 *            Constante derivativa
	 * @param ki
	 *            Constante integral
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
		controla = false;
		comandoAnt = MINAVANCE;
		Avanza(0);
	}

	/**
	 * Guarda los datos del veh�culo en un fichero
	 * 
	 * @param fichero
	 */
	public void startCaptura(String fichero) {
		if (!capturando) {
			captura = new Vector();
			ficheroCaptura = fichero;
			capturando = true;
		}
	}

	/**
	 * Termina de Guardar los datos vuelca al fichero y cierra
	 */
	public void stopCaptura() {
		if (capturando) {
			capturando = false;
			try {
				File fich = new File(ficheroCaptura);
				ObjectOutputStream oos = new ObjectOutputStream(
						new FileOutputStream(fich, false));
				for (int i = 0; i < captura.size(); i++) {
					oos.write(((Integer) captura.elementAt(i)).intValue());
				}
				oos.close();
			} catch (IOException ioe) {
				System.err.println("No se pudo abrir el flujo de datos: "
						+ ioe.getMessage());
			}
		}
	}

//	/**
//	 * Fija el intervalo de tiempo para el calculo de la velocidad
//	 * 
//	 * @tiempo Tiempo de actualizaci�n de la velocidad
//	 */
//	public void SetRefrescoVel(int tiempo) {
//		refresco = tiempo;
//	}

	/**
	 * Fija el incremento m�ximo entre dos comandos de tracci�n consecutivos
	 * para evitar aceleraciones muy bruscas
	 * 
	 * @param incremento
	 *            diferencia m�xima entre dos comandos
	 */
	public void setMaxIncremento(int incremento) {
		maxInc = incremento;
	}

	public int getPaquetes() {
		return NumPaquetes;
	}

	public int getPtroBufferEnvio() {
		return this.PtroBufferEnviar;
	}

	public int getPtroBufferRecibir() {
		return this.PtroBufferRecibir;
	}
	public void ClearBuffer() {
		PtroBufferRecibir=0;
		PtroBufferEnviar = 0;
	}
}
