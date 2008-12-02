//package carrito.server.serial;
/* @(#)SerialConnection.java	1.6 98/07/17 SMI
 *
 * Clase de Control del veh�culo Guistub
 */
package sibtra.controlcarro;

import gnu.io.CommPortIdentifier;
import gnu.io.CommPortOwnershipListener;
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

/**
 * A class that handles the details of a serial connection. Reads from one
 * TextArea and writes to a second TextArea. Holds the state of the connection.
 */
public class ControlCarro implements SerialPortEventListener,
		CommPortOwnershipListener {

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

	private boolean open;

	/** Punto central del volante del vehiculo */
	public final static int CARRO_CENTRO = 5280;

	/** Maximo y minimo comando en el avance **/
	public final static int MINAVANCE = 100, MAXAVANCE = 255;
	
	double TiempoTotal = 0;

	int Recibidos = 0;

	/**
	 * Numero de cuentas necesarias para alcanzar un metro
	 */
	public final static double PULSOS_METRO = 74;

	// public static final double RADIANES_POR_CUENTA =
	// 2*Math.PI/MAX_CUENTA_VOLANTE;
	public static final double RADIANES_POR_CUENTA = 0.25904573048913979374 / (5280 - 3300);

	/** Indica si la ultima vez se estaba acelerando o se estaba frenando */

	/**
	 * Indica el sentido que llevaba la aceleracion del vehiculo en la ultima
	 * instruccion
	 */

	static private int freqVel = 2;

	static double T = 0.087;

	int Cuentas[] = new int[freqVel];

	long tiempos[] = new long[freqVel];

	int indiceCuentas = 0;

	double errorAnt = 0;

	double derivativoAnt = 0;

	int comando = CARRO_CENTRO;

	double integral = 0;

	private ControlCarroSerialParameters parameters;

	private OutputStream os;

	private InputStream is;

	private CommPortIdentifier portId;

	private SerialPort sPort;

	private double velocidadCS = 0, velocidadMS = 0, velocidadKH = 0;

	private int TotalBytes = 0;

	private int volante = 32768;

	private static int avance = 0, avanceant = 0;

	private int alarma = 0;

	static private int MAXBUFFER = 10000;

	private int ConsignaVolanteHigh;

	private int ConsignaVolanteLow;

	private int ConsignaFreno = 0;

	private int ConsignaSentidoFreno = 0;

	private int ConsignaVelocidad = 0;

	private int ConsignaSentidoVelocidad = 0;

	private int ConsignaVolante = 32767;

	private int ConsignaNumPasosFreno = 0;

	private int NumPasosFreno = 0; // Numero de pasos de freno que se han dado

	private int BufferComandos[][] = new int[MAXBUFFER][5]; // Tiempo, Comando
															// volante, comando
															// velocidad, Fuerza
															// freno, Pasos
															// freno

	private double BufferRecibe[][] = new double[MAXBUFFER][5]; // Tiempo,
																// Volante,
																// cuentas,
																// velocidad,
																// Alarma

	private int PtroBufferRecibir = 0, PtroBufferEnviar = 0;

	private int RVolante;

	private int refresco = 300;

	private double kPFreno = 0.5;
	private double kPDesfreno = 1.5;
	
	// Variables del controlador PID
	private double kPAvance = 2.5;

	private double kDAvance = 0.0;

	private double kIAvance = 0.1;

	private static boolean controla = false;

	private static double consignaVel = 0;

	/**
	 * Maximo incremento permitido en el comando para evitar aceleraciones
	 * bruscas
	 */
	private int maxInc = 3;

	private int NumPaquetes = 0; /* Numero de paquetes recibidos validos */

	private long TiempoInicial = System.currentTimeMillis();

	private double comandoAnt = MINAVANCE;

	private int contadorFreno = 0;

	private boolean volanteBloqueado = false;

	private boolean capturando = false;

	private Vector captura = null;

	private String ficheroCaptura = "";

	/**
	 * Creates a SerialConnection object and initilizes variables passed in as
	 * params.
	 * 
	 * @param parameters
	 *            A SerialParameters object.
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

		// hilo = new Thread(this);
		// hilo.start();

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

		// Create a new KeyHandler to respond to key strokes in the
		// messageAreaOut. Add the KeyHandler as a keyListener to the
		// messageAreaOut.

		// messageAreaOut.addKeyListener(keyHandler);

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
		sPort.notifyOnBreakInterrupt(true);

		// Set receive timeout to allow breaking out of polling loop during
		// input handling.
		// try {
		// sPort.enableReceiveTimeout(30);
		// } catch (UnsupportedCommOperationException e) {
		// }

		// Add ownership listener to allow ownership event handling.
		portId.addPortOwnershipListener(this);

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

	/**
	 * Cierra el puerto y libera los elementos asociados
	 */
	public void closeConnection() {
		// If port is alread closed just return.
		if (!open) {
			return;
		}

		// Remove the key listener.
		// messageAreaOut.removeKeyListener(keyHandler);

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

			// Remove the ownership listener.
			portId.removePortOwnershipListener(this);
		}

		open = false;
	}

	/**
	 * Send a one second break signal.
	 */
	private void sendBreak() {
		sPort.sendBreak(1000);
	}

	/**
	 * Reports the open status of the port.
	 * 
	 * @return true if port is open, false if port is closed.
	 */
	public boolean isOpen() {
		return open;
	}

	/**
	 * Handles SerialPortEvents. The two types of SerialPortEvents that this
	 * program is registered to listen for are DATA_AVAILABLE and BI. During
	 * DATA_AVAILABLE the port buffer is read until it is drained, when no more
	 * data is availble and 30ms has passed the method returns. When a BI event
	 * occurs the words BREAK RECEIVED are written to the messageAreaIn.
	 */

	public void serialEvent(SerialPortEvent e) {
		// Create a StringBuffer and int to receive input data.

		int buffer[] = new int[100];
		int newData = 0, oldData = 0;

		// Determine type of event.
		switch (e.getEventType()) {

		// Read data until -1 is returned. If \r is received substitute
		// \n for correct newline handling.
		case SerialPortEvent.DATA_AVAILABLE:
			while (newData != -1) {

				try {
					newData = is.read();
					/*
					 * if(capturando) { captura.add(System.currentTimeMillis() -
					 * lastTime); captura.add(new Integer(newData)); lastTime =
					 * System.currentTimeMillis(); }
					 */
					TotalBytes++;
					if ((newData == 250)) {
						newData = is.read();
						if (newData == 251) {
							buffer[1] = is.read();
							buffer[2] = is.read();
							buffer[3] = is.read();
							buffer[4] = is.read();
							buffer[5] = is.read();
							newData = is.read();
							if (newData == 255) {

								volante = byte2entero(buffer[1], buffer[2]);

								if (ConsignaVolante == -1) {
									ConsignaVolante = volante;
									ConsignaVolanteLow = volante & 255;
									ConsignaVolanteHigh = (volante & 65280) >> 8;
								}

								if (avanceant <= buffer[3])
									avance = avance + buffer[3] - avanceant;
								else
									avance = avance + buffer[3]
											+ (255 - avanceant) + 1;

								Cuentas[indiceCuentas] = avance;
								tiempos[indiceCuentas] = System
										.currentTimeMillis();

								int IncCuentas = Cuentas[indiceCuentas]
										- Cuentas[(indiceCuentas + 1) % freqVel];
								velocidadCS = IncCuentas / (freqVel * T);
								velocidadMS = velocidadCS / PULSOS_METRO;
								velocidadKH = velocidadMS * 3600 / 1000;

								indiceCuentas = (indiceCuentas + 1) % freqVel;

								// System.out.println("T: " +
								// (System.currentTimeMillis() - lastPaquete) +
								// " Tmedio: " + (TiempoTotal/Recibidos));

								controlVel();

								avanceant = buffer[3];
								alarma = buffer[4];
								NumPasosFreno = buffer[5];

								if (ConsignaSentidoFreno != 0)
									System.out.println("NumPasosFreno = "
											+ NumPasosFreno);
								if (NumPasosFreno == 255) {
									if (ConsignaSentidoFreno == 2) {
										System.out.println("Frenando");
										if (contadorFreno > 0) {
											System.out
													.println("ContadorFreno1 = "
															+ contadorFreno);
											if (getFreno() == 1) {
												contadorFreno = 0;
											} else {
												NumPasosFreno = 0;
												contadorFreno--;
											}
											System.out
													.println("ContadorFreno2 = "
															+ contadorFreno);
										}
									} else if (ConsignaSentidoFreno == 1) {
										System.out.println("DesFrenando");
										if (contadorFreno > 0) {
											System.out
													.println("ContadorFreno1 = "
															+ contadorFreno);
											if (getDesfreno() == 1) {
												contadorFreno = 0;
											} else {
												NumPasosFreno = 0;
												contadorFreno--;
											}
											System.out
													.println("ContadorFreno2 = "
															+ contadorFreno);
										}
									}
								}

								if (contadorFreno == 0)
									volanteBloqueado = false;

								BufferRecibe[PtroBufferRecibir][0] = System.currentTimeMillis()	- this.TiempoInicial;
								BufferRecibe[PtroBufferRecibir][1] = volante;
								BufferRecibe[PtroBufferRecibir][2] = avance;
								BufferRecibe[PtroBufferRecibir][3] = velocidadCS;
								BufferRecibe[PtroBufferRecibir][4] = buffer[3];
								PtroBufferRecibir = (PtroBufferRecibir+1)
										% MAXBUFFER;
								NumPaquetes++;
							}
						}
					}

					if (newData == -1) {
						break;
					}

				} catch (IOException ex) {
					System.err.println(ex);
					return;
				}
			}

			break;

		}

	}

	public void ownershipChange(int type) {
	}

	/**
	 * Devuelve el Objeto de escritura del puerto serie
	 * 
	 */
	public OutputStream getOutputStream() {
		return os;

	}

	private int byte2entero(int a, int b) {

		return a * 256 + b;
	}

	private int byte2enteroSigno(int a, int b) {
		if (a > 129) {
			int atemp = a * 256 + b - 1;
			atemp = atemp ^ 65535;

			return -1 * atemp;
		}

		return a * 256 + b;
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
	 * Obtiene el giro del volante
	 * 
	 * @return Devuelve el giro del volante
	 */
	public int getRVolante() {
		return RVolante;
	}

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
	 * Obtiene la consigna de la velocidad
	 * 
	 * @return Devuelve la consigna de la velocidad
	 */
	public int getConsignaVelocidad() {
		return ConsignaVelocidad;
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

		if (volanteBloqueado)
			return;

		int a[] = new int[10];

		if (Angulo > 65535)
			Angulo = 65535;

		if (Angulo < 0)
			Angulo = 0;

		ConsignaVolante = Angulo;

		a[0] = 250;
		a[1] = 251;
		a[2] = ConsignaVolanteLow = Angulo & 255;
		a[3] = ConsignaVolanteHigh = (Angulo & 65280) >> 8;
		a[4] = ConsignaFreno;
		a[5] = ConsignaSentidoFreno;
		a[6] = ConsignaVelocidad;
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
		ConsignaVolante = Angulo;

		if (Angulo > 65535)
			Angulo = 65535;

		if (Angulo < 0)
			Angulo = 0;

		a[0] = 250;
		a[1] = 251;
		a[2] = ConsignaVolanteLow = Angulo & 255;
		a[3] = ConsignaVolanteHigh = (Angulo & 65280) >> 8;
		a[6] = ConsignaVelocidad;
		a[7] = ConsignaSentidoVelocidad;

		if (NumPasosFreno != 0) {
			a[5] = ConsignaSentidoFreno;
			a[4] = ConsignaFreno;
			a[8] = ConsignaNumPasosFreno = NumPasosFreno;
		} else {
			a[5] = ConsignaSentidoFreno = 0;
			a[4] = ConsignaFreno = 0;
			a[8] = ConsignaNumPasosFreno = NumPasosFreno;
		}

		a[9] = 255;

		Envia(a);
	}

	/**
	 * Frena el coche con una Fuerza dada
	 * 
	 * @param Fuerza
	 *            La fuerza con la que se frena el coche entre 0-255
	 */
	public void Frena(int Fuerza) {

		int a[] = new int[10];

		if (Fuerza > 255)
			Fuerza = 255;

		if (Fuerza < 0)
			Fuerza = 0;

		a[0] = 250;
		a[1] = 251;
		a[2] = ConsignaVolanteLow;
		a[3] = ConsignaVolanteHigh;
		a[4] = ConsignaFreno = Fuerza;
		a[5] = ConsignaSentidoFreno = 2;
		a[6] = ConsignaVelocidad = 0;
		a[7] = ConsignaSentidoVelocidad = 0;
		a[8] = ConsignaNumPasosFreno = 0;
		a[9] = 255;

		Envia(a);

		ConsignaFreno = 0;
		ConsignaNumPasosFreno = 0;
		ConsignaSentidoFreno = 0;
	}

	public void FrenaTotal() {

		int a[] = new int[10];

		contadorFreno = 10;
		volanteBloqueado = true;

		a[0] = 250;
		a[1] = 251;
		a[2] = ConsignaVolanteLow = volante & 255;
		a[3] = ConsignaVolanteHigh = (volante & 65280) >> 8;
		a[4] = ConsignaFreno = 255;
		a[5] = ConsignaSentidoFreno = 2;
		a[6] = ConsignaVelocidad = 0;
		a[7] = ConsignaSentidoVelocidad = 0;
		a[8] = ConsignaNumPasosFreno = NumPasosFreno = 0;
		a[9] = 255;

		Envia(a);

	}

	/**
	 * Desfrena el coche con una Fuerza dada
	 * 
	 * 
	 */

	public void DesFrenaTotal() {

		int a[] = new int[10];

		contadorFreno = 10;
		volanteBloqueado = true;

		a[0] = 250;
		a[1] = 251;
		a[2] = ConsignaVolanteLow = volante & 255;
		a[3] = ConsignaVolanteHigh = (volante & 65280) >> 8;
		a[4] = ConsignaFreno = 255;
		a[5] = ConsignaSentidoFreno = 1;
		a[6] = ConsignaVelocidad = 0;
		a[7] = ConsignaSentidoVelocidad = 0;
		a[8] = ConsignaNumPasosFreno = NumPasosFreno = 0;
		a[9] = 255;

		Envia(a);

	}

	/***************************************************************************
	 * 
	 * @param tiempo
	 */
	public void DesFrena(int tiempo) {

		int a[] = new int[10];

		if (tiempo > 255)
			tiempo = 255;

		if (tiempo < 0)
			tiempo = 0;

		a[0] = 250;
		a[1] = 251;
		a[2] = ConsignaVolanteLow;
		a[3] = ConsignaVolanteHigh;
		a[4] = ConsignaFreno = 255;
		a[5] = ConsignaSentidoFreno = 3;
		a[6] = ConsignaVelocidad = 0;
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

		if (getDesfreno() != 1) {
			DesFrena(255);
			return;
		}

		a[0] = 250;
		a[1] = 251;
		a[2] = ConsignaVolanteLow;
		a[3] = ConsignaVolanteHigh;
		a[4] = ConsignaFreno = 0;
		a[5] = ConsignaSentidoFreno = 0;
		a[6] = ConsignaVelocidad = Fuerza;
		a[7] = ConsignaSentidoVelocidad = 2;
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
		a[2] = ConsignaVolanteLow;
		a[3] = ConsignaVolanteHigh;
		a[4] = ConsignaFreno;
		a[5] = ConsignaSentidoFreno;
		a[6] = ConsignaVelocidad = Fuerza;
		a[7] = ConsignaSentidoVelocidad = 1;
		a[8] = ConsignaNumPasosFreno = 0;
		a[9] = 255;

		Envia(a);
	}

	public void masFrena(int valor, int tiempo) {
		int a[] = new int[10];

		if (valor > 255)
			valor = 255;

		if (valor < 0)
			valor = 0;

		a[0] = 250;
		a[1] = 251;
		a[2] = ConsignaVolanteLow;
		a[3] = ConsignaVolanteHigh;
		a[4] = ConsignaFreno = valor;
		a[5] = ConsignaSentidoFreno = 2;
		a[6] = ConsignaVelocidad = 0;
		a[7] = ConsignaSentidoVelocidad = 0;
		a[8] = ConsignaNumPasosFreno = tiempo;
		a[9] = 255;

		Envia(a);

		ConsignaFreno = 0;
		ConsignaNumPasosFreno = 0;
		ConsignaSentidoFreno = 0;
	}

	public void menosFrena(int valor, int tiempo) {
		int a[] = new int[10];

		if (valor > 255)
			valor = 255;

		if (valor < 0)
			valor = 0;

		a[0] = 250;
		a[1] = 251;
		a[2] = ConsignaVolanteLow;
		a[3] = ConsignaVolanteHigh;
		a[4] = ConsignaFreno = valor;
		a[5] = ConsignaSentidoFreno = 1;
		a[6] = ConsignaVelocidad = 0;
		a[7] = ConsignaSentidoVelocidad = 0;
		a[8] = ConsignaNumPasosFreno = tiempo;
		a[9] = 255;

		Envia(a);

		ConsignaFreno = 0;
		ConsignaNumPasosFreno = 0;
		ConsignaSentidoFreno = 0;
	}

	/**
	 * Da N pasos de frenado para ir aguantando el coche poco a poco.
	 * 
	 * 
	 */
	public void FrenaPasos(int Pasos) {
		int a[] = new int[10];

		if (Pasos > 255)
			Pasos = 255;

		if (Pasos < 1)
			Pasos = 1;

		a[0] = 250;
		a[1] = 251;
		a[2] = ConsignaVolanteLow;
		a[3] = ConsignaVolanteHigh;
		a[4] = ConsignaFreno = 255;
		a[5] = ConsignaSentidoFreno = 2;
		a[6] = ConsignaVelocidad = 0;
		a[7] = ConsignaSentidoVelocidad = 0;
		a[8] = ConsignaNumPasosFreno = Pasos;
		a[9] = 255;

		Envia(a);

		ConsignaFreno = 0;
		ConsignaNumPasosFreno = 0;
		ConsignaSentidoFreno = 0;
	}

	/**
	 * Da N pasos de desfrenado para ir aguantando el coche poco a poco.
	 * 
	 * 
	 */
	public void DesFrenaPasos(int Pasos) {
		int a[] = new int[10];

		if (Pasos > 255)
			Pasos = 255;

		if (Pasos < 1)
			Pasos = 1;

		a[0] = 250;
		a[1] = 251;
		a[2] = ConsignaVolanteLow;
		a[3] = ConsignaVolanteHigh;
		a[4] = ConsignaFreno = 255;
		a[5] = ConsignaSentidoFreno = 1;
		a[6] = ConsignaVelocidad = 0;
		a[7] = ConsignaSentidoVelocidad = 0;
		a[8] = ConsignaNumPasosFreno = Pasos;
		a[9] = 255;

		Envia(a);

		ConsignaFreno = 0;
		ConsignaNumPasosFreno = 0;
		ConsignaSentidoFreno = 0;
	}

	private void Envia(int a[]) {

		BufferComandos[PtroBufferEnviar][0] = (int) (System.currentTimeMillis() - this.TiempoInicial);
		BufferComandos[PtroBufferEnviar][1] = ConsignaVolante;
		BufferComandos[PtroBufferEnviar][2] = ConsignaVelocidad;
		BufferComandos[PtroBufferEnviar][3] = ConsignaFreno;
		BufferComandos[PtroBufferEnviar][4] = ConsignaNumPasosFreno;

		PtroBufferEnviar = (PtroBufferEnviar+1) % MAXBUFFER;

		
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
	 * Fuerza de freno 2.- Sentido de freno 3.- Consigna velocidad 4.- Sentido
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

		if (consignaVel == 0)
			stopControlVel();
		
		if (controla) {
			// System.out.println("Control de velocidad activo");
			// System.out.println("***********************");
			// System.out.println("Velocidad: " + velocidad);
			// System.out.println("Diferencia: " + dif);
			// System.out.println("Velocidad(m/s): " + (velocidad / 78 /
			// (refresco / 1000)));
			// System.out.println("Velocidad(Km/h): " + (((velocidad / 78 /
			// (refresco / 1000)) * 1000) / 3600));
			double error = consignaVel - velocidadCS;
			double derivativo = kDAvance * (error - errorAnt) + kDAvance
					* derivativoAnt;

			if ((comandoAnt > 0 || error > 0)
					&& (comandoAnt < 254 || error < 0))
				integral += kIAvance * errorAnt;

			// System.out.println("Error: " + error);

			errorAnt = error;
			derivativoAnt = derivativo;

			double comandotemp = kPAvance * error + derivativo + integral;
			double IncComando = comandotemp - comandoAnt;

			if (Math.abs(IncComando) > maxInc) {
				if (IncComando >= 0)
					comando = (int) (comandoAnt + maxInc);
				else
					comando = (int) (comandoAnt - maxInc);
			} else
				comando = (int) (comandoAnt + IncComando);
			if (comando >= 255)
				comando = 255;

			comandoAnt = comando;

			System.out.println("Avanzando: " + comando + " error " + (consignaVel - velocidadCS));
			Avanza(comando);
		}

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
	public void controlceCS(double valor) {
		consignaVel = valor;
		controla = true;
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

	/**
	 * Velocidad de refresco en el calculo de la velocidad
	 * 
	 * @return
	 */
	public int getRefrescoVel() {
		return refresco;
	}

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

	/**
	 * Fija el intervalo de tiempo para el calculo de la velocidad
	 * 
	 * @tiempo Tiempo de actualizaci�n de la velocidad
	 */
	public void SetRefrescoVel(int tiempo) {
		refresco = tiempo;
	}

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
