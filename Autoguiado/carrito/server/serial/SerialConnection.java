package carrito.server.serial;
 /* @(#)SerialConnection.java	1.6 98/07/17 SMI
 *
 * Copyright (c) 1998 Sun Microsystems, Inc. All Rights Reserved.
 *
 * Sun grants you ("Licensee") a non-exclusive, royalty free, license
 * to use, modify and redistribute this software in source and binary
 * code form, provided that i) this copyright notice and license appear
 * on all copies of the software; and ii) Licensee does not utilize the
 * software in a manner which is disparaging to Sun.
 *
 * This software is provided "AS IS," without a warranty of any kind.
 * ALL EXPRESS OR IMPLIED CONDITIONS, REPRESENTATIONS AND WARRANTIES,
 * INCLUDING ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE OR NON-INFRINGEMENT, ARE HEREBY EXCLUDED. SUN AND
 * ITS LICENSORS SHALL NOT BE LIABLE FOR ANY DAMAGES SUFFERED BY
 * LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING THE
 * SOFTWARE OR ITS DERIVATIVES. IN NO EVENT WILL SUN OR ITS LICENSORS
 * BE LIABLE FOR ANY LOST REVENUE, PROFIT OR DATA, OR FOR DIRECT,
 * INDIRECT, SPECIAL, CONSEQUENTIAL, INCIDENTAL OR PUNITIVE DAMAGES,
 * HOWEVER CAUSED AND REGARDLESS OF THE THEORY OF LIABILITY, ARISING
 * OUT OF THE USE OF OR INABILITY TO USE SOFTWARE, EVEN IF SUN HAS BEEN
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
 *
 * This software is not designed or intended for use in on-line control
 * of aircraft, air traffic, aircraft navigation or aircraft
 * communications; or in the design, construction, operation or
 * maintenance of any nuclear facility. Licensee represents and
 * warrants that it will not use or redistribute the Software for such
 * purposes.
 */

import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.io.*;
import java.util.TooManyListenersException;
import java.util.Vector;

import javax.comm.*;

/**
A class that handles the details of a serial connection. Reads from one
TextArea and writes to a second TextArea.
Holds the state of the connection.
*/
public class SerialConnection implements SerialPortEventListener,
					 CommPortOwnershipListener,
                                         Runnable {

   private boolean open;

//    private TextArea messageAreaOut;
//    private TextArea messageAreaIn;
    private SerialParameters parameters;
    private OutputStream os;
    private InputStream is;
    private KeyHandler keyHandler;

    private CommPortIdentifier portId;
    private SerialPort sPort;


   private int TotalBytes = 0;
   private int volante = 32768;
   private static int avance = 0;
   private static int avanceant = 0;
   private static int lastAvance = 0;
   private int alarma = 0;
   static private int MAXBUFFER = 10000;

   private int ConsignaVolanteHigh;
   private int ConsignaVolanteLow;
   private int ConsignaFreno = 0;
   private int ConsignaSentidoFreno = 0;
   private int ConsignaVelocidad = 0;
   private int ConsignaSentidoVelocidad = 0;
   public int ConsignaVolante = 32767;
   private int ConsignaNumPasosFreno = 0;
   private int NumPasosFreno = 0; // Numero de pasos de freno que se han dado
   private int BufferComandos[][] = new int[MAXBUFFER][7];
   private int BufferRecibe[][] = new int[MAXBUFFER][3];

   private int PtroBufferRecibir = 0, PtroBufferEnviar = 0;
    private int RVolante;
    
    private static double velocidad = 0;
    private static double velocidadAnt = 0;
    private static int refresco = 200;
    
    
    // Variables del controlador PID
    private double kPAvance = 0.3;   
    private double kPFreno = 0.5;   
    private double kPDesfreno = 1.5;
    
    private double kDAvance = 0;
    private double kIAvance = 0;
    
    private static boolean controla = false;
    private static double consignaVel = 0;
    
    private static int maxInc = 20;
    private static int minInc = -40;
    
    private int comando = 0;
    private int comandoAnt = 0;
    private double derivativoAnt = 0;
    private double errorAnt = 0;
    private double integral = 0;
    
    private Vector velocidades = null;
    private static int maxVeloc = 1;
    
    private int contadorFreno = 0;        
    
    private boolean volanteBloqueado = false;

    /**
    Creates a SerialConnection object and initilizes variables passed in
    as params.
    @param parameters A SerialParameters object.
    */
    public SerialConnection( SerialParameters parameters) {

	this.parameters = parameters;
	open = false;
        try {
            openConnection();
          } catch (SerialConnectionException e2) {

              System.out.println("Error al abrir el puerto");
              System.out.flush();
              System.exit(1);
       }
       if (isOpen())
           System.out.println("Puerto Abierto");

   }
    
   public SerialConnection(int num) {
        Thread hilo = new Thread(this);
        hilo.start();
   }


      public SerialConnection(String portName) {



          parameters = new SerialParameters(portName, 9600, 0, 0, 8, 1, 0);
          try {
            openConnection();
          } catch (SerialConnectionException e2) {

              System.out.println("Error al abrir el puerto " + portName);
              System.out.flush();
              System.exit(1);
       }
       if (isOpen())
           System.out.println("Puerto Abierto BaudRate "+ portName);
       
       Thread hilo = new Thread(this);
        hilo.start();

    }

   public SerialConnection(String portName,
                              int baudRate,
                              int flowControlIn,
                              int flowControlOut,
                              int databits,
                              int stopbits,
                              int parity) {



          parameters = new SerialParameters(portName, baudRate, flowControlIn, flowControlOut, databits, stopbits, parity);
          try {
            openConnection();
          } catch (SerialConnectionException e2) {

              System.out.println("Error al abrir el puerto " + portName);
              System.out.flush();

       }
       if (isOpen())
           System.out.println("Puerto Abierto BaudRate "+ baudRate);

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
	    sPort = (SerialPort)portId.open("SerialDemo", 30000);
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
	    os = sPort.getOutputStream();
	    is = sPort.getInputStream();
	} catch (IOException e) {
	    sPort.close();
	    throw new SerialConnectionException("Error opening i/o streams");
	}

	// Create a new KeyHandler to respond to key strokes in the
	// messageAreaOut. Add the KeyHandler as a keyListener to the
	// messageAreaOut.
	keyHandler = new KeyHandler(os);
//	messageAreaOut.addKeyListener(keyHandler);

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

	// Add ownership listener to allow ownership event handling.
	portId.addPortOwnershipListener(this);

	open = true;

        sPort.disableReceiveTimeout();
    }

    /**
    Sets the connection parameters to the setting in the parameters object.
    If set fails return the parameters object to origional settings and
    throw exception.
    */
    public void setConnectionParameters() throws SerialConnectionException {

	// Save state of parameters before trying a set.
	int oldBaudRate = sPort.getBaudRate();
	int oldDatabits = sPort.getDataBits();
	int oldStopbits = sPort.getStopBits();
	int oldParity   = sPort.getParity();
	int oldFlowControl = sPort.getFlowControlMode();

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
                            TotalBytes++;
                            if ((newData == 250)) {
                                newData = is.read();
                                if (newData == 251){
                                    buffer[1]=is.read();
                                    buffer[2]=is.read();
                                    buffer[3] = is.read();
                                    buffer[4] = is.read();
                                    buffer[5] = is.read();
                                    newData = is.read();
                                    if (newData == 255) {
                                        volante = byte2entero(buffer[1],buffer[2]);


                                        if (ConsignaVolante == -1) {
                                            ConsignaVolante = volante;
                                            ConsignaVolanteLow =  volante&255;
                                            ConsignaVolanteHigh = (volante & 65280) >> 8;
                                        }
                                        if (avanceant <= buffer[3])
                                            avance = avance + buffer[3] - avanceant;
                                        else
                                           avance = avance + buffer[3] + (255 - avanceant) + 1;
                                        avanceant = buffer[3];
                                        alarma = buffer[4];
                                        NumPasosFreno = buffer[5];
                                        
                                        if (ConsignaSentidoFreno != 0)
                                            System.out.println("NumPasosFreno = " + NumPasosFreno);
                                        if (NumPasosFreno == 255) {
                                            if (ConsignaSentidoFreno == 2) {
                                                System.out.println("Frenando");
                                                if (contadorFreno > 0) {
                                                    System.out.println("ContadorFreno1 = " + contadorFreno);
                                                    if (getFreno() == 1) {
                                                        contadorFreno = 0;                                                                                                                
                                                    } else {
                                                        NumPasosFreno = 0;                                                                                                                
                                                        contadorFreno--;
                                                    }
                                                    System.out.println("ContadorFreno2 = " + contadorFreno);
                                                }
                                            } else if (ConsignaSentidoFreno == 1) {
                                                System.out.println("DesFrenando");
                                                if (contadorFreno > 0) {
                                                    System.out.println("ContadorFreno1 = " + contadorFreno);
                                                    if (getDesfreno() == 1) {
                                                        contadorFreno = 0;
                                                    } else {
                                                        NumPasosFreno = 0;
                                                        contadorFreno--;
                                                    }
                                                    System.out.println("ContadorFreno2 = " + contadorFreno);
                                                }
                                            }
                                        }
                                        
                                        if (contadorFreno == 0)
                                            volanteBloqueado = false;
                                        
                                        BufferRecibe[PtroBufferRecibir][0] = volante;
                                        BufferRecibe[PtroBufferRecibir][1] = buffer[3];
                                        BufferRecibe[PtroBufferRecibir][2] = alarma;
                                        PtroBufferRecibir = (PtroBufferRecibir++) % MAXBUFFER;

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
    A class to handle <code>KeyEvent</code>s generated by the messageAreaOut.
    When a <code>KeyEvent</code> occurs the <code>char</code> that is
    generated by the event is read, converted to an <code>int</code> and
    writen to the <code>OutputStream</code> for the port.
    */
    class KeyHandler extends KeyAdapter {
	OutputStream os;

	/**
	Creates the KeyHandler.
	@param os The OutputStream for the port.
	*/
	public KeyHandler(OutputStream os) {
	    super();
	    this.os = os;
	}

	/**
	Handles the KeyEvent.
	Gets the <code>char</char> generated by the <code>KeyEvent</code>,
	converts it to an <code>int</code>, writes it to the <code>
	OutputStream</code> for the port.
	*/
        public void keyTyped(KeyEvent evt) {


        }
    }
    /**
    Devuelve el Objeto de escritura del puerto serie

    */
    public OutputStream getOutputStream() {
        return os;

    }

    private int byte2entero(int a, int b) {

        return a*256+b;
    }

    private int byte2enteroSigno(int a, int b) {
        if (a > 129) {
            int atemp = a*256 + b -1;
            atemp = atemp^65535;

            return -1 * atemp;
        }

            return a*256+b;
}
/**
    Devuelve el numero de Bytes que hemos recibido del coche

    */
   public int getBytes() {
       return TotalBytes;
   }

/**
    Devuelve el numero de Cuentas del encoder de la tracción, lo que se ha desplazado el coche

    */

    public int getAvance() {
       return avance;
   }


    /**
    Devuelve el numero de Pasos que quedan por dar para terminar la frenada

    */

    public int getPasosFreno() {
       return NumPasosFreno;
   }

    /**
    Devuelve la posicion actual del volante

    */
 public int getVolante() {
       return volante;
   }
 /**
    Devuelve si esta pulsada la alarma de desfrenado

    */
 public int getDesfreno() {
       if ((alarma & 16) == 16)
           return 1;
       return 0;
   }
 /**
    Devuelve si esta pulsada la alarma de frenado

    */
 public int getFreno() {
       if ((alarma & 8) == 8)
           return 1;
       return 0;
   }
 /**
    Devuelve si esta pulsada la alarma del fin de carrera izquierdo

    */
 public int getIzq() {
       if ((alarma & 4) == 4)
           return 1;
       return 0;
   }
 /**
    Devuelve si esta pulsada la alarma del fin de carrera derecho

    */
public int getDer() {
       if ((alarma & 2) == 2)
           return 1;
       return 0;
   }

/**
    Devuelve si esta pulsada la alarma global del coche bajo el freno

    */
public int getAlarma() {
       if ((alarma & 1) == 1)
           return 1;
       return 0;
   }

   /**
    * Obtiene el avance anterior
    * @return Devuelve el avance anterior
    */
   public int getAvanceant() {
        return avanceant;
    }

    /**
     * Obtiene el giro del volante
     * @return Devuelve el giro del volante
     */
    public int getRVolante() {
        return RVolante;
    }

    /**
     * Obtiene el número de pasos para frenar
     * @return Devuelve el número de pasos para frenar
     */
    public int getNumPasosFreno() {
        return NumPasosFreno;
    }

    /**
     * Obtiene la consigna del volante inferior
     * @return Devuelve la consigna del volante inferior
     */
    public int getConsignaVolanteLow() {
        return ConsignaVolanteLow;
    }

    /**
     * Obtiene la consigna del volante superior
     * @return Devuelve la consigna del volante superior
     */
    public int getConsignaVolanteHigh() {
        return ConsignaVolanteHigh;
    }

    /**
     * Obtiene la consigna del volante
     * @return Devuelve la consigna del volante
     */
    public int getConsignaVolante() {
        return ConsignaVolante;
    }

    /**
     * Obtiene la consigna de la velocidad
     * @return Devuelve la consigna de la velocidad
     */
    public int getConsignaVelocidad() {
        return ConsignaVelocidad;
    }

    /**
     * Obtiene la consigna del sentido de la velocidad
     * @return Devuelve la consigna del sentido de la velocidad
     */
    public int getConsignaSentidoVelocidad() {
        return ConsignaSentidoVelocidad;
    }

    /**
     * Obtiene la consigna del sentido del freno
     * @return Devuelve la consigna del sentido del freno
     */
    public int getConsignaSentidoFreno() {
        return ConsignaSentidoFreno;
    }

    /**
     * Obtiene la consigna del número de pasos del freno
     * @return Devuelve la consigna del número de pasos del freno
     */
    public int getConsignaNumPasosFreno() {
        return ConsignaNumPasosFreno;
    }

    /**
     * Obtiene la consigna del freno
     * @return Devuelve la consigna del freno
     */
    public int getConsignaFreno() {
        return ConsignaFreno;
    }

    /**
    Fija la posicion del volante a las cuentas indicadas

    @param Angulo Numero de cuentas a las que fijar el volante
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
 a[2] = ConsignaVolanteLow =  Angulo&255;
 a[3] = ConsignaVolanteHigh = (Angulo & 65280) >> 8;
 a[4] = ConsignaFreno;
 a[5] = ConsignaSentidoFreno;
 a[6] = ConsignaVelocidad;
 a[7] = ConsignaSentidoVelocidad;
 a[8] = ConsignaNumPasosFreno = NumPasosFreno;
 /*if (NumPasosFreno != 0) {
    a[5] = ConsignaSentidoFreno;
    a[4] = ConsignaFreno;
    a[8] = ConsignaNumPasosFreno = NumPasosFreno;
 } else {
    a[5] = ConsignaSentidoFreno = 0;
    a[4] = ConsignaFreno = 0;
    a[8] = ConsignaNumPasosFreno = NumPasosFreno;
 }*/
 a[9] = 255;


 Envia(a);

}

/**
    Fija la posicion del volante a n cuentas a partir de la posicion actual

    @param Angulo Numero de cuentas a desplazar a partir de la posicion actual
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
 a[2] = ConsignaVolanteLow =  Angulo&255;
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
    Frena el coche con una Fuerza dada

    @param Fuerza La fuerza con la que se frena el coche entre 0-255
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
 a[6] = ConsignaVelocidad=0;
 a[7] = ConsignaSentidoVelocidad=0;
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
 a[2] = ConsignaVolanteLow =  volante&255;
 a[3] = ConsignaVolanteHigh = (volante & 65280) >> 8;
 a[4] = ConsignaFreno = 255;
 a[5] = ConsignaSentidoFreno = 2;
 a[6] = ConsignaVelocidad=0;
 a[7] = ConsignaSentidoVelocidad=0;
 a[8] = ConsignaNumPasosFreno = NumPasosFreno = 0;
 a[9] = 255;

 Envia(a);

}

/**
    Desfrena el coche con una Fuerza dada


    */

public void DesFrenaTotal() {
 
 int a[] = new int[10];

 contadorFreno = 10;
 volanteBloqueado = true;

 a[0] = 250;
 a[1] = 251;
 a[2] = ConsignaVolanteLow =  volante&255;
 a[3] = ConsignaVolanteHigh = (volante & 65280) >> 8;
 a[4] = ConsignaFreno = 255;
 a[5] = ConsignaSentidoFreno = 1;
 a[6] = ConsignaVelocidad=0;
 a[7] = ConsignaSentidoVelocidad=0;
 a[8] = ConsignaNumPasosFreno = NumPasosFreno = 0;
 a[9] = 255;


 Envia(a);

}

public void DesFrena(int Fuerza) {
 
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
 a[5] = ConsignaSentidoFreno = 1;
 a[6] = ConsignaVelocidad=0;
 a[7] = ConsignaSentidoVelocidad=0;
 a[8] = ConsignaNumPasosFreno = 0;
 a[9] = 255;


 Envia(a);

  ConsignaFreno = 0;
  ConsignaNumPasosFreno = 0;
  ConsignaSentidoFreno = 0;
}


/**
    Fija la velocidad hacia delante con una Fuerza dada entre 0-255


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
    Fija la velocidad hacia delante con una Fuerza dada entre 0-255


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



/**
    Da N pasos de frenado para ir aguantando el coche poco a poco.


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
    Da N pasos de desfrenado para ir aguantando el coche poco a poco.


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

    for (int i = 1; i < 6; i++)
        BufferComandos[PtroBufferEnviar][i] = a[i+4];

    BufferComandos[PtroBufferEnviar][0] = ConsignaVolante;
    BufferComandos[PtroBufferEnviar][6] = PtroBufferRecibir;

    PtroBufferEnviar = (PtroBufferEnviar++) % MAXBUFFER;


 for (int i = 0; i < 10; i++)
     try {
        os.write(a[i]);

     } catch (Exception e){
        System.out.println("Error al enviar, " + e.getMessage());
        System.out.println(e.getStackTrace());
        try {
            System.out.println("Se va a proceder a vaciar el buffer");
            os.flush();
        } catch (Exception e2) {
            System.out.println("Error al vaciar el buffer, " + e2.getMessage());            
        }
     };




}
/****
 * Recupera el historial de comandos enviados,
 * 0.- Posicion Volante
 * 1.- Fuerza de freno
 * 2.- Sentido de freno
 * 3.- Consigna velocidad
 * 4.- Sentido velocidad
 * 5.- Num Pasos Freno
 *
****/
public int getBufferEnvio(int i, int j) {
    return BufferComandos[i][j];

}

/****
 * Recupera el historial de comandos enviados,
 * 0.- Volante
 * 1.- Avance del Encoder
 * 2.- Alarmas
 *
****/
public int getBufferRecibir(int i, int j) {
    return BufferRecibe[i][j];

}

public void run() {
        
    velocidades = new Vector();
    
    while (true) {                      
        double vel = (avance - lastAvance); //* (1000 / refresco);
        lastAvance = avance;
        
        /*velocidades.add(vel);
        
        while (velocidades.size() > maxVeloc)
            velocidades.remove(0);
        
        double suma = 0;
        for (int i = 0; i < velocidades.size(); i++)
            suma += ((Double)velocidades.elementAt(i)).doubleValue();
        suma /= velocidades.size();            
        
        velocidad = suma;*/
        
        //velocidad = vel;

        double dif = Math.abs(vel - velocidadAnt);
        if (dif < 30)
            velocidad = vel;

        velocidadAnt = vel;
        
        if (controla) {
            System.out.println("Control de velocidad activo");
            System.out.println("***********************");
            System.out.println("Velocidad: " + velocidad);
            System.out.println("Diferencia: " + dif);
            System.out.println("Velocidad(m/s): " + (velocidad / 78 / (refresco / 1000)));            
            System.out.println("Velocidad(Km/h): " + (((velocidad / 78 / (refresco / 1000)) * 1000) / 3600));
            double error = consignaVel - velocidad;
            double derivativo = kDAvance * (error - errorAnt) + kDAvance * derivativoAnt;
            
            if((comandoAnt > 0 || error > 0) && (comandoAnt < 254 || error < 0))
                integral += kIAvance * errorAnt;
                    
            System.out.println("Error: " + error);                    
            
            int incremento = (int)(kPAvance * error + derivativo + integral);
            
            errorAnt = error;
            derivativoAnt = derivativo;
            
            if (incremento > maxInc)
                incremento = maxInc;
            
           if (incremento < minInc)
                incremento = minInc;
            
            System.out.println("Inc: " + incremento);                    
            
            comando += incremento;
            
            if ((comando > 0) && (comando < 255)) {
                if (incremento > 0)
                    comando += 60;
                if (incremento < 0)
                    comando += 0;
            }

            comandoAnt = comando;
            
            if (comando > 255)
                comando = 255;
    
            if (comando < 0)
                comando = 0;
                
            System.out.println("Avanzando: " + comando);
            Avanza(comando);
        }
        
        try {
            Thread.sleep(refresco);
        } catch (Exception e) {}
    }
}

public void setConsigna(double valor) {
    boolean retroceso = false;
    
    if (valor < 0) {
        retroceso = true;
        valor *= -1;
    }
    double error = valor - velocidad;
    System.out.println("Error: " + error);
    
    if (error >= 0) {                
        if (getDesfreno() == 0) {
         comando += (int)(kPDesfreno * error);

         DesFrenaPasos(comando);
         System.out.println("Desfrenando: " + comando);
        } else {
            comando += (int)(kPAvance * error);
    
            if (retroceso == true) {
                System.out.println("Retrocediendo: " + comando);
                Retrocede(comando);
            } else {
                System.out.println("Avanzando: " + comando);
                Avanza(comando);
            }
        }
    } else {
        comando += (int)(kPFreno * error);

        Avanza(0);
        FrenaPasos(comando);
    }
}

public void setConsignaAvance(double valor) {
    consignaVel = valor;
    controla = true;     
}

public double getVelocidad() {
    return velocidad;
}

public int getRefresco() {
    return refresco;
}

public void setRefresco(int valor) {
    this.refresco = valor;
}

public void setKPAvance(double valor) {
    this.kPAvance = valor;
}

public void setKPFreno(double valor) {
    this.kPFreno = valor;
}

public void setKPDesfreno(double valor) {
    this.kPDesfreno = valor;
}

public double getKPAvance() {
    return kPAvance;
}

public double getKPFreno() {
    return kPFreno;
}

public double getKPDesfreno() {
    return kPDesfreno;
}

public void setMaxInc(int val) {
    this.maxInc = val;
}

public void setMinInc(int val) {
    this.minInc = val;
}

public void stopControl() {
    controla = false;
}

public void setMaxVeloc(int val) {
    this.maxVeloc = val;
}

public void setKDAvance(double val) {
    this.kDAvance = val;
}

public void setKIAvance(double val) {
    this.kIAvance = val;
}
}



