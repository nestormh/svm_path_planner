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

import javax.comm.*;

import carrito.configura.Constantes;

/**
 A class that handles the details of a serial connection. Reads from one
 TextArea and writes to a second TextArea.
 Holds the state of the connection.
 */
public class CamaraConnection implements SerialPortEventListener,
        CommPortOwnershipListener {

    private boolean open;

    private SerialParameters parameters;
    private OutputStream os;
    private InputStream is;
    private KeyHandler keyHandler;

    private CommPortIdentifier portId;
    private SerialPort sPort;
    private int altura;
    private int[] angulo;

    /**
         Creates a SerialConnection object and initilizes variables passed in
         as params.

         @param parameters A SerialParameters object.
     */
    public CamaraConnection(SerialParameters parameters) {

        this.parameters = parameters;
        open = false;
        try {
            openConnection();
        } catch (SerialConnectionException e2) {

            System.out.println("Error al abrir el puerto");
            System.out.flush();
            System.exit(1);
        }
        if (isOpen()) {
            System.out.println("Puerto Abierto");
        }

    }

    public CamaraConnection(String portName) {
        parameters = new SerialParameters(portName, 9600, 0, 0, 8, 1, 0);
        try {
            openConnection();
        } catch (SerialConnectionException e2) {

            System.out.println("Error al abrir el puerto " + portName);
            System.out.flush();
            System.exit(1);
        }
        if (isOpen()) {
            System.out.println("Puerto Abierto BaudRate " + portName);
        }

    }

    public CamaraConnection(String portName,
                            int baudRate,
                            int flowControlIn,
                            int flowControlOut,
                            int databits,
                            int stopbits,
                            int parity) {

        parameters = new SerialParameters(portName, baudRate, flowControlIn,
                                          flowControlOut, databits, stopbits,
                                          parity);
        try {
            openConnection();
        } catch (SerialConnectionException e2) {

            System.out.println("Error al abrir el puerto " + portName);
            System.out.flush();

        }
        if (isOpen()) {
            System.out.println("Puerto Abierto BaudRate " + baudRate);
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
        int oldParity = sPort.getParity();
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
     * Obtiene el ángulo en altura de las cámaras
     * @return Devuelve el ángulo en altura de las cámaras
     */
    public int getAltura() {
        return altura;
    }

    /**
     * Obtiene el ángulo lateral de la cámara i
     * @return Devuelve el ángulo lateral de la cámara i
     */
    public int getAngulo(int i) {
        return angulo[i];
    }

    /**
         Handles SerialPortEvents. The two types of SerialPortEvents that this
         program is registered to listen for are DATA_AVAILABLE and BI. During
     DATA_AVAILABLE the port buffer is read until it is drained, when no more
         data is availble and 30ms has passed the method returns. When a BI
     event occurs the words BREAK RECEIVED are written to the messageAreaIn.
     */

    public void serialEvent(SerialPortEvent e) {
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
     * Envía los parámetros de control al puerto COM que van a permitir modificar
     * el ángulo lateral de la cámara
     * @param camara Identificador de la cámara
     * @param angulo Ángulo deseado
     */
    public void setAngulo(int camara, int angulo) {
        int a[] = new int[3];

        if (angulo > 255)
            angulo = 255;

        if (angulo < 0)
            angulo = 0;

        if ((camara > 255) || (camara < 0)) {
            Constantes.mensaje("La cámara " + camara + " se sale del rango");
            return;
        }

        a[0] = 255;
        a[1] = camara;
        a[2] = angulo;

        Envia(a);
    }

    /**
     * Envía los parámetros de control al puerto COM que van a permitir modificar
     * el ángulo vertical de las cámaras
     * @param angulo Ángulo deseado
     */
    public void setAltura(int angulo) {
        int a[] = new int[3];

        if (angulo > 255)
            angulo = 255;

        if (angulo < 0)
            angulo = 0;

        a[0] = 255;
        a[1] = 0;
        a[2] = angulo;

        System.out.println(a[0] + ", " + a[1] + ", " + a[2]);

        Envia(a);
    }


    private void Envia(int a[]) {

        for (int i = 0; i < 3; i++) {
            try {
                os.write(a[i]);

            } catch (Exception e) {
                System.out.println("Error al enviar");
            }
        }

    }
}

